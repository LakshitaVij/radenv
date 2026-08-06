"""
agent_episode.py

Runs one computer-use evaluation episode: Gemini Pro (via OpenRouter,
OpenAI-compatible API) drives a real browser (via Playwright) through the
actual task - look at a real chest X-ray, navigate into the real OpenEMR
instance for that patient's visit, read the real clinical context/history,
and select the action(s) it believes are correct from our real action
hierarchy (Procedures -> Configuration -> Configure Orders and Results).

This is a literal computer-use loop: screenshot -> model decides one tool
call -> executed via Playwright -> repeat. Custom small toolset
(click/type/key/scroll), not the full OS-level "computer" tool, since only
browser control is needed.

Every step is logged to episodes/<patient>_<visit>/log.jsonl - this is the
log of record for the whole episode, including which patient/visit each
action was for and which hierarchy node(s) got selected. Necessary because
OpenEMR's "Configure Orders and Results" screen (where selection actually
happens - see plan doc for why the real per-patient Procedure Order screen
doesn't work) has no patient/encounter concept at all, so nothing about
"the agent picked X for this visit" can be read back out of OpenEMR itself.
"""

import argparse
import base64
import json
import time
from pathlib import Path

from openai import OpenAI
from playwright.sync_api import sync_playwright

# OpenRouter key: a local .openrouter_key file next to this script, created
# directly by the user in their own terminal - never pasted into chat, so
# it never lands in the conversation transcript.
_LOCAL_KEY_FILE = Path(__file__).parent / ".openrouter_key"
if not _LOCAL_KEY_FILE.exists():
    raise SystemExit(
        f"Missing {_LOCAL_KEY_FILE} - create it yourself (echo \"sk-or-...\" > {_LOCAL_KEY_FILE}) "
        "with your OpenRouter API key before running an episode."
    )
OPENROUTER_API_KEY = _LOCAL_KEY_FILE.read_text().strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

FRONTEND_URL = "http://localhost:5173"
OPENEMR_BASE = "https://localhost:9300"
OPENEMR_USER = "admin"
OPENEMR_PASS = "pass"
MODEL = "google/gemini-3.1-pro-preview"  # newest multimodal Gemini Pro on OpenRouter (resolved from the ~google/gemini-pro-latest alias)
MAX_STEPS_DEFAULT = 30
VIEWPORT = {"width": 1600, "height": 1000}

EPISODES_DIR = Path(__file__).parent / "episodes"

# Static PatientID -> OpenEMR internal pid, matches xray-viewer-frontend's
# App.jsx OPENEMR_PID_MAP - confirmed against patient_data, stable since
# all 10 gold-set patients were seeded once in this fixed order.
OPENEMR_PID_MAP = {
    "GRDN003M0HXJ2YHT": 1,
    "GRDN004RP5BFHE0T": 2,
    "GRDN006MCYEVYFE8": 3,
    "GRDN006VFINFR877": 4,
    "GRDN006WOMKG4AVY": 5,
    "GRDN007ELLXXY3OJ": 6,
    "GRDN00A5ZYWHZY7D": 7,
    "GRDN00BE97VCHW4E": 8,
    "GRDN00BJ2R8QRO0L": 9,
    "GRDN00BKCEOKC7S1": 10,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click at a point in the current screenshot. Use Gemini's native "
                "normalized coordinate space: x and y each range from 0 to 1000, "
                "representing position as a fraction of the image width/height "
                "(0=left/top edge, 1000=right/bottom edge) - NOT raw pixel counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "0-1000, fraction of image width."},
                    "y": {"type": "integer", "description": "0-1000, fraction of image height."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into whatever element is currently focused.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a single key, e.g. 'Enter', 'Tab', 'Escape'.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": (
                "Scroll/zoom the page. Use this to actually interact with "
                "the X-ray image (zoom in/out, pan) as well as to scroll "
                "long pages like Configure Orders and Results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {"type": "integer", "description": "Horizontal scroll delta."},
                    "dy": {"type": "integer", "description": "Vertical scroll delta (negative = up/zoom in convention if the viewer supports wheel-zoom)."},
                },
                "required": ["dx", "dy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_action",
            "description": (
                "Record your final clinical decision for this visit. Call "
                "this AFTER you have navigated to Configure Orders and "
                "Results and identified the correct action(s) to select "
                "for this specific patient's visit. This is the log of "
                "record for grading, separate from any on-screen click - "
                "use the EXACT text labels as they appear on screen. You "
                "may call this more than once if multiple actions apply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level_1": {"type": "string", "description": "Exact Level-1 category label, e.g. 'Order CT Chest'."},
                    "level_2": {"type": "string", "description": "Exact Level-2 action label, e.g. 'Obtain CT chest'."},
                    "level_3": {"type": "string", "description": "Exact Level-3 option label if one applies, else empty string."},
                    "reasoning": {"type": "string", "description": "Why this action is correct given the X-ray and clinical context."},
                },
                "required": ["level_1", "level_2", "level_3", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this when the episode is complete - all correct action(s) have been recorded via select_action.",
            "parameters": {
                "type": "object",
                "properties": {"notes": {"type": "string"}},
                "required": ["notes"],
            },
        },
    },
]

SYSTEM_PROMPT = """You have access to a real web browser via tool calls.

Your job is to review this patient's chest X-ray and, using OpenEMR (the hospital's real \
EHR), do two things for this encounter: (1) document your clinical report - findings, \
impression, and your recommended follow-up - directly in the patient's encounter notes, \
and (2) go to Procedures -> Configuration -> Configure Orders and Results to select the \
correct action(s) for this patient.

Structure your written note with the exact section labels "Findings:", "Impression:", and \
"Follow-up:" - each on its own, written out literally - so your findings, impression, and \
follow-up recommendation are each clearly separated.

To log into OpenEMR, use these credentials: username "{user}" / password "{password}".

OpenEMR may also have this patient's vitals/clinical context, visit history, and prior \
X-ray reports available - use your discretion on whether and how to use them.

Call finish once you've recorded your decision.

click(x, y) uses coordinates normalized 0-1000 (a fraction of the current screenshot's \
width/height), not raw pixels.

Take one tool action per turn.
"""


def log_step(log_path: Path, entry: dict) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def screenshot_b64(page) -> str:
    png_bytes = page.screenshot()
    return base64.b64encode(png_bytes).decode("utf-8")


def run_episode(patient_id: str, visit_date: str, max_steps: int, headless: bool) -> Path:
    pid = OPENEMR_PID_MAP.get(patient_id)
    if pid is None:
        raise ValueError(f"Unknown patient_id {patient_id}, not in OPENEMR_PID_MAP")

    episode_dir = EPISODES_DIR / f"{patient_id}_{visit_date}_{int(time.time())}"
    screenshots_dir = episode_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    log_path = episode_dir / "log.jsonl"

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    system_prompt = SYSTEM_PROMPT.format(
        user=OPENEMR_USER, password=OPENEMR_PASS, patient_id=patient_id, pid=pid, visit_date=visit_date,
    )
    log_step(log_path, {"type": "episode_start", "patient_id": patient_id, "visit_date": visit_date, "pid": pid})

    selected_actions = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(viewport=VIEWPORT, ignore_https_errors=True)
        page = context.new_page()
        page.goto(FRONTEND_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)

        # Deterministically select the patient + visit for this episode -
        # we already know exactly which visit this episode is about (CLI
        # args), so the harness picks the dropdowns rather than spending
        # agent turns on trivial navigation the episode design doesn't
        # actually need to test.
        selects = page.locator("select")
        selects.nth(0).select_option(patient_id)
        page.wait_for_timeout(1000)
        selects.nth(1).select_option(label=visit_date)

        # A fixed 1500ms wait wasn't always enough for the X-ray <img>
        # elements to finish fetching/rendering (thumbnail endpoint does a
        # cold DICOM->PNG conversion) - a real run once handed the agent a
        # genuinely blank image and it (correctly) reported "insufficient
        # evidence," which is right behavior on wrong input. Wait for every
        # <img> on the page to actually finish loading instead of guessing
        # a timeout.
        try:
            page.wait_for_function(
                "Array.from(document.querySelectorAll('img')).every(img => img.complete && img.naturalWidth > 0)",
                timeout=15000,
            )
        except Exception:  # noqa: BLE001 - log and continue rather than abort the episode
            log_step(log_path, {"type": "setup", "note": "WARNING: X-ray image(s) did not finish loading within 15s"})
        page.wait_for_timeout(500)
        log_step(log_path, {"type": "setup", "note": "patient/visit pre-selected by harness"})

        messages = [{"role": "system", "content": system_prompt}]

        for step in range(1, max_steps + 1):
            shot_path = screenshots_dir / f"step_{step:03d}.png"
            page.screenshot(path=str(shot_path))
            img_b64 = base64.b64encode(shot_path.read_bytes()).decode("utf-8")

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Step {step}. Current screenshot:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            })

            # Occasionally the provider returns a 200 with an empty/missing
            # choices list (upstream hiccup, not something our code causes) -
            # retry a couple times rather than crashing the whole episode
            # over one bad response.
            #
            # One specific known cause: OpenRouter/Gemini 3's "Corrupted
            # thought signature" error - a real, still-unresolved issue
            # across many tools using Gemini reasoning + tool calls via a
            # proxy (see e.g. github.com/google-gemini/gemini-cli#13467).
            # Confirmed via a real run that retrying the IDENTICAL message
            # history fails identically every time - the corruption lives in
            # a prior assistant turn's reasoning/signature data already in
            # `messages`, so blindly retrying is guaranteed to fail. This is
            # a mitigation, not a confirmed root-cause fix (root cause isn't
            # publicly nailed down yet, even in tracked upstream issues):
            # on this specific error, strip reasoning/signature-carrying
            # extra fields from prior assistant turns before retrying, so
            # the retry has an actual chance instead of none.
            choice = None
            last_error = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="required",
                        max_tokens=2048,  # this model spends real tokens on hidden reasoning before the tool call
                    )
                    if response.choices:
                        choice = response.choices[0].message
                        break
                    last_error = f"empty choices in response: {response!r}"
                    response_error = getattr(response, "error", None)
                    if response_error and "signature" in str(response_error).lower():
                        log_step(log_path, {
                            "type": "step", "step": step,
                            "note": f"Detected thought-signature error, stripping reasoning fields from prior turns before retry: {response_error!r}",
                        })
                        for m in messages:
                            if m.get("role") == "assistant":
                                for key in list(m.keys()):
                                    if "reasoning" in key.lower() or "signature" in key.lower():
                                        del m[key]
                except Exception as e:  # noqa: BLE001 - transient API errors, retry
                    last_error = str(e)
                time.sleep(2 * (attempt + 1))

            if choice is None:
                log_step(log_path, {"type": "step", "step": step, "note": f"API failed after retries: {last_error}, ending episode"})
                break

            messages.append(choice.model_dump(exclude_none=True))

            if not choice.tool_calls:
                log_step(log_path, {"type": "step", "step": step, "note": "no tool call, stopping"})
                break

            # The model may return more than one tool_call in a single turn
            # (parallel tool calls) even though tool_choice="required" only
            # asks for at least one - every tool_call_id in the assistant
            # message MUST get a matching tool response or the next request
            # 400s, so only the first is actually executed but all get a
            # response (extras marked as skipped rather than silently
            # dropped, so the model sees why nothing happened for them).
            episode_finished = False
            for i, tool_call in enumerate(choice.tool_calls):
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")

                if i > 0:
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "content": "skipped: only one tool call is executed per turn",
                    })
                    continue

                log_step(log_path, {
                    "type": "step", "step": step, "screenshot": str(shot_path),
                    "reasoning": choice.content, "tool": name, "args": args,
                })

                tool_result = "ok"
                try:
                    if name == "click":
                        # Gemini outputs coordinates on its native 0-1000
                        # normalized scale, not raw pixels - confirmed by the
                        # consistent ~1.6x undershoot on x (=1600/1000, our
                        # viewport width) while y looked coincidentally
                        # correct (viewport height is also 1000). Rescale
                        # into real pixel coordinates before clicking.
                        real_x = args["x"] / 1000 * VIEWPORT["width"]
                        real_y = args["y"] / 1000 * VIEWPORT["height"]
                        page.mouse.click(real_x, real_y)
                    elif name == "type_text":
                        page.keyboard.type(args["text"])
                    elif name == "press_key":
                        page.keyboard.press(args["key"])
                    elif name == "scroll":
                        page.mouse.wheel(args.get("dx", 0), args.get("dy", 0))
                    elif name == "select_action":
                        selected_actions.append(args)
                        log_step(log_path, {"type": "select_action", "step": step, **args})
                    elif name == "finish":
                        log_step(log_path, {"type": "episode_end", "step": step, "notes": args.get("notes", "")})
                        tool_result = "finished"
                        episode_finished = True
                    else:
                        tool_result = f"unknown tool {name}"
                except Exception as e:  # noqa: BLE001 - log and let the model see/react to it
                    tool_result = f"error: {e}"

                page.wait_for_timeout(800)

                # target="_blank" links (e.g. "View in OpenEMR") open a new
                # tab that Playwright does NOT follow automatically - if one
                # appeared, switch to it so subsequent screenshots/actions
                # target the right tab instead of the now-stale original.
                if len(context.pages) > 1 and context.pages[-1] != page:
                    page = context.pages[-1]
                    page.bring_to_front()
                    page.wait_for_timeout(500)
                    tool_result += " (new tab opened, switched to it)"
                    log_step(log_path, {"type": "tab_switch", "step": step, "new_url": page.url})

                messages.append({
                    "role": "tool", "tool_call_id": tool_call.id, "content": tool_result,
                })

                # The URL(s) after this action executed - lets
                # grade_process.py verify Z1 steps 5-6 (correct patient/
                # encounter) and Z2 steps 9-10 (reached Configure Orders
                # and Results) against real ground truth instead of
                # guessing from a screenshot. OpenEMR's UI is heavily
                # frame-based - the top-level page stays on main.php while
                # patient/encounter/Configure Orders content loads inside
                # iframes (confirmed via a real run: page.url() showed
                # main.php for 12 consecutive steps while the agent visibly
                # navigated through Patient Finder/encounter/orders inside
                # the page) - so every frame's URL is captured, not just
                # the top-level one.
                try:
                    log_step(log_path, {"type": "page_state", "step": step, "url": page.url,
                                         "frame_urls": [f.url for f in page.frames]})
                except Exception:  # noqa: BLE001 - page may have closed/navigated mid-check, non-fatal
                    pass

            if episode_finished:
                break

        browser.close()

    log_step(log_path, {
        "type": "episode_summary",
        "selected_actions": selected_actions,
    })
    print(f"Episode complete. Log: {log_path}")
    print(f"Selected actions: {json.dumps(selected_actions, indent=2)}")
    return episode_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--visit-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--headed", action="store_true", help="Show the browser window instead of running headless.")
    args = parser.parse_args()

    run_episode(args.patient_id, args.visit_date, args.max_steps, headless=not args.headed)


if __name__ == "__main__":
    main()
