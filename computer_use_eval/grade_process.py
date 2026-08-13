"""
grade_process.py

Process Axis (Z1/Z2) grader for one agent_episode.py run. Reads a
completed episode's log.jsonl and computes the exact point rules from the
Process Axis design doc.

Design decisions confirmed with the user before writing this (see plan
doc "Marry Process Axis and Accuracy Axis into one graded episode",
revised for the embedded-in-OpenEMR app - the agent now starts inside
OpenEMR already, so Z1 runs login -> patient -> encounter -> vitals ->
office visit/history -> open X-ray -> interact with X-ray, the reverse
of the old standalone-viewer-first order):
  - Z1 steps 4-5 (engages vitals / office visit): each clicks a real,
    independently identifiable form (formname=vitals / formname=newpatient
    in the URL, since OpenEMR routes every form-open click through the
    same openEncounterForm() JS function) - unlike the old standalone app,
    these ARE independently detectable now, not a shared proxy. Step 5 is
    "only if given": an encounter with no Office Visit/newpatient form
    attached scores +1 regardless, since there's nothing to click into.
  - Z1 steps 6-7 (open X-ray / interact with X-ray): same formname-based
    detection, now the LAST two Z1 steps instead of the first two.
  - Z1 step 8 (engages prior imaging): same "only if given" carve-out as
    step 5 - a patient with no prior study on file can't be penalized for
    not comparing to it. Detected via a URL query param the frontend
    stamps itself (XrayFullScreenViewer.jsx's stampUrlParam) rather than a
    formname= pattern, since the prior-studies timeline is a client-side
    React state change, not a page/form navigation - there's no other way
    for a post-hoc log scan to see it happened.
  - Z2 steps renumbered sequentially (9-13) per explicit user request - an
    earlier revision left gaps (9, 12, 14, 15, 16) where dropped checks used
    to sit, on purpose, so labels matched exactly what appeared in old
    episode logs. Now fully renumbered instead, code and doc kept in sync:
      Z2.9  Navigate to Procedures/Configuration
      Z2.10 No mutual exclusivity conflicts
      Z2.11 Abstention calibration
      Z2.12 Action count calibration
      Z2.13 Step efficiency
  - Z2.9 (navigate Procedures / Configuration): ONE check, not two - both
    used to be separate ScoreItems reading the exact same underlying
    boolean ("did the log ever show a URL for Configure Orders and
    Results, types.php"), which silently double-counted a single event in
    the Z2 sum. Merged into one +1/-1 item.
  - A since-removed valid-action check used to score catalog-validity PER
    selected action (+1/-1 each, summed). Catalog-validity is now purely
    an Accuracy-axis concern (A3's hallucinated_by_model handles it);
    Process no longer scores it, though the underlying catalog match still
    runs internally since Z2.10 and the action-count check both need to
    know which selections are real.
  - Z2.10 (mutual exclusivity conflicts): no longer scored per pair. Per-
    pair scoring made this combinatorial (C(n,2) pairs for n actions - a
    6-action episode could earn +15 from this ONE check alone, dwarfing
    every real penalty in the system). Now: each of the n selected actions
    contributes +-1/n, so the whole check always sums to within [-1,+1]
    regardless of how many actions were selected. A parallel, since-
    removed "can_combine" check was dropped outright: that field is True
    for every single action in the catalog (confirmed against the real
    xlsx), so it could never detect a conflict and only contributed free,
    uninformative points that scaled with action count.
  - Z2.12 ("too many/too few actions"): compared against THIS VISIT's
    actual gold action count (via oracle.py), not a fixed universal band
    - real visits have 1-6 gold actions, not always 1-3. Direct, uncapped
    per-action penalty (no longer normalized by gold_n): -5.0 per missing
    action (same magnitude as a missed item on the Accuracy axis - a
    missing gold action is the same underlying failure either way) vs
    -0.25 per extra action (a harmless additional order is a minor
    inefficiency, not a missed necessity). Missing 4 of 4 gold actions
    costs -20.0, not a flat -0.75 - this DOES now scale with task size on
    purpose (PROCESS_FLOOR in grade_episode.py accounts for this
    per-episode via gold_n).
  - Z2.11 (abstention calibration): "should abstain" ground truth = gold
    action list is empty OR contains ACT_ABSTAIN/ACT_CLINICAL_CORRELATION.
    Same definition used in judge.py's A3 abstention handling so Process
    and Accuracy don't quietly disagree.
  - Z1 range [-8.5,8] (was [-7.5,7] before step 8 was added). Z2.10 is
    normalized to [-1,+1] regardless of action count, but Z2.12's
    per-action penalty (-0.25/-1.0) DOES grow with how far off the action
    count is, which is intentional: a badly-miscalibrated action count
    (e.g. Task 4's 6-action case answered with 0) should be able to swing
    the total further negative than a fixed-range check would allow.

Ground truth patient/encounter comes from a direct DB query (same
`docker compose exec mysql` pattern used throughout this session), not
hardcoded - so this grader stays correct if the seed data changes.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import openpyxl

COMPOSE_DIR = Path("/Users/lakshitavij/synthetic pipeline/openemr/docker/development-easy")
ACTION_LIBRARY_XLSX = Path("/Users/lakshitavij/ChartR-ClinicalRL/pulmonary_action_library_with_2026_guideline_rewards.xlsx")

# The 3 actions added this session that aren't in the xlsx - same
# can_combine/mutually_exclusive_group defaults used when they were
# registered into OpenEMR (see seed_action_hierarchy.py / the plan doc).
_NEW_ACTIONS_META = {
    "ACT_REPOSITION_LINE": {"name": "Reposition Or Retract Central Line/Catheter", "can_combine": True, "group": None},
    "ACT_MANAGE_PLEURAL_CATHETER": {"name": "Manage Indwelling Pleural Catheter", "can_combine": True, "group": None},
    "ACT_OPTIMIZE_BP": {"name": "Optimize Blood Pressure/Hemodynamics", "can_combine": True, "group": None},
}

ABSTAIN_ACTION_IDS = {"ACT_ABSTAIN", "ACT_CLINICAL_CORRELATION"}


def _humanize(action_id: str) -> str:
    return " ".join(w.capitalize() for w in action_id.replace("ACT_", "").split("_"))


def _load_action_catalog() -> dict[str, dict]:
    """action_id -> {name, can_combine, group}, for matching select_action's
    human-readable level_2 labels back to the real catalog."""
    catalog = {}
    if ACTION_LIBRARY_XLSX.exists():
        wb = openpyxl.load_workbook(ACTION_LIBRARY_XLSX, data_only=True)
        ws = wb["Action Library"]
        rows = list(ws.iter_rows(min_row=1, values_only=True))
        header = rows[0]
        for r in rows[1:]:
            d = dict(zip(header, r))
            if d.get("action_id"):
                catalog[d["action_id"]] = {
                    "name": d["action_name"],
                    "can_combine": bool(d.get("can_combine", True)),
                    "group": d.get("mutually_exclusive_group") or None,
                }
    for action_id, meta in _NEW_ACTIONS_META.items():
        catalog[action_id] = meta
    return catalog


_CATALOG = _load_action_catalog()
_NAME_TO_ID = {meta["name"].strip().lower(): action_id for action_id, meta in _CATALOG.items()}
_NAME_TO_ID.update({_humanize(action_id).strip().lower(): action_id for action_id in _CATALOG})


def _match_action_id(level_2_label: str) -> str | None:
    """Map select_action's on-screen label back to a real action_id - exact
    match first, falling back to a normalized/loose match since the agent
    may not reproduce the label byte-for-byte."""
    label = (level_2_label or "").strip().lower()
    if label in _NAME_TO_ID:
        return _NAME_TO_ID[label]
    for name, action_id in _NAME_TO_ID.items():
        if name in label or label in name:
            return action_id
    return None


def _run_sql(sql: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "mysql", "mariadb", "-uroot", "-proot", "openemr", "-e", sql],
        cwd=COMPOSE_DIR, capture_output=True, text=True, check=True,
    )
    return result.stdout


def _all_urls(page_state: dict) -> str:
    """OpenEMR's UI is frame-based - the top-level page stays on main.php
    while patient/encounter/Configure Orders content loads in iframes
    (confirmed via a real run where page.url() showed main.php for 12
    consecutive steps while the agent visibly navigated internally).
    Search every frame's URL, not just the top-level one."""
    return " ".join([page_state.get("url", "")] + page_state.get("frame_urls", []))


def ground_truth_pid_encounter(patient_id: str, visit_date: str) -> tuple[int | None, int | None]:
    """Real (pid, encounter_id) for this patient's visit, via direct DB
    query - not hardcoded, so this stays correct if seed data changes."""
    out = _run_sql(
        "SELECT pd.pid, fe.encounter FROM patient_data pd "
        "JOIN form_encounter fe ON fe.pid=pd.pid "
        f"WHERE pd.lname='{patient_id}' AND DATE(fe.date)='{visit_date}';"
    )
    lines = [l for l in out.strip().splitlines()[1:] if l.strip()]
    if not lines:
        return None, None
    pid, encounter = lines[0].split("\t")
    return int(pid), int(encounter)


# Real OpenEMR note-type forms an agent might legitimately choose to
# document in, now that the prompt no longer tells it which one to use -
# confirmed real, via each form's own table.sql (not guessed). Each entry:
# (real table name, the text field(s) worth reading as "the note").
NOTE_FORM_REGISTRY: dict[str, tuple[str, list[str]]] = {
    "soap": ("form_soap", ["subjective", "objective", "assessment", "plan"]),
    "clinical_notes": ("form_clinical_notes", ["description"]),
    "clinic_note": ("form_clinic_note", ["history", "examination", "plan"]),
    "note": ("form_note", ["message"]),
    "clinical_instructions": ("form_clinical_instructions", ["instruction"]),
}


def read_all_note_forms(patient_id: str, visit_date: str, after_ts: float | None = None) -> dict[str, dict[str, str]]:
    """Every real note-type form saved for this patient's encounter, across
    all the forms an agent might have legitimately chosen to use - not just
    one assumed form type. Returns {formdir: {field: text}} for whichever
    forms actually have a saved row; forms with nothing saved are omitted.

    after_ts (a unix timestamp) restricts this to forms saved AFTER that
    time - critical when the same patient/encounter gets used across
    multiple episode runs (exactly what happens re-running the same
    reference tasks repeatedly): without this, a later episode that wrote
    NOTHING could get graded on an earlier episode's leftover saved note,
    since `forms` rows persist in OpenEMR across runs and this query has no
    other way to tell "written by this episode" apart from "written by
    literally any prior run against this same patient/encounter"."""
    true_pid, true_encounter = ground_truth_pid_encounter(patient_id, visit_date)
    if true_pid is None:
        return {}
    ts_filter = f" AND f.date >= FROM_UNIXTIME({after_ts})" if after_ts is not None else ""
    found = {}
    for formdir, (table, fields) in NOTE_FORM_REGISTRY.items():
        field_list = ", ".join(f"s.{f}" for f in fields)
        try:
            out = _run_sql(
                f"SELECT {field_list} FROM forms f JOIN {table} s ON s.id = f.form_id "
                f"WHERE f.pid={true_pid} AND f.encounter={true_encounter} AND f.formdir='{formdir}'{ts_filter} "
                "ORDER BY f.id DESC LIMIT 1;"
            )
        except subprocess.CalledProcessError:
            continue  # table doesn't exist in this OpenEMR install - skip, not fatal
        lines = [l for l in out.strip().splitlines()[1:] if l.strip() or l == ""]
        if not lines:
            continue
        values = lines[0].split("\t")
        field_map = dict(zip(fields, values))
        if any(v.strip() for v in field_map.values()):
            found[formdir] = field_map
    return found


def all_typed_text(entries: list[dict]) -> str:
    """Every type_text call's text, concatenated in call order - no length
    filtering, no exclusion list. This is a raw utility, not safe to grade
    directly - see documentation_typed_after_xray() for the guarded
    version actually used for grading."""
    return "\n\n".join(
        e["args"]["text"] for e in entries
        if e.get("type") == "step" and e.get("tool") == "type_text"
    )


def documentation_typed_after_xray(entries: list[dict], xray_step: int | None) -> str:
    """The report is now one continuous free-text block (per explicit user
    decision - no more 'Findings:'/'Impression:'/'Follow-up:' labels), so
    there's no label to safely filter on anymore. The safety concern
    labels used to solve - a login credential or patient-search string
    getting swept into 'documentation' - is instead solved by a step
    cutoff: nothing typed before the agent actually opened the X-ray
    Viewer could possibly be a clinical report (login, patient search, and
    encounter navigation all happen earlier), so only type_text calls at
    or after xray_step count. Returns "" if the X-ray was never opened -
    consistent with split_three()'s old "no marker at all, grade nothing"
    behavior."""
    if xray_step is None:
        return ""
    return "\n\n".join(
        e["args"]["text"] for e in entries
        if e.get("type") == "step" and e.get("tool") == "type_text" and e.get("step", 0) >= xray_step
    )


def score_documentation_saved(entries: list[dict], saved_notes: dict[str, dict[str, str]], xray_step: int | None) -> ScoreItem | None:
    """New check, per explicit user request: if the agent typed a real
    report (after opening the X-ray) but none of it ended up in any real
    saved note form, that's a real failure - it believed it documented
    something and didn't. -0.25 for that; +1 if typed content was found
    saved; None (not scored) if nothing was typed after opening the X-ray
    at all, since this check is specifically about typed-but-lost content,
    not about whether documentation was attempted."""
    typed = documentation_typed_after_xray(entries, xray_step)
    if not typed:
        return None

    saved_text_blob = " ".join(
        v for field_map in saved_notes.values() for v in field_map.values()
    ).lower()

    # A real save may have been edited/retyped slightly - require a solid
    # substring match (first 40 chars) rather than exact equality.
    needle = typed.strip().lower()[:40]
    if needle and needle in saved_text_blob:
        return ScoreItem("Documentation actually saved", 1, "Typed a report after opening the X-ray and it was found in a real saved note form.")
    return ScoreItem("Documentation actually saved", -0.25,
                      "Typed a report after opening the X-ray but none of it was found in any saved note form - "
                      "likely typed into a form and never clicked Save, or navigated away before saving.")


def load_episode_log(log_path: Path) -> list[dict]:
    with log_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class ScoreItem:
    def __init__(self, step: str, points: float, note: str):
        self.step = step
        self.points = points
        self.note = note

    def to_row(self, axis: str) -> dict:
        return {"axis": axis, "item": self.step, "points": self.points, "note": self.note}


def _form_exists(pid: int, encounter: int, formdir: str) -> bool:
    """Ground truth: does this encounter actually have a form of this type
    attached at all? Needed for Z1.5's "only if given" rule - an encounter
    with no Office Visit/newpatient form shouldn't penalize the agent for
    not engaging with something that was never there."""
    out = _run_sql(
        f"SELECT COUNT(*) FROM forms WHERE formdir='{formdir}' AND pid={pid} AND encounter={encounter} AND deleted=0;"
    )
    lines = [l for l in out.strip().splitlines()[1:] if l.strip()]
    return bool(lines) and lines[0] != "0"


def _reached_form_step(page_states: list[dict], formname: str) -> int | None:
    """OpenEMR routes every 'open this attached form' click (Vitals,
    Office Visit/newpatient, X-ray Viewer, ...) through the same
    openEncounterForm(formdir, formname, formid) JS function, which lands
    on a URL containing formname=<formdir> - whether that's a same-frame
    navigation, a new browser tab, or a new iframe added to the current
    page (all three are real, observed behaviors of twAddFrameTab).
    Checking every page_state's full URL set (top-level + all frames)
    catches all three without needing to guess from click coordinates.
    Returns the step number of the first page_state showing this
    formname, or None if it never appeared."""
    pattern = re.compile(rf"formname={formname}\b")
    for ps in page_states:
        if pattern.search(_all_urls(ps)):
            return ps.get("step")
    return None


def score_z1(entries: list[dict], patient_id: str, visit_date: str) -> tuple[list[ScoreItem], list[dict]]:
    """New embedded-app step order: the agent starts inside OpenEMR (no
    separate X-ray app to switch into), so information-gathering now goes
    login -> find patient -> find encounter -> engage vitals -> engage
    office visit/history -> open the X-ray form -> interact with it -
    the reverse of the old standalone-viewer-first order."""
    steps = [e for e in entries if e.get("type") == "step"]
    page_states = [e for e in entries if e.get("type") == "page_state"]

    items = []

    typed_texts = [e["args"].get("text", "").lower() for e in steps if e.get("tool") == "type_text"]
    logged_in = any("admin" in t for t in typed_texts) and any("pass" in t for t in typed_texts)
    items.append(ScoreItem("Z1.1 Logs in", 1 if logged_in else -1,
                            "Typed both username and password." if logged_in else "Login credentials not both entered."))

    true_pid, true_encounter = ground_truth_pid_encounter(patient_id, visit_date)
    if true_pid is None:
        items.append(ScoreItem("Z1.2 Correct patient", -1, "No ground-truth encounter found for this visit - unverifiable."))
        items.append(ScoreItem("Z1.3 Correct encounter", -1, "No ground-truth encounter found for this visit - unverifiable."))
        items.append(ScoreItem("Z1.4 Engages vitals", -1, "No ground-truth encounter - unverifiable."))
        items.append(ScoreItem("Z1.5 Engages history/office visit", -1, "No ground-truth encounter - unverifiable."))
        items.append(ScoreItem("Z1.6 Clicks open X-ray", -2, "No ground-truth encounter - unverifiable."))
        items.append(ScoreItem("Z1.7 Interacts with X-ray", -0.5, "No ground-truth encounter - unverifiable."))
        return items, page_states

    pid_pattern = re.compile(rf"set_pid=0*{true_pid}\b")
    reached_correct_pid = any(pid_pattern.search(_all_urls(ps)) for ps in page_states)
    if reached_correct_pid:
        items.append(ScoreItem("Z1.2 Correct patient", 1, f"Reached set_pid={true_pid}."))
    else:
        items.append(ScoreItem("Z1.2 Correct patient", -1, "Never reached the correct patient page."))

    enc_pattern = re.compile(rf"set_encounter=0*{true_encounter}\b")
    reached_correct_enc = any(enc_pattern.search(_all_urls(ps)) for ps in page_states)
    if reached_correct_enc:
        items.append(ScoreItem("Z1.3 Correct encounter", 1, f"Reached set_encounter={true_encounter}."))
    else:
        items.append(ScoreItem("Z1.3 Correct encounter", -1, "Never reached the correct encounter page."))

    # Z1.4/Z1.5: "clicked + scrolled/read" - clicked into the real form
    # (formname=vitals / formname=newpatient in the URL, per
    # _reached_form_step) AND took at least one further action there
    # before moving on, same "did something after arriving" proxy the old
    # shared vitals/history check used - "scrolled/read" isn't a distinct
    # tool call the way X-ray zoom/pan is, so genuine reading can't be
    # detected more precisely than "didn't immediately navigate away."
    def _engagement_item(step_name: str, formname: str, required: bool) -> ScoreItem:
        reached_step = _reached_form_step(page_states, formname)
        if reached_step is None:
            if required:
                return ScoreItem(step_name, -1, f"Never opened the {formname} form.")
            # "only if given" - this encounter has no such form attached at
            # all, so the agent had nothing to click into; don't penalize.
            return ScoreItem(step_name, 1, f"No {formname} form exists for this encounter - not counted against the agent.")
        took_further_action = any(e.get("type") == "step" and e.get("step", 0) > reached_step for e in entries)
        if took_further_action:
            return ScoreItem(step_name, 1, f"Opened the {formname} form and took further action there.")
        return ScoreItem(step_name, -1, f"Opened the {formname} form but showed no further engagement.")

    items.append(_engagement_item("Z1.4 Engages vitals", "vitals", required=True))

    office_visit_exists = _form_exists(true_pid, true_encounter, "newpatient")
    items.append(_engagement_item("Z1.5 Engages history/office visit", "newpatient", required=office_visit_exists))

    xray_step = _reached_form_step(page_states, "xray_viewer")
    items.append(ScoreItem("Z1.6 Clicks open X-ray", 1 if xray_step is not None else -2,
                            "Opened the X-ray Viewer form." if xray_step is not None else "Never opened the X-ray Viewer form."))

    interacted = xray_step is not None and any(
        e.get("type") == "step" and e.get("tool") == "scroll" and e.get("step", 0) > xray_step
        for e in entries
    )
    items.append(ScoreItem("Z1.7 Interacts with X-ray", 1 if interacted else -0.5,
                            "scroll tool call after opening the X-ray." if interacted else "No scroll/zoom after opening the X-ray."))

    # Same "nothing to engage with" carve-out as Z1.5 - a patient with no
    # prior imaging on file can't be penalized for not comparing to it.
    # Detected the same way every other Z1 checkpoint is: a pattern in the
    # logged URLs, not a live query - the frontend stamps
    # priorStudiesAvailable=0/1 into the URL once it knows, and
    # viewedPriorStudy=1 if the agent ever clicks a timeline entry (see
    # XrayFullScreenViewer.jsx's stampUrlParam). If priorStudiesAvailable
    # never appears at all, the X-ray was never opened far enough to reach
    # this - already penalized once via Z1.6, not double-counted here.
    prior_checked = any("priorStudiesAvailable=" in _all_urls(ps) for ps in page_states)
    prior_available = any("priorStudiesAvailable=1" in _all_urls(ps) for ps in page_states)
    viewed_prior = any("viewedPriorStudy=1" in _all_urls(ps) for ps in page_states)
    if not prior_checked or not prior_available:
        items.append(ScoreItem("Z1.8 Engages prior imaging", 1, "No prior imaging study exists for this patient - not counted against the agent."))
    elif viewed_prior:
        items.append(ScoreItem("Z1.8 Engages prior imaging", 1, "Viewed a prior study via the timeline."))
    else:
        items.append(ScoreItem("Z1.8 Engages prior imaging", -1, "Prior imaging exists for this patient but was never viewed via the timeline."))

    return items, page_states


def score_z2_navigation(page_states: list[dict]) -> ScoreItem:
    """'Navigate to Procedures' and 'Navigate to Configuration' collapse into
    one check, not two - reaching types.php - since the intermediate
    Procedures menu click isn't a separate page we can detect, and scoring
    the same single boolean as two rows silently double-weighted it in the
    Z2 sum (formerly Z2.9 + Z2.10, +1/-1 each)."""
    reached = any("types.php" in _all_urls(ps) for ps in page_states)
    return ScoreItem(
        "Z2.9 Navigate to Procedures/Configuration", 1 if reached else -1,
        "Reached Configure Orders and Results." if reached else "Never reached Configure Orders and Results.",
    )


def score_z2_actions(selected_actions: list[dict], gold_action_ids: list[str]) -> list[ScoreItem]:
    """No longer scores catalog-validity itself (formerly Z2.11, +1/-1 per
    selected action) - an agent hallucinating a nonexistent action is an
    Accuracy-axis concern (A3's hallucinated_by_model), not a Process one.
    Matching against the catalog still has to happen internally, though -
    Z2.10 and the action-count check below both need to know which
    selections are real actions."""
    items = []
    valid_ids = [_match_action_id(sa.get("level_2", "")) for sa in selected_actions]
    valid_ids = [a for a in valid_ids if a is not None]

    # Z2.10: mutual exclusivity - one non-scaling check, not scored per pair
    # (per-pair scoring made this combinatorial: C(n,2) pairs meant a
    # 6-action episode could earn +15 from this single check alone,
    # dwarfing every real penalty in the system). Each of the n selected
    # actions contributes +-1/n instead, so the whole check always sums to
    # within [-1, +1] no matter how many actions were selected.
    n = len(valid_ids)
    if n > 0:
        groups = [_CATALOG.get(a, {}).get("group") for a in valid_ids]
        conflicted = [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                if groups[i] and groups[i] == groups[j]:
                    conflicted[i] = conflicted[j] = True
        per_action = 1.0 / n
        points = round(sum(per_action if not c else -per_action for c in conflicted), 3)
        n_conflicted = sum(conflicted)
        items.append(ScoreItem("Z2.10 No mutual exclusivity conflicts", points,
                                f"{n - n_conflicted}/{n} action(s) conflict-free."))

    # "Too many/too few" relative to THIS visit's real gold count - direct,
    # uncapped per-action penalty (no longer normalized by gold_n). A missing
    # action is a missed GOLD action - same -5.0 magnitude as a missed item
    # on the Accuracy axis (judge.py's match_status penalty), since it's the
    # same underlying failure (a required real action never happened), just
    # observed from the Process side (EMR order execution) instead of the
    # Accuracy side (judged output). An extra action costs -0.25 - much
    # cheaper, since one harmless additional order is a minor inefficiency,
    # not a missed necessity. Unlike the old normalized version (which
    # always capped at exactly -0.75/-0.25 regardless of how far off the
    # count was), this scales with the actual gap - selecting 0 of 4
    # expected actions now costs -20.0, not a flat -0.75. A conflicting
    # extra action (contradicts another selected action) is scored
    # separately by Z2.10's mutual-exclusivity check above; this checkpoint
    # only counts raw over/under-selection, not conflicts.
    gold_n, model_n = len(gold_action_ids), len(valid_ids)
    if model_n > gold_n:
        extra = model_n - gold_n
        penalty = round(-0.25 * extra, 3)
        items.append(ScoreItem("Z2.12 Action count calibration", penalty, f"Selected {model_n} actions vs. {gold_n} expected - {extra} too many."))
    elif model_n < gold_n and gold_n > 0:
        missing = gold_n - model_n
        penalty = round(-5.0 * missing, 3)
        items.append(ScoreItem("Z2.12 Action count calibration", penalty, f"Selected {model_n} actions vs. {gold_n} expected - {missing} too few."))

    return items, valid_ids


def score_z2_abstention(valid_ids: list[str], gold_action_ids: list[str]) -> ScoreItem:
    gold_says_abstain = len(gold_action_ids) == 0 or any(a in ABSTAIN_ACTION_IDS for a in gold_action_ids)
    model_abstained = len(valid_ids) == 0 or any(a in ABSTAIN_ACTION_IDS for a in valid_ids)

    if gold_says_abstain and model_abstained:
        return ScoreItem("Z2.11 Abstention calibration", 1, "Gold expected abstention, agent abstained.")
    if not gold_says_abstain and model_abstained:
        return ScoreItem("Z2.11 Abstention calibration", -2, "Gold expected real action(s), agent abstained instead.")
    if gold_says_abstain and not model_abstained:
        return ScoreItem("Z2.11 Abstention calibration", 0.25, "Gold expected abstention, agent acted anyway.")
    return ScoreItem("Z2.11 Abstention calibration", 1, "Gold expected real action(s), agent acted.")


STEP_EFFICIENCY_FIXED_OVERHEAD = 15
# 1 X-ray interaction (zoom/pan) + 1 click to open the X-ray Viewer form +
# 2 type login credentials (username, password) + 1 submit login + 1
# find/reach the correct patient + 1 find/reach the correct encounter + 1
# engage vitals +
# 1 engage history + 1 navigate to a note form + 1 type documentation +
# 1 click Save + 1 navigate to Procedures + 1 navigate to Configuration/
# Configure Orders and Results + 1 finish = 15. Interaction with the X-ray
# is treated as a required step, not optional - per explicit user
# decision, since a real clinical read requires actually inspecting the
# image, not just glancing at the default view. This is a transparent,
# fixed floor - not a claim about the exact optimal click path (that
# depends on OpenEMR's UI in ways not verifiable from the log alone) -
# plus one select_action call per gold action for this visit (minimum 1,
# even for an abstention decision).

STEP_EFFICIENCY_INTERACTION_BUFFER = 2
# Extra allowance on top of the 1 required interaction step, since a real
# read of the X-ray plausibly needs more than one zoom/pan (e.g. checking
# multiple regions) - this buffer absorbs that without penalizing
# reasonable additional interaction as "wasted steps."


def _episode_fully_completed(process_items: list[ScoreItem]) -> bool:
    """'Completed fully end to end, without skipping anything' - stricter
    than a positive Z1/Z2 point total (which can stay positive even with a
    real gap, e.g. several correct action selections offsetting one bad
    one). Every checkpoint below must have cleanly passed."""
    by_step: dict[str, list[float]] = {}
    for item in process_items:
        by_step.setdefault(item.step, []).append(item.points)

    def _all_positive(step_name: str) -> bool:
        return step_name in by_step and all(p > 0 for p in by_step[step_name])

    required_clean = [
        "Z1.2 Correct patient", "Z1.3 Correct encounter",
        "Z1.4 Engages vitals", "Z1.5 Engages history/office visit",
        "Z1.6 Clicks open X-ray", "Z1.7 Interacts with X-ray",
        "Z1.8 Engages prior imaging",
        "Z2.9 Navigate to Procedures/Configuration",
    ]
    if not all(_all_positive(s) for s in required_clean):
        return False
    if any(p < 0 for p in by_step.get("Z2.10 No mutual exclusivity conflicts", [])):
        return False
    if "Z2.12 Action count calibration" in by_step:  # any row here means the count didn't match exactly
        return False
    doc_points = by_step.get("Documentation actually saved", [])
    if not doc_points or doc_points[0] != 1:
        return False
    return True


def score_step_efficiency(entries: list[dict], process_items: list[ScoreItem], gold_action_ids: list[str]) -> ScoreItem:
    """New check, per explicit user request: reward completing the full
    required workflow in the fewest steps, penalize unnecessary extra
    steps. Full +1 credit only if the episode both (a) completed the
    entire required workflow with no skipped/failed checkpoint (per
    _episode_fully_completed) and (b) took no more than the minimum step
    count. -0.25 per step beyond the minimum applies regardless of
    completion - unnecessary steps are penalized either way."""
    actual_steps = len([e for e in entries if e.get("type") == "step"])
    min_actions = max(1, len(gold_action_ids))
    minimum_steps = STEP_EFFICIENCY_FIXED_OVERHEAD + STEP_EFFICIENCY_INTERACTION_BUFFER + min_actions
    extra_steps = max(0, actual_steps - minimum_steps)
    penalty = round(-0.25 * extra_steps, 2)

    fully_completed = _episode_fully_completed(process_items)
    bonus = 1 if (fully_completed and extra_steps == 0) else 0
    points = round(bonus + penalty, 2)

    note = (
        f"{actual_steps} actual step(s) vs. {minimum_steps} minimum "
        f"({STEP_EFFICIENCY_FIXED_OVERHEAD} fixed incl. required X-ray interaction + "
        f"{STEP_EFFICIENCY_INTERACTION_BUFFER} interaction buffer + {min_actions} for "
        f"{len(gold_action_ids)} gold action(s)). "
        f"{'Fully completed' if fully_completed else 'NOT fully completed (skipped/failed a required checkpoint)'}, "
        f"{extra_steps} step(s) over minimum."
    )
    return ScoreItem("Z2.13 Step efficiency", points, note)


def grade_process(log_path: Path, patient_id: str, visit_date: str, gold_action_ids: list[str]) -> list[ScoreItem]:
    entries = load_episode_log(log_path)
    z1_items, page_states = score_z1(entries, patient_id, visit_date)

    z2_nav = score_z2_navigation(page_states)
    selected_actions = [e for e in entries if e.get("type") == "select_action"]
    z2_actions, valid_ids = score_z2_actions(selected_actions, gold_action_ids)
    z2_abstain = score_z2_abstention(valid_ids, gold_action_ids)

    xray_step = _reached_form_step(page_states, "xray_viewer")
    # after_ts scopes the note lookup to THIS episode only - agent_episode.py
    # names each episode dir with its own start time
    # (f"{patient_id}_{visit_date}_{int(time.time())}"), so a rerun against
    # the same patient/visit can never get graded on a prior run's leftover
    # saved note (see grade_episode.py for the full explanation).
    episode_start_ts = int(log_path.parent.name.rsplit("_", 1)[-1])
    saved_notes = read_all_note_forms(patient_id, visit_date, after_ts=episode_start_ts)
    doc_saved_item = score_documentation_saved(entries, saved_notes, xray_step)
    doc_items = [doc_saved_item] if doc_saved_item else []

    all_items = z1_items + [z2_nav] + z2_actions + [z2_abstain] + doc_items
    efficiency_item = score_step_efficiency(entries, all_items, gold_action_ids)

    return all_items + [efficiency_item]


if __name__ == "__main__":
    import sys
    log_path = Path(sys.argv[1])
    patient_id = sys.argv[2]
    visit_date = sys.argv[3]
    gold_ids = sys.argv[4].split(";") if len(sys.argv) > 4 and sys.argv[4] else []

    items = grade_process(log_path, patient_id, visit_date, gold_ids)
    total = sum(i.points for i in items)
    for i in items:
        print(f"  {i.points:+.2f}  {i.step}: {i.note}")
    print(f"\nProcess total: {total:+.2f}")
