"""
run_model_eval.py

Real evaluation run: sends actual chest X-ray images to real models
(MedGemma via a Modal endpoint, Gemini 3.6/3.5 Flash via Google AI Studio),
gets each model's findings/impressions, judges them against clinician-
approved gold, and saves via judge_and_save(). No placeholder data, no
gold-vs-itself smoketest. GPT-5.5 is out of scope for this run (excluded
from READERS below, not deleted).

Four ablation conditions (2x2: prior-context x response-format):
  1. no_prior_full  - single image, no history, full prose (current default)
  2. no_prior_short - single image, no history, short phrases
  3. prior_full     - image + the model's OWN previous-visit report as
                      context, full prose
  4. prior_short    - image + the model's OWN previous-visit report as
                      context, short phrases

Why prior context matters: 42% of gold findings/impressions (64/152 rows,
checked directly against consolidated_real_reports_with_followups.csv)
contain change/comparison language ("worsening", "unchanged", "new", etc.)
- language the model structurally cannot match without being told what the
prior visit looked like. The no_prior conditions are a known-unfair test on
those rows; the prior conditions fix that.

The prior conditions use the model's OWN prior output (not gold) as context,
chained per-patient in chronological order, separately per model - so each
model builds on its own history, not another model's or gold's. This means
prior-condition cases must run sequentially per patient per model (not a
flat, order-independent case list like the no_prior conditions).

Results land as separate rows tagged "<model>__<condition>" (e.g.
"medgemma-1.5-4b-it__prior_full") in judge_results.jsonl.

Usage:
    python run_model_eval.py [N_CASES] [MODEL] [CONDITION]
    MODEL:     "medgemma" | "gemini-3.6-flash" | "gemini-3.5-flash" | "all" (default: all)
    CONDITION: "no_prior_full" | "no_prior_short" | "prior_full" | "prior_short" | "all" (default: all)
    Up to 152 gold rows available (10 patients) - pass 152 for the full set.

Gemini requires GOOGLE_API_KEY (or GEMINI_API_KEY) to be set - get one at
https://ai.google.dev, and `pip install google-genai` if not already installed.
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict

import requests
from pydantic import BaseModel, Field
import openai

from oracle import _GOLD_INDEX, has_gold, oracle_answer, format_clinical_context
from pacs_access import list_visits, get_image
from judge import judge_and_save

GPT_MODEL_NAME = "gpt-5.5"
MEDGEMMA_MODEL_NAME = "medgemma-1.5-4b-it"
MEDGEMMA_URL = "https://chartr--medgemma-app-generate-endpoint.modal.run"

client = openai.OpenAI()

# OpenRouter - one OpenAI-compatible client, model_id varies per call. Verify these
# slugs against https://openrouter.ai/models before a real run - naming may have
# shifted since these were picked.
import os  # noqa: E402
_openrouter_client = None


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _openrouter_client

MAX_RETRIES = 5
BASE_DELAY_S = 5


def with_retries(fn, *args, label: str = "", **kwargs):
    """Runs fn(*args, **kwargs) with exponential backoff on failure - a single
    transient error (rate limit, 503, flaky network) on one case shouldn't kill
    a run that may be hundreds of cases long. Re-raises after MAX_RETRIES."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            delay = BASE_DELAY_S * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{MAX_RETRIES}] {label} failed "
                  f"({type(exc).__name__}: {exc}) - retrying in {delay}s...")
            time.sleep(delay)
    raise last_exc

READER_SYSTEM_PROMPT = """You are a radiologist reading a chest X-ray. Write a real \
radiology report for the image shown: FINDINGS (what you observe, systematically) \
and IMPRESSION (the clinical conclusion, prioritized)."""

# Tips, verbatim in substance from the "Components of a Radiology Report" reference
# doc, scoped to only the two sections this eval grades (Findings + Impression) - no
# Who/How/Comparison-exam-header/Why, per the locked scope decision.
TIPS_FULL = """Tips:
- Use complete sentences and avoid abbreviations whenever possible.
- Separate the findings you make from the impression you give: the findings should be
  simply stated using proper radiologic terminology, while the impression should be a
  synthesis of the findings (and not merely a restatement of them).
- Keep the observations brief and to the point; succinctness is a virtue.
- The impression should give the overall impression of what the exam shows, in the
  form of a likely diagnosis or differential diagnosis."""

# Sample, trimmed to only Findings + Impression (Who/How/Comparison lines removed per
# the locked scope decision) - a real few-shot example of the target style. The
# anti-anchoring line guards against the model copying the EXAMPLE's content
# (finding a right lower lobe pneumonia on every case) rather than treating it as a
# style reference only.
SAMPLE_FULL = """Example of the target style:
FINDINGS: Cardiomediastinal silhouette is normal. There is a right lower lobe opacity
with an air bronchogram; the remaining lung fields are clear. There are no effusions.
The bony thorax appears normal.
IMPRESSION: Probable right lower lobe pneumonia.

Examples demonstrate reporting style only. Do not copy their findings or diagnoses
unless supported by the current image."""

# Hallucination safeguard - added to every condition prompt directly (not just the
# system prompt) since MedGemma's reader doesn't use a separate system prompt at all,
# so this must live in the prompt text itself to reach all three models consistently.
HALLUCINATION_SAFEGUARD = (
    "Do not invent abnormalities, clinical history, or interval change; report only "
    "what is supported by the current image and supplied context, use calibrated "
    "uncertainty when evidence is limited, and do not infer absence from omission in "
    "a prior report."
)

# Clinical-context instruction - added to every condition prompt, same reasoning as
# HALLUCINATION_SAFEGUARD (must live in prompt text, not just a system prompt, to
# reach MedGemma/OpenRouter models that don't use one). Explains WHY context matters
# (same finding, different urgency) so the model treats it as decision input, not
# decoration - mirrors the instruction already added to generate_followups.py's
# gold-follow-up prompt, kept in sync in spirit though the wording differs slightly
# since this one is read-the-image-and-write-the-report, not read-an-existing-report.
CLINICAL_CONTEXT_INSTRUCTION = (
    "You are also given CLINICAL CONTEXT (vitals/exam findings at the time of this "
    "study). Use it alongside the image: it can support or raise the index of "
    "suspicion for a finding (e.g. fever and hypoxemia make a subtle opacity more "
    "likely to represent real consolidation, not artifact), and it should drive how "
    "urgent your FOLLOW-UP recommendation is - the same imaging finding can warrant a "
    "very different follow-up depending on whether the patient is physiologically "
    "stable or in distress. Do not let clinical context override what the image "
    "actually shows; use it to inform interpretation and urgency, not to invent "
    "findings the image doesn't support."
)

# --- no_prior prompts: single image, no history ---
# {context} is filled with the visit's clinical-context block (see
# oracle.format_clinical_context) at call time.

NO_PRIOR_FULL = (
    f"Write a FINDINGS section , IMPRESSION section, and FOLLOW-UP section for this chest X-ray, following "
    f"standard radiology reporting practice.\n\n{TIPS_FULL}\n\n{SAMPLE_FULL}\n\n"
    f"{HALLUCINATION_SAFEGUARD}\n\n{CLINICAL_CONTEXT_INSTRUCTION}\n\n"
    f"CLINICAL CONTEXT:\n{{context}}\n\n"
    f"You are not given any prior exam for this patient - do not speculate about change "
    f"over time, and you will not be penalized for omitting it."
)

NO_PRIOR_SHORT = (
    "Write a FINDINGS section and an IMPRESSION section  and FOLLOW-UP section for this chest X-ray. Follow "
    "standard radiology reporting principles: findings should describe pertinent "
    "positive and negative findings using proper radiologic terminology; the impression "
    "should be a synthesis (a likely diagnosis or differential), not merely a "
    "restatement of the findings"
    f"{HALLUCINATION_SAFEGUARD} {CLINICAL_CONTEXT_INSTRUCTION}\n\n"
    f"CLINICAL CONTEXT:\n{{context}}\n\n"
    "You are not given any prior exam for this patient - do not speculate about change "
    "over time, and you will not be penalized for omitting it."
)

# --- prior prompts: image + the model's own previous-visit report ---
# {prior} is filled with the model's prior output when one exists; when a case is a
# patient's first visit on record, there is genuinely no prior, so these fall back to
# the no_prior prompt text for that one case (a real, expected first-visit condition,
# not a bug). {context} is filled the same way as the no_prior prompts above.

PRIOR_FULL = (
    f"Write a FINDINGS section and an IMPRESSION section and FOLLOW-UP section for this chest X-ray, following "
    f"standard radiology reporting practice.\n\n{TIPS_FULL}\n\n{SAMPLE_FULL}\n\n"
    f"{HALLUCINATION_SAFEGUARD}\n\n{CLINICAL_CONTEXT_INSTRUCTION}\n\n"
    f"CLINICAL CONTEXT:\n{{context}}\n\n"
    f"You ARE given this same patient's most recent PRIOR report in the history, if available column - use it to "
    f"describe change over time  where "
    f"relevant. If there's nothing relevant to compare, you will not be penalized for "
    f"omitting change language.\n\nPRIOR REPORT:\n{{prior}}"
)

PRIOR_SHORT = (
    "Write a FINDINGS section and an IMPRESSION section and FOLLOW-UP section for this chest X-ray. Follow "
    "standard radiology reporting principles: findings should describe pertinent "
    "positive and negative findings using proper radiologic terminology; the impression "
    "should be a synthesis (a likely diagnosis or differential), not merely a "
    "restatement of the findings."
    f"{HALLUCINATION_SAFEGUARD} {CLINICAL_CONTEXT_INSTRUCTION}\n\n"
    f"CLINICAL CONTEXT:\n{{context}}\n\n"
    "You ARE given this same "
    "patient's most recent PRIOR report below - use it to describe change over time  where relevant. If there's nothing "
    "relevant to compare, you will not be penalized for omitting change language."
    "\n\nPRIOR REPORT:\n{prior}"
)

CONDITIONS = {
    "no_prior_full": {"uses_prior": False, "prompt": NO_PRIOR_FULL},
    "no_prior_short": {"uses_prior": False, "prompt": NO_PRIOR_SHORT},
    "prior_full": {"uses_prior": True, "prompt": PRIOR_FULL, "no_prior_fallback": NO_PRIOR_FULL},
    "prior_short": {"uses_prior": True, "prompt": PRIOR_SHORT, "no_prior_fallback": NO_PRIOR_SHORT},
}


class ModelReport(BaseModel):
    findings: str = Field(description="The FINDINGS section of the report.")
    impressions: str = Field(description="The IMPRESSION section of the report.")
    follow_up: str = Field(
        default="",
        description="The FOLLOW-UP section: the recommended next step, informed by "
                    "findings, impressions, and the supplied clinical context together.",
    )


def read_xray_gpt(patient_id: str, visit_folder: str, series_name: str, prompt: str) -> ModelReport:
    image_b64 = get_image(patient_id, visit_folder, series_name)
    response = client.beta.chat.completions.parse(
        model=GPT_MODEL_NAME,
        messages=[
            {"role": "system", "content": READER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
        response_format=ModelReport,
    )
    return response.choices[0].message.parsed


def _parse_medgemma_output(text: str) -> ModelReport:
    """MedGemma (and OpenRouter models, which reuse this) return free text with
    FINDINGS:/IMPRESSION:/FOLLOW-UP: headers (confirmed by manual test), not
    structured output - split on those headers ourselves. FOLLOW-UP is optional
    since not every prompt variant asks for one and not every model reliably
    includes the header."""
    match = re.search(
        r"FINDINGS:\s*(.*?)\s*IMPRESSION:\s*(.*?)(?:\s*FOLLOW-UP:\s*(.*))?$",
        text, re.IGNORECASE | re.DOTALL,
    )
    if not match:
        # Fallback: couldn't find both required headers - treat whole thing as
        # findings, leave impressions/follow_up empty rather than guessing.
        return ModelReport(findings=text.strip(), impressions="", follow_up="")
    findings = match.group(1).strip()
    impressions = match.group(2).strip()
    follow_up = (match.group(3) or "").strip()
    return ModelReport(findings=findings, impressions=impressions, follow_up=follow_up)


def read_xray_medgemma(patient_id: str, visit_folder: str, series_name: str, prompt: str) -> ModelReport:
    image_b64 = get_image(patient_id, visit_folder, series_name)
    resp = requests.post(
        MEDGEMMA_URL,
        json={"image_b64": image_b64, "prompt": prompt},
        timeout=300,
    )
    resp.raise_for_status()
    return _parse_medgemma_output(resp.json()["text"])


def read_xray_gemini(model_id: str, patient_id: str, visit_folder: str, series_name: str,
                      prompt: str) -> ModelReport:
    import base64
    from google import genai
    from google.genai import types

    image_b64 = get_image(patient_id, visit_folder, series_name)
    gemini_client = genai.Client()  # reads GOOGLE_API_KEY / GEMINI_API_KEY from env

    response = gemini_client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/png"),
            READER_SYSTEM_PROMPT + "\n\n" + prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ModelReport,
        ),
    )
    return response.parsed


GEMINI_36_FLASH = "gemini-3.6-flash"
GEMINI_35_FLASH = "gemini-3.5-flash"


def read_xray_openrouter(model_id: str, patient_id: str, visit_folder: str,
                          series_name: str, prompt: str) -> ModelReport:
    """Generic reader for any OpenRouter vision-capable model - OpenAI-compatible
    chat completions API, same image_url content-block shape as read_xray_gpt."""
    image_b64 = get_image(patient_id, visit_folder, series_name)
    or_client = _get_openrouter_client()
    response = or_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": READER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
    )
    text = response.choices[0].message.content or ""
    # OpenRouter models don't all support response_format=json_schema reliably -
    # reuse MedGemma's FINDINGS:/IMPRESSION: text parsing instead of structured
    # output, since the prompt already asks for that format explicitly.
    return _parse_medgemma_output(text)


# Verify these against https://openrouter.ai/models before a real run - pick the
# current best vision-capable slug for each family, naming shifts over time.
NEMOTRON_VL_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
QWEN_VL_MODEL = "qwen/qwen2.5-vl-72b-instruct"
KIMI_VL_MODEL = "moonshotai/kimi-vl-a3b-thinking"
DEEPSEEK_VL_MODEL = "deepseek-ai/deepseek-vl2"

# GPT-5.5 intentionally excluded from the active roster per current eval scope.
# Each entry is (model_name, read_fn) where read_fn(patient_id, visit_folder,
# series_name, prompt).
READERS = {
    "medgemma": (MEDGEMMA_MODEL_NAME, read_xray_medgemma),
    "gemini-3.6-flash": (GEMINI_36_FLASH, lambda pid, vf, sn, prompt:
                          read_xray_gemini(GEMINI_36_FLASH, pid, vf, sn, prompt)),
    "gemini-3.5-flash": (GEMINI_35_FLASH, lambda pid, vf, sn, prompt:
                          read_xray_gemini(GEMINI_35_FLASH, pid, vf, sn, prompt)),
    "nemotron": (NEMOTRON_VL_MODEL, lambda pid, vf, sn, prompt:
                 read_xray_openrouter(NEMOTRON_VL_MODEL, pid, vf, sn, prompt)),
    "qwen": (QWEN_VL_MODEL, lambda pid, vf, sn, prompt:
             read_xray_openrouter(QWEN_VL_MODEL, pid, vf, sn, prompt)),
    "kimi": (KIMI_VL_MODEL, lambda pid, vf, sn, prompt:
             read_xray_openrouter(KIMI_VL_MODEL, pid, vf, sn, prompt)),
    "deepseek": (DEEPSEEK_VL_MODEL, lambda pid, vf, sn, prompt:
                 read_xray_openrouter(DEEPSEEK_VL_MODEL, pid, vf, sn, prompt)),
}


def cases_by_patient_chronological(n_cases: int):
    """Returns [(patient_id, [(visit, gold), ...chronological...]), ...], filling whole
    patient sequences before moving to the next, capped so total cases <= n_cases."""
    by_patient = defaultdict(list)
    for gold in _GOLD_INDEX.values():
        by_patient[gold.patient_id].append(gold)

    patients = []
    total = 0
    for pid, golds in by_patient.items():
        golds.sort(key=lambda g: (g.study_date, g.study_time or ""))
        try:
            visits = list_visits(pid)
        except ValueError:
            continue
        visit_by_key = {(v.study_date, v.study_time or ""): v for v in visits if v.series}

        seq = []
        for gold in golds:
            v = visit_by_key.get((gold.study_date, gold.study_time or ""))
            if v is not None:
                seq.append((v, gold))
        if not seq:
            continue

        patients.append((pid, seq))
        total += len(seq)
        if total >= n_cases:
            break
    return patients


def cases_from_visit_list(visit_dates: list[tuple[str, str]]):
    """Same return shape as cases_by_patient_chronological (grouped by
    patient, chronological within each patient), but built from an
    explicit (patient_id, 'YYYY-MM-DD') list instead of 'first N visits' -
    lets a specific batch (e.g. one already run through the computer-use
    harness) be re-run here for a direct, same-visits comparison.

    Matched via (patient_id, study_date) only, not full study_time - fine
    for lists that were already checked for same-day collisions (as the
    computer-use PILOT_VISITS list was), but will raise if a given
    patient/date pair is genuinely ambiguous, rather than silently picking
    one of two real visits."""
    by_patient: dict[str, list[str]] = defaultdict(list)
    for pid, date_str in visit_dates:
        by_patient[pid].append(date_str.replace("-", ""))

    patients = []
    for pid, dates in by_patient.items():
        golds_by_date = defaultdict(list)
        for gold in _GOLD_INDEX.values():
            if gold.patient_id == pid:
                golds_by_date[gold.study_date].append(gold)

        try:
            visits = list_visits(pid)
        except ValueError:
            continue
        visit_by_key = {(v.study_date, v.study_time or ""): v for v in visits if v.series}

        seq = []
        for date_str in sorted(dates):
            candidates = golds_by_date.get(date_str, [])
            if len(candidates) > 1:
                raise ValueError(f"{pid}/{date_str} is ambiguous ({len(candidates)} visits same day) - "
                                  "pick a different visit or add study_time to disambiguate.")
            if not candidates:
                raise ValueError(f"No gold found for {pid}/{date_str}.")
            gold = candidates[0]
            v = visit_by_key.get((gold.study_date, gold.study_time or ""))
            if v is None:
                raise ValueError(f"No matching DICOM visit for {pid}/{date_str}.")
            seq.append((v, gold))
        if seq:
            patients.append((pid, seq))
    return patients


def run_no_prior(model_name: str, read_fn, condition_name: str, prompt_template: str, patients):
    tagged_model_name = f"{model_name}__{condition_name}"
    for patient_id, seq in patients:
        for visit, gold in seq:
            series_name = visit.series[0].series_name
            print(f"--- {tagged_model_name} | {patient_id}/{visit.visit_folder} ---")
            prompt = prompt_template.format(context=format_clinical_context(gold))
            report = with_retries(read_fn, patient_id, visit.visit_folder, series_name, prompt,
                                   label=f"{tagged_model_name} read")
            # No prior context ever given in this condition - judge must not penalize
            # missing change language.
            _print_and_judge(patient_id, visit, gold, tagged_model_name, report,
                              had_prior_context=False)


def run_prior(model_name: str, read_fn, condition_name: str, prompt_template: str,
              no_prior_fallback: str, patients):
    tagged_model_name = f"{model_name}__{condition_name}"
    for patient_id, seq in patients:
        prior_report_text = None  # this model's own previous output for this patient
        for visit, gold in seq:
            series_name = visit.series[0].series_name
            has_prior = prior_report_text is not None
            print(f"--- {tagged_model_name} | {patient_id}/{visit.visit_folder} "
                  f"({'has prior' if has_prior else 'first visit, no prior'}) ---")
            context = format_clinical_context(gold)
            if not has_prior:
                # genuinely no prior exists for this case
                prompt = no_prior_fallback.format(context=context)
            else:
                prompt = prompt_template.format(prior=prior_report_text, context=context)
            report = with_retries(read_fn, patient_id, visit.visit_folder, series_name, prompt,
                                   label=f"{tagged_model_name} read")
            # had_prior_context reflects whether THIS case actually had a real prior,
            # not just whether the condition is "prior_*" - a patient's first visit in
            # a prior_* condition still has no real prior to compare against.
            _print_and_judge(patient_id, visit, gold, tagged_model_name, report,
                              had_prior_context=has_prior)
            prior_report_text = f"FINDINGS: {report.findings}\nIMPRESSION: {report.impressions}"


def _print_and_judge(patient_id, visit, gold, tagged_model_name, report, had_prior_context: bool):
    print(f"MODEL findings:    {report.findings[:150]}")
    print(f"MODEL impressions: {report.impressions[:150]}")
    print(f"MODEL follow-up:   {report.follow_up[:150]}")
    print(f"GOLD  findings:    {gold.findings[:150]}")
    print(f"GOLD  impressions: {gold.impressions[:150]}")

    record = with_retries(
        judge_and_save,
        patient_id=patient_id,
        visit_folder=visit.visit_folder,
        model_name=tagged_model_name,
        model_findings=report.findings,
        model_impressions=report.impressions,
        model_follow_up=report.follow_up,
        gold_findings=gold.findings,
        gold_impressions=gold.impressions,
        had_prior_context=had_prior_context,
        gold_action_ids=gold.gold_action_ids or None,  # None (not []) skips A3 scoring when this visit has no follow-up gold
        gold_recommendation=gold.gold_recommendation,
        label=f"{tagged_model_name} judge",
    )
    followup_note = f", followup={record['followup_score']:.3f}" if record["followup_score"] != "" else " (no followup gold for this visit)"
    print(f"SCORE: findings={record['findings_score']:.3f}, "
          f"impressions={record['impressions_score']:.3f}{followup_note}\n")


# Same 10 (patient_id, visit_date) pairs the computer-use harness
# (agent_episode.py / run_batch.py's PILOT_VISITS) was run on - stratified
# across all 10 real clinical conditions, all checked for same-day
# collisions. Used via `python run_model_eval.py pilot [MODEL] [CONDITION]`
# so MedGemma (or any reader) can be evaluated on the exact same visits
# for a direct comparison.
PILOT_VISIT_DATES = [
    ("GRDN004RP5BFHE0T", "2011-09-01"),  # Normal
    ("GRDN00BJ2R8QRO0L", "2016-03-29"),  # Cardiomegaly
    ("GRDN00BJ2R8QRO0L", "2013-04-02"),  # Atelectasis
    ("GRDN004RP5BFHE0T", "2009-06-01"),  # Effusion
    ("GRDN00BKCEOKC7S1", "2024-01-12"),  # Consolidation
    ("GRDN006VFINFR877", "2013-03-25"),  # Emphysema
    ("GRDN00BKCEOKC7S1", "2025-02-12"),  # Pneumothorax
    ("GRDN006MCYEVYFE8", "2011-07-25"),  # Edema
    ("GRDN004RP5BFHE0T", "2009-06-15"),  # Pneumonia
    ("GRDN00BE97VCHW4E", "2025-01-05"),  # Lung opacity
]


if __name__ == "__main__":
    use_pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"
    n_cases = 3 if use_pilot else (int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    cond_arg = sys.argv[3] if len(sys.argv) > 3 else "no_prior_full"

    readers_to_run = READERS.items() if which == "all" else [(which, READERS[which])]
    conditions_to_run = CONDITIONS.items() if cond_arg == "all" else [(cond_arg, CONDITIONS[cond_arg])]

    patients = cases_from_visit_list(PILOT_VISIT_DATES) if use_pilot else cases_by_patient_chronological(n_cases)
    total_cases = sum(len(seq) for _, seq in patients)
    print(f"Running {[k for k, _ in readers_to_run]} x {list(dict(conditions_to_run))} "
          f"on {total_cases} case(s) across {len(patients)} patient(s)...\n")

    for reader_key, (model_name, read_fn) in readers_to_run:
        for condition_name, cfg in conditions_to_run:
            print(f"=== {model_name} / {condition_name} ===")
            if cfg["uses_prior"]:
                run_prior(model_name, read_fn, condition_name, cfg["prompt"],
                          cfg["no_prior_fallback"], patients)
            else:
                run_no_prior(model_name, read_fn, condition_name, cfg["prompt"], patients)

    print("Done. Real model output + scores saved to judge_results.jsonl / judge_results.csv.")
