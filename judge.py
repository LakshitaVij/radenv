"""
judge.py

The reward function for Task 2's Accuracy Axis (A1/A2/A3). Findings,
impressions, and follow-ups are free text (or free-text-derived), so
there's no formula (unlike the RCM env's tier-band reward) - an LLM judge
stands in for a human grader, comparing a model's output against
clinician-approved gold for semantic (not exact-string) equivalence.

Rewritten from the earlier 0.0-1.0-per-dimension/weighted-average design to
match the discrete point-value scoring in the "Accuracy Axis" design doc
(A1: 8 metrics/finding, range [-10,+7]; A2: 8 metrics/diagnosis, range
[-15,+7]; A3: 9 metrics/action, range documented as [-15,+8]).

Two things confirmed by hand-checking the doc's own stated ranges against
its own metric point values, both flagged to and confirmed with the user
before implementing this way (do not "fix" silently if this file is
revisited later without re-deriving the same math):
  - Despite the doc saying "weighted average," the stated ranges for A1
    ([-10,+7]) and A2 ([-15,+7]) only make arithmetic sense as a RAW SUM of
    the per-metric point values, not a percentage-weighted average - the
    percentages in the doc don't change the computed score here. Verified:
    A1 max = 1+1+1+1+1+1+1+0 = 7, min = -1-1-1-2-1-1-2-1 = -10 (exact -
    presence_absence's incorrect case is -1, not 0, per a later revision).
    A2 max = 1+1+1+1+1+0+1+1 = 7, min = -2-2-2-1-2-2-2-2 = -15 (exact).
  - A3's stated range ([-15,+8]) does NOT match this same raw-sum approach
    applied to its 9 metrics (raw sum gives max=+9, min=-18). Per explicit
    user direction: implement the 9 metrics literally as specified, do not
    invent mutual-exclusivity rules to force the numbers to -15/+8 - the
    doc's stated range is a label, not a hard constraint the code enforces.

Scoring is decomposed per-item (per finding / per diagnosis / per action)
rather than one holistic score per section, so a low score tells you WHAT
was wrong and on WHICH item, not just that something was wrong. Each item
is matched (model output paired to a gold item), missed (a gold item the
model never addressed), or hallucinated (a model item with no gold
counterpart) - see _score_a1/_score_a2/_score_a3 docstrings for exactly
which metrics apply to which item type.

A3's ground truth source: the design doc says `generated_followups.csv`,
but that file is empty (0 bytes, confirmed) - using
`generated_followups_fin.csv` instead (real data for all 152 visits:
generated_action_ids + generated_recommendation columns), which is the
file already used throughout the OpenEMR action-hierarchy work this
session. Action IDs are translated to human-readable names via the same
xlsx the OpenEMR seed script uses, so the judge sees names, not codes.

Usage:
    from judge import judge, judge_and_save

    result = judge(
        model_findings="...", model_impressions="...",
        gold_findings="...", gold_impressions="...",
    )
    print(result.a1_score, result.a2_score)

    # or, scored AND persisted in one call (A3 follow-up scoring is
    # optional - omit gold_action_ids to skip it, e.g. for older callers):
    record = judge_and_save(
        patient_id="...", visit_folder="...", model_name="...",
        model_findings="...", model_impressions="...",
        gold_findings="...", gold_impressions="...",
        model_follow_up="...", gold_action_ids=["ACT_..."], gold_recommendation="...",
    )
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import openai

JUDGE_MODEL = "gpt-5.5"

ROOT = Path(__file__).resolve().parent
RESULTS_JSONL = ROOT / "judge_results.jsonl"
RESULTS_CSV = ROOT / "judge_results.csv"
ACTION_LIBRARY_XLSX = Path("/Users/lakshitavij/ChartR-ClinicalRL/pulmonary_action_library_with_2026_guideline_rewards.xlsx")

# Always prefer the key in computer_use_eval/.openai_key over whatever's in
# the shell environment - the environment's copy has been stale before
# (confirmed multiple times this session), and every subprocess call chain
# (run_batch.py -> grade_episode.py -> judge.py) needs this to actually work,
# not just interactive shells that happen to have exported the right one.
_LOCAL_KEY_FILE = ROOT / "computer_use_eval" / ".openai_key"
if _LOCAL_KEY_FILE.exists():
    os.environ["OPENAI_API_KEY"] = _LOCAL_KEY_FILE.read_text().strip()

client = openai.OpenAI()


# --------------------------------------------------------------------------
# Action ID -> human-readable name, for A3 prompts (gold ground truth is
# action_ids like "ACT_THORACIC_US"; the judge should see "Thoracic
# Ultrasound", the same names used throughout the OpenEMR hierarchy work).
# --------------------------------------------------------------------------

def _load_action_names() -> dict[str, str]:
    if not ACTION_LIBRARY_XLSX.exists():
        return {}
    import openpyxl
    wb = openpyxl.load_workbook(ACTION_LIBRARY_XLSX, data_only=True)
    ws = wb["Action Library"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = rows[0]
    names = {}
    for r in rows[1:]:
        d = dict(zip(header, r))
        if d.get("action_id"):
            names[d["action_id"]] = d["action_name"]
    return names


_ACTION_NAMES = _load_action_names()


def _action_label(action_id: str) -> str:
    return f"{_ACTION_NAMES.get(action_id, action_id)} ({action_id})"


# --------------------------------------------------------------------------
# A1: Findings - per-finding items
# --------------------------------------------------------------------------

class FindingScoreItem(BaseModel):
    finding_description: str = Field(description="The finding this item scores - the model's wording if matched/hallucinated, gold's wording if missed.")
    match_status: Literal["matched", "missed_by_model", "hallucinated_by_model"]

    # Only meaningful for match_status="matched" - score 0 otherwise.
    presence_absence: float = Field(description="+1 correct / -2 incorrect. 0 if not matched.")
    laterality: float = Field(description="+1 correct / -1 wrong or hallucination / 0 if not lateralized or not matched.")
    severity: float = Field(description="+1 correct / -0.5 over-escalating / -2 under-calling or hallucination / 0 if not matched.")
    location: float = Field(description="+1 precise / +0.5 general region / -1 wrong or hallucination / 0 if not matched.")
    confidence_wording: float = Field(description="+1 match / -0.5 over- or under-confident / -2 hallucination. 0 if not matched.")
    clinical_reasoning: float = Field(description="+1 sound / +0.5 weak / -2 incorrect (includes hallucinated reasoning - not a separate case). 0 if not matched.")
    comparison_accuracy: float = Field(description="+1 correctly describes change from a prior study (unchanged/improved/worsened/new) when gold's finding is itself comparative / -1 wrong or omitted comparison gold explicitly makes. 0 if gold's finding isn't comparative, or not matched.")

    reasoning: str = Field(description="One or two sentences: why this item was scored this way.")

    @property
    def _match_penalty(self) -> float:
        """Replaces the old false_negatives/false_positives fields - both were
        pure functions of match_status, so the LLM was independently
        "scoring" something already implied by match_status itself. Derived
        here instead of asked for, removing the risk of an inconsistent
        combination (e.g. match_status="missed_by_model" but a model output
        of false_negatives=+1). Symmetric -5.0/-5.0: a missed real finding
        and a fabricated one are treated as equally bad - a clinician has to
        either notice the omission themselves or actively rule out a
        hallucinated one, both real safety costs. Raised from -2.0/-1.0 to
        -5.0/-5.0 per explicit user request, applied uniformly across
        A1/A2/A3 for consistency (same change in DiagnosisScoreItem and
        FollowupScoreItem below)."""
        return {"matched": 1.0, "missed_by_model": -5.0, "hallucinated_by_model": -5.0}[self.match_status]

    @property
    def raw_total(self) -> float:
        return (
            self.presence_absence + self.laterality + self.severity + self.location
            + self.confidence_wording + self.clinical_reasoning + self.comparison_accuracy
            + self._match_penalty
        )


class FindingsJudgement(BaseModel):
    items: list[FindingScoreItem] = Field(
        description="One item per gold finding (matched or missed_by_model), plus one item "
                    "per model finding with no gold counterpart (hallucinated_by_model). "
                    "Include pertinent negatives (e.g. 'no pneumothorax') as findings too."
    )


# --------------------------------------------------------------------------
# A2: Impressions - per-diagnosis items
# --------------------------------------------------------------------------

class DiagnosisScoreItem(BaseModel):
    diagnosis_description: str = Field(description="The diagnosis this item scores - model's wording if matched/hallucinated, gold's if missed.")
    match_status: Literal["matched", "missed_by_model", "hallucinated_by_model"]

    diagnosis_match: float = Field(description="+1 correct diagnosis at correct prominence (primary listed as primary, secondary listed as secondary) / +0.5 correct diagnosis but wrong prominence / -2 wrong diagnosis. 0 if not matched.")
    prioritization: float = Field(description="+1 correct order / +0.5 partial / -2 wrong. 0 if not matched.")
    severity_tone: float = Field(description="+1 match / -0.5 alarmist when unnecessary / -2 dismissive of critical. 0 if not matched.")
    confidence_wording: float = Field(description="+1 match / -0.5 over- or under-confident. 0 if not matched.")
    clinical_reasoning: float = Field(description="+1 sound / +0.5 weak / -2 incorrect (includes hallucinated reasoning - not a separate case). 0 if not matched.")

    reasoning: str = Field(description="One or two sentences: why this item was scored this way.")

    @property
    def _match_penalty(self) -> float:
        """Same deterministic-from-match_status derivation as
        FindingScoreItem._match_penalty - see that docstring (symmetric
        -5.0/-5.0)."""
        return {"matched": 1.0, "missed_by_model": -5.0, "hallucinated_by_model": -5.0}[self.match_status]

    @property
    def raw_total(self) -> float:
        return (
            self.diagnosis_match + self.prioritization
            + self.severity_tone + self.confidence_wording + self.clinical_reasoning
            + self._match_penalty
        )


class ImpressionsJudgement(BaseModel):
    items: list[DiagnosisScoreItem] = Field(
        description="One item per gold diagnosis (matched or missed_by_model), in gold's "
                    "priority order, plus one item per model diagnosis with no gold counterpart."
    )


# --------------------------------------------------------------------------
# A3: Follow-ups - per-action items
# --------------------------------------------------------------------------

class FollowupScoreItem(BaseModel):
    action_description: str = Field(description="The action this item scores - model's wording if matched/hallucinated, gold's action name if missed.")
    match_status: Literal["matched", "missed_by_model", "hallucinated_by_model"]

    severity_appropriate_intensity: float = Field(description="+1 match / +0.5 slight off / -0.5 over-urgent / -2 under-urgent. 0 if not matched (no gold urgency to compare against).")
    conflict_avoidance: float = Field(description="+1 no conflicts / -0.5 one conflict / -1 multiple / -2 contraindicated, relative to the OTHER actions actually selected this episode. 0 if missed (nothing was selected to check) - stays live for hallucinated_by_model, since a fabricated action can still genuinely conflict with something else selected.")
    tone: float = Field(description="+1 match / +0.5 slight off / -0.5 alarmist / -2 dismissive. 0 if not matched.")
    prioritization: float = Field(description="+1 correct order / +0.5 partial / -1 wrong, relative to gold's action ordering. 0 if not matched.")
    confidence_wording: float = Field(description="+1 match / -0.5 over-confident / -1 under-confident. 0 if not matched.")

    reasoning: str = Field(description="One or two sentences: why this item was scored this way.")

    @property
    def _match_penalty(self) -> float:
        """Replaces the old essential_action_recall/secondary_capture fields
        - both were just match_status restated (a primary/secondary split of
        the same "caught/missed/hallucinated" fact, same redundancy pattern
        as A1/A2's false_negatives/positives and diagnosis_match merge).
        Symmetric -5.0/-5.0, same as A1/A2 - see FindingScoreItem._match_penalty."""
        return {"matched": 1.0, "missed_by_model": -5.0, "hallucinated_by_model": -5.0}[self.match_status]

    @property
    def raw_total(self) -> float:
        return (
            self.severity_appropriate_intensity + self.conflict_avoidance
            + self.tone + self.prioritization + self.confidence_wording + self._match_penalty
        )


class FollowupJudgement(BaseModel):
    items: list[FollowupScoreItem] = Field(
        description="One item per gold action (matched or missed_by_model), plus one item "
                    "per model-recommended action with no gold counterpart."
    )
    abstention_calibration: float = Field(
        description="+1 correct / -0.5 acted when should abstain / -2 abstained when should act. "
                    "ONE episode-level score, not per-action - see FOLLOWUP_SYSTEM_PROMPT."
    )
    unnecessary_avoidance: float = Field(
        description="+1 only necessary actions selected / -0.5 one unnecessary / -1 multiple / "
                    "-2 excessive. ONE episode-level score covering the whole action set, not "
                    "per-action - see FOLLOWUP_SYSTEM_PROMPT."
    )


# --------------------------------------------------------------------------
# Combined result + persistence
# --------------------------------------------------------------------------

class AccuracyJudgement(BaseModel):
    findings: FindingsJudgement
    impressions: ImpressionsJudgement
    follow_up: FollowupJudgement | None = None

    @property
    def a1_score(self) -> float:
        return sum(i.raw_total for i in self.findings.items)

    @property
    def a2_score(self) -> float:
        return sum(i.raw_total for i in self.impressions.items)

    @property
    def a3_score(self) -> float | None:
        if self.follow_up is None:
            return None
        return (
            sum(i.raw_total for i in self.follow_up.items)
            + self.follow_up.abstention_calibration
            + self.follow_up.unnecessary_avoidance
        )

    @property
    def total_score(self) -> float:
        return self.a1_score + self.a2_score + (self.a3_score or 0.0)


FINDINGS_SYSTEM_PROMPT = """You are grading an AI model's FINDINGS description of a chest X-ray \
against a clinician-approved gold-standard FINDINGS section for the SAME image. You are not \
re-reading the image yourself - only comparing two descriptions for clinical equivalence.

For each finding in gold, output ONE item: "matched" if the model reported an equivalent \
finding (score presence_absence/laterality/severity/location/confidence_wording/ \
clinical_reasoning/comparison_accuracy per the field descriptions), or "missed_by_model" if the \
model never addressed it (all other fields=0 - match_status alone captures that it was missed).

For each model finding with NO equivalent in gold, output one "hallucinated_by_model" item \
(all other fields=0 - match_status alone captures that it was hallucinated).

COMPARISON_ACCURACY: some gold findings are inherently comparative - they describe change (or \
lack of it) relative to a prior study (e.g. "unchanged left lower lobe nodule," "increased \
right mediastinal mass," "new since prior"). When gold's finding is comparative, score whether \
the model's finding correctly captures that same direction of change (+1), gets it wrong or \
omits the comparison entirely (-1). If gold's finding is NOT comparative (a one-time \
observation with nothing to compare against), score 0 - this only applies when gold itself \
makes a comparison.

Include pertinent negatives (e.g. "no pneumothorax") as findings if either side states them. \
Grade on CLINICAL MEANING, not wording - different phrasing of the same finding should score \
well on the applicable metrics."""

IMPRESSIONS_SYSTEM_PROMPT = """You are grading an AI model's IMPRESSION (diagnostic synthesis) \
against a clinician-approved gold-standard IMPRESSION for the same case, given the FINDINGS both \
are based on. You are not re-reading the image - only comparing the two impressions.

For each diagnosis in gold (in gold's priority order), output ONE item: "matched" if the model \
reported an equivalent diagnosis (score diagnosis_match/prioritization/severity_tone/ \
confidence_wording/clinical_reasoning per the field descriptions), or "missed_by_model" if the \
model never reported it (all other fields=0 - match_status alone captures that it was missed).

For each model diagnosis with NO equivalent in gold, output one "hallucinated_by_model" item \
(all other fields=0 - match_status alone captures that it was hallucinated).

Grade on CLINICAL MEANING, not wording."""

FOLLOWUP_SYSTEM_PROMPT = """You are grading an AI model's recommended follow-up action(s) for a \
patient's chest X-ray against the clinician-consensus gold-standard action(s) for the same case.

For each gold action, output ONE item: "matched" if the model recommended an equivalent action \
(score severity_appropriate_intensity/conflict_avoidance/tone/prioritization/confidence_wording \
per the field descriptions), or "missed_by_model" if the model never recommended it (all fields \
except conflict_avoidance=0 - match_status alone captures that it was missed; conflict_avoidance \
is always 0 for a missed item since nothing was actually selected to check for conflicts).

For each model-recommended action with NO gold equivalent, output one "hallucinated_by_model" \
item. severity_appropriate_intensity/tone/prioritization/confidence_wording=0 (no gold \
counterpart to compare against), but conflict_avoidance stays live - a fabricated action can \
still genuinely conflict with something else the model actually selected.

TWO SEPARATE EPISODE-LEVEL SCORES (not per-action - set once each on the response, covering the \
WHOLE action set, not any single item):

ABSTENTION_CALIBRATION: if gold's action list is empty/abstention-only and the model correctly \
recommended no action or explicitly abstained, score +1. If gold expected real action(s) and the \
model abstained/recommended nothing, score -2. If gold expected abstention but the model still \
recommended something, score -0.5. Otherwise (model correctly acted when action was warranted), \
score +1.

UNNECESSARY_AVOIDANCE: looking at the model's FULL set of recommended actions together, how many \
were unnecessary/excessive relative to what gold and the clinical picture actually warrant? +1 if \
every recommended action was necessary, -0.5 if exactly one was unnecessary, -1 if multiple were \
unnecessary, -2 if the action set was excessive overall.

Grade on CLINICAL MEANING, not exact wording - a model recommending "chest ultrasound" for a gold \
action of "thoracic ultrasound" is a match."""


def judge_findings(model_findings: str, gold_findings: str) -> FindingsJudgement:
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": FINDINGS_SYSTEM_PROMPT},
            {"role": "user", "content": f"GOLD FINDINGS:\n{gold_findings or '(empty)'}\n\n"
                                        f"MODEL FINDINGS:\n{model_findings or '(empty)'}"},
        ],
        response_format=FindingsJudgement,
    )
    return response.choices[0].message.parsed


def judge_impressions(model_impressions: str, gold_impressions: str,
                       model_findings: str = "", gold_findings: str = "") -> ImpressionsJudgement:
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": IMPRESSIONS_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"GOLD FINDINGS (context):\n{gold_findings or '(empty)'}\n\n"
                f"GOLD IMPRESSION:\n{gold_impressions or '(empty)'}\n\n"
                f"MODEL FINDINGS (context):\n{model_findings or '(empty)'}\n\n"
                f"MODEL IMPRESSION:\n{model_impressions or '(empty)'}"
            )},
        ],
        response_format=ImpressionsJudgement,
    )
    return response.choices[0].message.parsed


def judge_followup(model_follow_up: str, gold_action_ids: list[str], gold_recommendation: str) -> FollowupJudgement:
    gold_actions_labeled = "\n".join(f"- {_action_label(a)}" for a in gold_action_ids) or "(none - abstention expected)"
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"GOLD ACTIONS:\n{gold_actions_labeled}\n\n"
                f"GOLD RECOMMENDATION (context/rationale):\n{gold_recommendation or '(empty)'}\n\n"
                f"MODEL FOLLOW-UP:\n{model_follow_up or '(empty)'}"
            )},
        ],
        response_format=FollowupJudgement,
    )
    return response.choices[0].message.parsed


def judge(
    model_findings: str,
    model_impressions: str,
    gold_findings: str,
    gold_impressions: str,
    model_follow_up: str = "",
    gold_action_ids: list[str] | None = None,
    gold_recommendation: str = "",
) -> AccuracyJudgement:
    """Scores A1 (findings) and A2 (impressions) always. Scores A3
    (follow-up) only if gold_action_ids is given - omit it for callers that
    don't have structured follow-up ground truth yet."""
    findings = judge_findings(model_findings, gold_findings)
    impressions = judge_impressions(model_impressions, gold_impressions, model_findings, gold_findings)
    follow_up = None
    if gold_action_ids is not None:
        follow_up = judge_followup(model_follow_up, gold_action_ids, gold_recommendation)
    return AccuracyJudgement(findings=findings, impressions=impressions, follow_up=follow_up)


def save_result(record: dict, jsonl_path: Path = RESULTS_JSONL, csv_path: Path = RESULTS_CSV) -> None:
    """Append one record to both JSONL (source of truth, easy to reprocess)
    and CSV (easy to open/skim)."""
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def judge_and_save(
    patient_id: str,
    visit_folder: str,
    model_name: str,
    model_findings: str,
    model_impressions: str,
    gold_findings: str,
    gold_impressions: str,
    had_prior_context: bool = False,
    model_follow_up: str = "",
    gold_action_ids: list[str] | None = None,
    gold_recommendation: str = "",
) -> dict:
    """Score a case and persist the result. Returns the saved record.

    had_prior_context is accepted for backward compatibility with existing
    callers but no longer affects scoring (change_accuracy was dropped from
    A1 per the revised design doc - see module docstring)."""
    result = judge(
        model_findings, model_impressions, gold_findings, gold_impressions,
        model_follow_up=model_follow_up, gold_action_ids=gold_action_ids,
        gold_recommendation=gold_recommendation,
    )

    record = {
        "patient_id": patient_id,
        "visit_folder": visit_folder,
        "model": model_name,
        "judge_model": JUDGE_MODEL,
        "model_findings": model_findings,
        "model_impressions": model_impressions,
        "model_follow_up": model_follow_up,
        "gold_findings": gold_findings,
        "gold_impressions": gold_impressions,
        "gold_action_ids": ";".join(gold_action_ids) if gold_action_ids else "",
        "findings_score": result.a1_score,
        "findings_items": json.dumps([i.model_dump() for i in result.findings.items]),
        "impressions_score": result.a2_score,
        "impressions_items": json.dumps([i.model_dump() for i in result.impressions.items]),
        "followup_score": result.a3_score if result.a3_score is not None else "",
        "followup_items": json.dumps([i.model_dump() for i in result.follow_up.items]) if result.follow_up else "",
        "total_accuracy_score": result.total_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_result(record)
    return record


if __name__ == "__main__":
    import sys
    from pacs_access import list_visits
    from oracle import has_gold, oracle_answer

    patient_id = sys.argv[1] if len(sys.argv) > 1 else "GRDN00BKCEOKC7S1"

    visits = list_visits(patient_id)
    case = next((v for v in visits if has_gold(patient_id, v)), None)
    if case is None:
        sys.exit(f"No gold case found for {patient_id!r} to self-test against.")

    gold = oracle_answer(patient_id, case)
    print(f"Self-test case: {patient_id} / {case.visit_folder} (VisitNum={gold.visit_num})")
    print(f"Gold findings:    {gold.findings[:100]!r}")
    print(f"Gold impressions: {gold.impressions[:100]!r}")

    print("\n--- Judging gold against ITSELF (sanity check, expect high per-item scores) ---")
    result = judge(
        model_findings=gold.findings, model_impressions=gold.impressions,
        gold_findings=gold.findings, gold_impressions=gold.impressions,
    )
    print(f"A1 (findings):    {result.a1_score:.2f}  ({len(result.findings.items)} items)")
    for item in result.findings.items:
        print(f"  [{item.match_status}] {item.finding_description}: {item.raw_total:+.1f} - {item.reasoning}")
    print(f"A2 (impressions): {result.a2_score:.2f}  ({len(result.impressions.items)} items)")
    for item in result.impressions.items:
        print(f"  [{item.match_status}] {item.diagnosis_description}: {item.raw_total:+.1f} - {item.reasoning}")

    print("\n--- Judging a deliberately wrong answer (sanity check, expect negative scores) ---")
    result = judge(
        model_findings="The lungs are clear. No acute cardiopulmonary process.",
        model_impressions="Normal chest radiograph.",
        gold_findings=gold.findings, gold_impressions=gold.impressions,
    )
    print(f"A1 (findings):    {result.a1_score:.2f}  ({len(result.findings.items)} items)")
    for item in result.findings.items:
        print(f"  [{item.match_status}] {item.finding_description}: {item.raw_total:+.1f} - {item.reasoning}")
    print(f"A2 (impressions): {result.a2_score:.2f}  ({len(result.impressions.items)} items)")
    for item in result.impressions.items:
        print(f"  [{item.match_status}] {item.diagnosis_description}: {item.raw_total:+.1f} - {item.reasoning}")

    print(f"\n(Self-test runs above are not saved. Use judge_and_save() for real "
          f"model outputs - they'll land in {RESULTS_JSONL.name} / {RESULTS_CSV.name}.)")
