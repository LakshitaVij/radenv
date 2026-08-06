"""
oracle.py

The "perfect player" for Task 2: given a case, always returns the correct
answer. Built directly from clinician-approved gold
(csv/consolidated_real_reports.csv - the regex-fixed extraction, 149/152
cases with valid findings+impressions vs. the earlier
consolidated_real_reports_with_followups.csv's 58/152) - no reasoning, just
lookup.

Follow-up gold (gold_action_ids, gold_recommendation) is merged in from
generated_followups_fin.csv, keyed the same way (PatientID/StudyDate/
StudyTime) - this is the real, populated follow-up ground truth used
throughout the OpenEMR action-hierarchy work this session (the doc-specified
`generated_followups.csv` is empty, 0 bytes - not used). A case can have
gold findings/impressions but no follow-up row, or vice versa; gold_action_ids
is an empty list and gold_recommendation is "" when there's no follow-up row
for that visit, rather than raising - not every case needs to be answerable
on every axis to be usable for A1/A2.

Used to:
  - Prove a case is answerable at all (if the oracle can't find gold for a
    case, that case isn't usable in the eval - it's a data gap, not a model
    failure).
  - Give agent-vs-oracle comparisons a real ceiling, instead of just
    agent-vs-random.

Keyed on (PatientID, StudyDate, StudyTime) - StudyDate alone is NOT unique,
a patient can have two separate visits on the same day (seen in real data:
GRDN00BKCEOKC7S1 has two studies both dated 20240525). StudyTime, read
straight from the DICOM header via pacs_access, disambiguates and matches
the gold CSV's StudyTime column exactly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from pacs_access import list_visits, VisitInfo

GOLD_CSV = Path(__file__).resolve().parent / "csv" / "consolidated_real_reports.csv"
FOLLOWUP_CSV = Path(__file__).resolve().parent / "generated_followups_fin.csv"


@dataclass
class GoldAnswer:
    patient_id: str
    visit_num: str
    study_date: str
    study_time: str
    findings: str
    impressions: str
    follow_up: str
    # Synthetic clinical context (vitals/exam at time of study), clinician-
    # reviewed - merged into consolidated_real_reports.csv by
    # csv/merge_clinical_context.py. condition/phenotype record which row of
    # the source xlsx's "Global Context by Condition" sheet was sampled from
    # ("Normal" + blank phenotype for non-pathological visits).
    condition: str
    phenotype: str
    oxygen_saturation_percent: str
    oxygen_device: str
    respiratory_rate_per_min: str
    work_of_breathing: str
    systolic_bp_mmHg: str
    temperature_C: str
    mental_status: str
    current_care_setting: str
    # Follow-up ground truth, merged in from generated_followups_fin.csv -
    # empty list/string when that visit has no follow-up row (not every
    # case needs a follow-up answer to be usable for A1/A2).
    gold_action_ids: list[str] = field(default_factory=list)
    gold_recommendation: str = ""


def _load_followups() -> dict[tuple[str, str, str], tuple[list[str], str]]:
    """Index generated_followups_fin.csv rows by (PatientID, StudyDate,
    StudyTime) -> (action_ids, recommendation text). Same key shape as
    _load_gold() so the two merge directly."""
    index: dict[tuple[str, str, str], tuple[list[str], str]] = {}
    if not FOLLOWUP_CSV.exists():
        return index
    with FOLLOWUP_CSV.open() as f:
        for row in csv.DictReader(f):
            key = (row["PatientID"], row["StudyDate"], row["StudyTime"])
            action_ids = [a for a in (row.get("generated_action_ids") or "").split(";") if a]
            index[key] = (action_ids, row.get("generated_recommendation", ""))
    return index


def _load_gold() -> dict[tuple[str, str, str], GoldAnswer]:
    """Index gold rows by (PatientID, StudyDate, StudyTime)."""
    followups = _load_followups()
    index: dict[tuple[str, str, str], GoldAnswer] = {}
    with GOLD_CSV.open() as f:
        for row in csv.DictReader(f):
            key = (row["PatientID"], row["StudyDate"], row["StudyTime"])
            action_ids, recommendation = followups.get(key, ([], ""))
            index[key] = GoldAnswer(
                patient_id=row["PatientID"],
                visit_num=row["VisitNum"],
                study_date=row["StudyDate"],
                study_time=row["StudyTime"],
                findings=row["FindingsText"],
                impressions=row["ImpressionText"],
                follow_up=row.get("Followup", ""),
                condition=row.get("condition", ""),
                phenotype=row.get("phenotype", ""),
                oxygen_saturation_percent=row.get("oxygen_saturation_percent", ""),
                oxygen_device=row.get("oxygen_device", ""),
                respiratory_rate_per_min=row.get("respiratory_rate_per_min", ""),
                work_of_breathing=row.get("work_of_breathing", ""),
                systolic_bp_mmHg=row.get("systolic_bp_mmHg", ""),
                temperature_C=row.get("temperature_C", ""),
                mental_status=row.get("mental_status", ""),
                current_care_setting=row.get("current_care_setting", ""),
                gold_action_ids=action_ids,
                gold_recommendation=recommendation,
            )
    return index


def format_clinical_context(gold: "GoldAnswer") -> str:
    """Renders a GoldAnswer's clinical context as a labeled block for prompts.
    Used by both generate_followups.py and run_model_eval.py so the exact
    wording models/graders see stays in sync in one place."""
    if not gold.oxygen_saturation_percent:
        return "(not available for this case)"
    return (
        f"- SpO2: {gold.oxygen_saturation_percent}% on {gold.oxygen_device}\n"
        f"- Respiratory rate: {gold.respiratory_rate_per_min}/min\n"
        f"- Work of breathing: {gold.work_of_breathing}\n"
        f"- Systolic BP: {gold.systolic_bp_mmHg} mmHg\n"
        f"- Temperature: {gold.temperature_C}°C\n"
        f"- Mental status: {gold.mental_status}\n"
        f"- Care setting: {gold.current_care_setting}"
    )


_GOLD_INDEX = _load_gold()


def _key_for(patient_id: str, visit: VisitInfo) -> tuple[str, str, str]:
    return (patient_id, visit.study_date, visit.study_time or "")


def has_gold(patient_id: str, visit: VisitInfo) -> bool:
    return _key_for(patient_id, visit) in _GOLD_INDEX


def oracle_answer(patient_id: str, visit: VisitInfo) -> GoldAnswer:
    """The oracle's answer for a case: exactly the clinician-approved gold."""
    key = _key_for(patient_id, visit)
    if key not in _GOLD_INDEX:
        raise ValueError(
            f"No gold for patient {patient_id!r}, visit_folder {visit.visit_folder!r} "
            f"(study_date={visit.study_date!r}, study_time={visit.study_time!r}). "
            f"This case has no gold and should be excluded from the eval, not scored."
        )
    return _GOLD_INDEX[key]


def gold_case_count() -> int:
    return len(_GOLD_INDEX)


if __name__ == "__main__":
    import sys

    print(f"Loaded {gold_case_count()} gold cases from {GOLD_CSV.name}")

    patient_id = sys.argv[1] if len(sys.argv) > 1 else next(iter(_GOLD_INDEX))[0]
    visits = list_visits(patient_id)
    print(f"\n{patient_id}: {len(visits)} visits on disk")

    for v in visits:
        if has_gold(patient_id, v):
            ans = oracle_answer(patient_id, v)
            print(f"\n  {v.visit_folder} (StudyTime={v.study_time}) -- GOLD FOUND (VisitNum={ans.visit_num})")
            print(f"    Findings:    {ans.findings[:80]}...")
            print(f"    Impressions: {ans.impressions[:80]}...")
            print(f"    Follow-up:   {ans.follow_up[:80]}...")
        else:
            print(f"\n  {v.visit_folder} (StudyTime={v.study_time}) -- no gold (exclude from eval)")
