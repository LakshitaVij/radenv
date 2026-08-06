"""
grade_episode.py

Orchestrates full grading for one agent_episode.py run: Process Axis
(via grade_process.py) + Accuracy Axis (via judge.py, using THIS
episode's own output - not a separate text-generation call). Writes one
combined receipt per episode - every Z1/Z2 step and every A1/A2/A3 scored
item as rows, plus a totals summary.

Findings/impressions grading prefers the real SOAP note (Objective/
Assessment) the agent wrote into OpenEMR using its ordinary click/
type_text tools - a genuine per-patient/per-encounter written artifact,
not just our own tool-call capture (report_findings/report_impression
still exist and are used as a fallback for episodes with no SOAP note).

Every graded episode ALSO gets appended to a single running store
(all_receipts.csv / all_receipts.jsonl, in this directory) - same rows as
the per-episode receipt, plus patient_id/visit_date/episode_dir columns
for traceability, so results across many episodes can be opened/queried
together in one place instead of hunting through each episode's own
folder. The per-episode receipt still gets written too - this is
additive, not a replacement.

This is the "marriage" point: A3 follow-up scoring runs against the
agent's real select_action tool calls (real clicks in Configure Orders
and Results), matched against oracle.py's gold_action_ids for the same
visit - not free text.

Usage:
    python grade_episode.py episodes/<patient>_<visit>_<timestamp>/
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for judge.py / oracle.py

from grade_process import grade_process, ScoreItem, read_all_note_forms, split_three, all_typed_text
from judge import judge, judge_followup, AccuracyJudgement, FollowupJudgement
from oracle import _GOLD_INDEX, GoldAnswer

FOLLOWUP_CSV = Path(__file__).resolve().parent.parent / "generated_followups_fin.csv"

# Per note-form, which real fields best correspond to Findings vs.
# Impression - used to pick content out of whichever form(s) the agent
# actually chose (the prompt no longer tells it which one). If more than
# one form has content, all of it is concatenated per section rather than
# arbitrarily picking one - real content shouldn't be silently dropped.
_FINDINGS_FIELDS = {
    "soap": ["objective"], "clinical_notes": ["description"],
    "clinic_note": ["history", "examination"], "note": ["message"],
    "clinical_instructions": ["instruction"],
}
_IMPRESSION_FIELDS = {
    "soap": ["assessment"], "clinical_notes": [], "clinic_note": [], "note": [], "clinical_instructions": [],
}


def _extract_written_note(saved_notes: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    """Best-effort split of whatever the agent actually wrote across
    whichever real note form(s) it used into (findings_text,
    impression_text, written_followup_text). 'soap' cleanly separates
    assessment (impression) from the rest via its own field - findings and
    any written follow-up both fall under 'objective' there, so that field
    is still run through split_three too, to pull out a written follow-up
    section if the agent included one. Other forms are single free-text
    fields, split entirely via split_three (grade_process.py) on the
    'Findings:'/'Impression:'/'Follow-up:' markers the system prompt asks
    for - nothing gets graded unless the agent actually labeled it."""
    findings_parts, impression_parts, followup_parts = [], [], []
    for formdir, fields in saved_notes.items():
        for field, value in fields.items():
            if not value.strip():
                continue
            if field in _IMPRESSION_FIELDS.get(formdir, []):
                impression_parts.append(value)
                continue
            f, i, fu = split_three(value)
            if f:
                findings_parts.append(f)
            if i:
                impression_parts.append(i)
            if fu:
                followup_parts.append(fu)
    return (
        "\n\n".join(p for p in findings_parts if p),
        "\n\n".join(p for p in impression_parts if p),
        "\n\n".join(p for p in followup_parts if p),
    )


def _study_time_for(patient_id: str, study_date: str) -> str | None:
    """oracle.py's gold is keyed on (patient_id, study_date, study_time),
    but agent_episode.py only knows (patient_id, visit_date as YYYY-MM-DD)
    - look up the matching StudyTime by PatientID+StudyDate."""
    import csv as _csv
    with FOLLOWUP_CSV.open() as f:
        for row in _csv.DictReader(f):
            if row["PatientID"] == patient_id and row["StudyDate"] == study_date:
                return row["StudyTime"]
    return None


def _lookup_gold(patient_id: str, visit_date: str) -> GoldAnswer | None:
    study_date = visit_date.replace("-", "")  # YYYY-MM-DD -> YYYYMMDD
    study_time = _study_time_for(patient_id, study_date)
    if study_time is None:
        return None
    return _GOLD_INDEX.get((patient_id, study_date, study_time))


def _flatten_findings(findings: list[dict]) -> str:
    lines = []
    for f in findings:
        lines.append(
            f"- {f.get('description', '')} (laterality: {f.get('laterality', '')}, "
            f"severity: {f.get('severity', '')}, location: {f.get('location', '')}, "
            f"confidence: {f.get('confidence', '')}): {f.get('reasoning', '')}"
        )
    return "\n".join(lines)


def _flatten_impression(diagnoses: list[dict]) -> str:
    lines = []
    for d in diagnoses:
        tag = "PRIMARY" if d.get("is_primary") else "secondary"
        lines.append(f"- [{tag}] {d.get('diagnosis', '')} (confidence: {d.get('confidence', '')}): {d.get('reasoning', '')}")
    return "\n".join(lines)


def _flatten_actions(selected_actions: list[dict]) -> str:
    lines = []
    for a in selected_actions:
        path = " -> ".join(x for x in [a.get("level_1"), a.get("level_2"), a.get("level_3")] if x)
        lines.append(f"- {path}: {a.get('reasoning', '')}")
    return "\n".join(lines)


def _load_episode_summary(log_path: Path) -> dict:
    with log_path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]
    summary = next((e for e in reversed(entries) if e.get("type") == "episode_summary"), None)
    start = next((e for e in entries if e.get("type") == "episode_start"), {})
    if summary is None:
        raise ValueError(f"No episode_summary entry found in {log_path} - episode may not have completed.")
    return {
        "patient_id": start.get("patient_id"),
        "visit_date": start.get("visit_date"),
        "reported_findings": summary.get("reported_findings", []),
        "reported_impression": summary.get("reported_impression", []),
        "selected_actions": summary.get("selected_actions", []),
        "entries": entries,
    }


def grade_episode(episode_dir: Path) -> dict:
    log_path = episode_dir / "log.jsonl"
    ep = _load_episode_summary(log_path)
    patient_id, visit_date = ep["patient_id"], ep["visit_date"]

    gold = _lookup_gold(patient_id, visit_date)
    if gold is None:
        raise ValueError(f"No gold found for {patient_id}/{visit_date} - can't grade Accuracy Axis for this visit.")

    process_items = grade_process(log_path, patient_id, visit_date, gold.gold_action_ids)

    # Prefer whatever real note form(s) the agent actually wrote into
    # OpenEMR (genuine per-patient/per-encounter artifacts) over our own
    # tool-call capture for findings/impressions - checks every real note
    # form in the registry, not just SOAP, since the prompt no longer tells
    # it which one to use.
    saved_notes = read_all_note_forms(patient_id, visit_date)
    model_findings, model_impressions, model_written_followup = ("", "", "")
    if saved_notes:
        model_findings, model_impressions, model_written_followup = _extract_written_note(saved_notes)

    # Nothing saved (or a save that came back empty) - fall back to
    # whatever the agent actually typed, even though it never got saved.
    # Process still penalizes this (-0.25, "Documentation actually saved"
    # in grade_process.py) - but a correct answer that got lost shouldn't
    # ALSO be graded identically to no answer at all on the Accuracy axis.
    # split_three() only grades content the agent actually labeled
    # Findings:/Impression:/Follow-up: - no length heuristics, no risk of
    # a login credential or patient ID slipping in, since neither is ever
    # typed under one of those labels. Only falls further back to the old
    # report_findings/report_impression tool-call capture (empty for any
    # episode run after those tools were removed) if nothing was labeled
    # as documentation at all.
    if not model_findings and not model_impressions and not model_written_followup:
        model_findings, model_impressions, model_written_followup = split_three(all_typed_text(ep["entries"]))
        if not model_findings and not model_impressions and not model_written_followup:
            model_findings = _flatten_findings(ep["reported_findings"])
            model_impressions = _flatten_impression(ep["reported_impression"])

    # A3 is graded TWICE, independently, per explicit user decision:
    # (1) "A3 Follow-up (actions)" - the original "marriage" point, tied to
    #     the agent's real select_action clicks, unchanged. Can't be faked
    #     by writing about a follow-up without actually clicking it.
    # (2) "A3 Follow-up (written)" - the agent's actual written Follow-up
    #     note text (if any), graded against the same gold independently.
    #     These two scores are NOT combined into one - kept as separate
    #     axes so a mismatch between what was clicked and what was written
    #     stays visible rather than averaged away.
    model_follow_up_actions = _flatten_actions(ep["selected_actions"])

    accuracy: AccuracyJudgement = judge(
        model_findings=model_findings,
        model_impressions=model_impressions,
        gold_findings=gold.findings,
        gold_impressions=gold.impressions,
        model_follow_up=model_follow_up_actions,
        gold_action_ids=gold.gold_action_ids,
        gold_recommendation=gold.gold_recommendation,
    )

    written_followup_judgement: FollowupJudgement | None = None
    if model_written_followup:
        written_followup_judgement = judge_followup(model_written_followup, gold.gold_action_ids, gold.gold_recommendation)

    rows = []
    for item in process_items:
        rows.append({"axis": "Process", "item": item.step, "component": "", "points": round(item.points, 2), "note": item.note})

    # Non-metric fields on each Pydantic item - everything else is a real
    # scored sub-metric and gets its own row, per explicit request to show
    # every component individually rather than folding into one sum.
    _NON_METRIC_FIELDS = {"finding_description", "diagnosis_description", "action_description", "match_status", "reasoning"}

    def _explode(axis_label: str, item, description: str):
        data = item.model_dump()
        for field, value in data.items():
            if field in _NON_METRIC_FIELDS:
                continue
            rows.append({
                "axis": axis_label, "item": f"[{item.match_status}] {description}",
                "component": field, "points": value, "note": item.reasoning,
            })
        rows.append({
            "axis": axis_label, "item": f"[{item.match_status}] {description}",
            "component": "TOTAL (sum of components above)", "points": round(item.raw_total, 2), "note": item.reasoning,
        })

    for item in accuracy.findings.items:
        _explode("A1 Findings", item, item.finding_description)
    for item in accuracy.impressions.items:
        _explode("A2 Impressions", item, item.diagnosis_description)
    if accuracy.follow_up:
        for item in accuracy.follow_up.items:
            _explode("A3 Follow-up (actions)", item, item.action_description)
    if written_followup_judgement:
        for item in written_followup_judgement.items:
            _explode("A3 Follow-up (written)", item, item.action_description)

    written_followup_score = (
        sum(i.raw_total for i in written_followup_judgement.items) / len(written_followup_judgement.items)
        if written_followup_judgement and written_followup_judgement.items else None
    )

    z1_total = sum(i.points for i in process_items if i.step.startswith("Z1"))
    z2_total = sum(i.points for i in process_items if i.step.startswith("Z2"))
    totals = {
        "Z1 (Information-Gathering) total": round(z1_total, 2),
        "Z2 (Action-Execution) total": round(z2_total, 2),
        "Process total (Z1+Z2)": round(z1_total + z2_total, 2),
        "A1 (Findings) total": round(accuracy.a1_score, 2),
        "A2 (Impressions) total": round(accuracy.a2_score, 2),
        "A3 (Follow-up - actions) total": round(accuracy.a3_score, 2) if accuracy.a3_score is not None else "",
        "A3 (Follow-up - written) total": round(written_followup_score, 2) if written_followup_score is not None else "",
        "Accuracy total (A1+A2+A3-actions)": round(accuracy.total_score, 2),
    }
    for label, value in totals.items():
        rows.append({"axis": "TOTAL", "item": label, "component": "", "points": value, "note": ""})

    csv_path = episode_dir / "receipt.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["axis", "item", "component", "points", "note"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = episode_dir / "receipt.json"
    with json_path.open("w") as f:
        json.dump({"patient_id": patient_id, "visit_date": visit_date, "rows": rows, "totals": totals}, f, indent=2)

    _append_to_combined_store(patient_id, visit_date, episode_dir.name, rows)

    return {"rows": rows, "totals": totals, "csv_path": csv_path, "json_path": json_path}


ALL_RECEIPTS_CSV = Path(__file__).resolve().parent / "all_receipts.csv"
ALL_RECEIPTS_JSONL = Path(__file__).resolve().parent / "all_receipts.jsonl"
_ALL_RECEIPTS_FIELDS = ["patient_id", "visit_date", "episode_dir", "axis", "item", "component", "points", "note"]


def _append_to_combined_store(patient_id: str, visit_date: str, episode_dir_name: str, rows: list[dict]) -> None:
    """Appends this episode's rows to the running cross-episode store -
    additive, never overwrites, so results across many episodes/runs
    accumulate in one place rather than being scattered one-per-folder."""
    write_header = not ALL_RECEIPTS_CSV.exists()
    with ALL_RECEIPTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_RECEIPTS_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"patient_id": patient_id, "visit_date": visit_date, "episode_dir": episode_dir_name, **row})

    with ALL_RECEIPTS_JSONL.open("a") as f:
        for row in rows:
            f.write(json.dumps({"patient_id": patient_id, "visit_date": visit_date, "episode_dir": episode_dir_name, **row}) + "\n")


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    result = grade_episode(episode_dir)
    for row in result["rows"]:
        comp = f" :: {row['component']}" if row.get("component") else ""
        print(f"  {row['points']:+.2f}  [{row['axis']}] {row['item']}{comp}")
    print()
    for label, value in result["totals"].items():
        print(f"{label}: {value}")
    print(f"\nReceipt written to {result['csv_path']} and {result['json_path']}")
