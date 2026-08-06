"""
regrade_documentation_fix_batch.py

Re-grades the same 10 episodes from the original stratified batch (same
patients, same visits, same conditions - no new agent runs) using the
documentation-loss fix in grade_episode.py: typed-but-never-saved content
now gets graded on Accuracy instead of being treated as if nothing was
written at all (Process still penalizes the failure to save, -0.25, via
grade_process.py's "Documentation actually saved" check - unchanged).

Deliberately writes to ISOLATED output files, not the shared
all_receipts.csv / per-episode receipt.csv - those already back the
report that shipped, and must not be silently altered or mixed with this
re-grade. Output:
    computer_use_eval/regraded_fixed_batch/<episode_dir>/receipt.csv
    computer_use_eval/regraded_fixed_batch/<episode_dir>/receipt.json
    computer_use_eval/regraded_fixed_batch/all_receipts.csv
    computer_use_eval/regraded_fixed_batch/all_receipts.jsonl

Real judge() LLM calls - not free, not instant. Same 10 (patient_id,
visit_date) pairs as the original batch.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grade_process import grade_process, read_all_note_forms, long_typed_texts
from judge import judge, AccuracyJudgement
from grade_episode import (
    _load_episode_summary, _extract_written_note, _split_typed_documentation,
    _flatten_findings, _flatten_impression, _flatten_actions, _lookup_gold,
)

EPISODES_DIR = Path(__file__).resolve().parent / "episodes"
OUT_DIR = Path(__file__).resolve().parent / "regraded_fixed_batch"
OUT_DIR.mkdir(exist_ok=True)

OUT_ALL_CSV = OUT_DIR / "all_receipts.csv"
OUT_ALL_JSONL = OUT_DIR / "all_receipts.jsonl"
_ALL_FIELDS = ["patient_id", "visit_date", "episode_dir", "axis", "item", "component", "points", "note"]

EPISODE_DIRS = [
    "GRDN004RP5BFHE0T_2009-06-01_1785991219",
    "GRDN004RP5BFHE0T_2009-06-15_1785992551",
    "GRDN004RP5BFHE0T_2011-09-01_1785989933",
    "GRDN006MCYEVYFE8_2011-07-25_1785992166",
    "GRDN006VFINFR877_2013-03-25_1785991582",
    "GRDN00BE97VCHW4E_2025-01-05_1785992661",
    "GRDN00BJ2R8QRO0L_2013-04-02_1785990708",
    "GRDN00BJ2R8QRO0L_2016-03-29_1785990362",
    "GRDN00BKCEOKC7S1_2024-01-12_1785991400",
    "GRDN00BKCEOKC7S1_2025-02-12_1785991963",
]

_NON_METRIC_FIELDS = {"finding_description", "diagnosis_description", "action_description", "match_status", "reasoning"}


def _explode(rows: list[dict], axis_label: str, item, description: str) -> None:
    data = item.model_dump()
    for field, value in data.items():
        if field in _NON_METRIC_FIELDS:
            continue
        rows.append({"axis": axis_label, "item": f"[{item.match_status}] {description}",
                      "component": field, "points": value, "note": item.reasoning})
    rows.append({"axis": axis_label, "item": f"[{item.match_status}] {description}",
                  "component": "TOTAL (sum of components above)", "points": round(item.raw_total, 2), "note": item.reasoning})


def regrade_one(episode_dir_name: str) -> dict:
    episode_dir = EPISODES_DIR / episode_dir_name
    log_path = episode_dir / "log.jsonl"
    ep = _load_episode_summary(log_path)
    patient_id, visit_date = ep["patient_id"], ep["visit_date"]

    gold = _lookup_gold(patient_id, visit_date)
    if gold is None:
        raise ValueError(f"No gold found for {patient_id}/{visit_date}")

    process_items = grade_process(log_path, patient_id, visit_date, gold.gold_action_ids)

    saved_notes = read_all_note_forms(patient_id, visit_date)
    model_findings, model_impressions = ("", "")
    if saved_notes:
        model_findings, model_impressions = _extract_written_note(saved_notes)

    used_fallback = False
    if not model_findings and not model_impressions:
        typed = long_typed_texts(ep["entries"])
        if typed:
            model_findings, model_impressions = _split_typed_documentation(typed)
            used_fallback = True
        else:
            model_findings = _flatten_findings(ep["reported_findings"])
            model_impressions = _flatten_impression(ep["reported_impression"])
    model_follow_up = _flatten_actions(ep["selected_actions"])

    accuracy: AccuracyJudgement = judge(
        model_findings=model_findings, model_impressions=model_impressions,
        gold_findings=gold.findings, gold_impressions=gold.impressions,
        model_follow_up=model_follow_up, gold_action_ids=gold.gold_action_ids,
        gold_recommendation=gold.gold_recommendation,
    )

    rows = []
    for item in process_items:
        rows.append({"axis": "Process", "item": item.step, "component": "", "points": round(item.points, 2), "note": item.note})
    for item in accuracy.findings.items:
        _explode(rows, "A1 Findings", item, item.finding_description)
    for item in accuracy.impressions.items:
        _explode(rows, "A2 Impressions", item, item.diagnosis_description)
    if accuracy.follow_up:
        for item in accuracy.follow_up.items:
            _explode(rows, "A3 Follow-up", item, item.action_description)

    z1_total = sum(i.points for i in process_items if i.step.startswith("Z1"))
    z2_total = sum(i.points for i in process_items if i.step.startswith("Z2"))
    totals = {
        "Z1 (Information-Gathering) total": round(z1_total, 2),
        "Z2 (Action-Execution) total": round(z2_total, 2),
        "Process total (Z1+Z2)": round(z1_total + z2_total, 2),
        "A1 (Findings) total": round(accuracy.a1_score, 2),
        "A2 (Impressions) total": round(accuracy.a2_score, 2),
        "A3 (Follow-up) total": round(accuracy.a3_score, 2) if accuracy.a3_score is not None else "",
        "Accuracy total (A1+A2+A3)": round(accuracy.total_score, 2),
    }
    for label, value in totals.items():
        rows.append({"axis": "TOTAL", "item": label, "component": "", "points": value, "note": ""})

    out_ep_dir = OUT_DIR / episode_dir_name
    out_ep_dir.mkdir(exist_ok=True)
    with (out_ep_dir / "receipt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["axis", "item", "component", "points", "note"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_ep_dir / "receipt.json").open("w") as f:
        json.dump({"patient_id": patient_id, "visit_date": visit_date, "used_typed_fallback": used_fallback,
                    "rows": rows, "totals": totals}, f, indent=2)

    write_header = not OUT_ALL_CSV.exists()
    with OUT_ALL_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"patient_id": patient_id, "visit_date": visit_date, "episode_dir": episode_dir_name, **row})
    with OUT_ALL_JSONL.open("a") as f:
        for row in rows:
            f.write(json.dumps({"patient_id": patient_id, "visit_date": visit_date, "episode_dir": episode_dir_name, **row}) + "\n")

    print(f"{episode_dir_name}: used_typed_fallback={used_fallback}  totals={totals}")
    return {"episode_dir": episode_dir_name, "used_typed_fallback": used_fallback, "totals": totals}


if __name__ == "__main__":
    if OUT_ALL_CSV.exists():
        OUT_ALL_CSV.unlink()
    if OUT_ALL_JSONL.exists():
        OUT_ALL_JSONL.unlink()
    results = []
    for ep_dir in EPISODE_DIRS:
        try:
            results.append(regrade_one(ep_dir))
        except Exception as e:  # noqa: BLE001
            print(f"{ep_dir}: FAILED - {e}")
    print("\n=== Summary ===")
    for r in results:
        print(r)
