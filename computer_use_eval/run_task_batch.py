"""
run_task_batch.py

Runs the new 8-task designed evaluation set (replaces the old 10-visit
condition-stratified sample). Each task was chosen and verified against
real gold data tonight to target a specific, named failure mode - see
the plan doc for the full rationale per task.

Carries every grading/prompt fix from tonight automatically, since this
calls the real agent_episode.py and grade_episode.grade_episode() -
documentation-loss fallback, marker-only Findings/Impression/Follow-up
grading, Z2.15/Z2.16 checkpoints, dual A3 grading, the thought-signature
retry mitigation.

Resilience: same as run_full_batch_v2.py - up to 3 attempts per task if
an attempt ends almost immediately (signature of an infra failure, not
real model behavior).

Isolation: grading writes to task_batch_v1/all_receipts.csv, never
touches the original shipped all_receipts.csv or episodes/ receipts.

Real cost: 8 real agent episodes (OpenRouter calls) + 8 real gradings
(OpenAI judge calls). Not free, not fast.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grade_episode  # noqa: E402

EPISODES_DIR = Path(__file__).resolve().parent / "episodes"
OUT_DIR = Path(__file__).resolve().parent / "task_batch_v1"
OUT_DIR.mkdir(exist_ok=True)

PYTHON = "/opt/homebrew/bin/python3.10"
AGENT_SCRIPT = Path(__file__).resolve().parent / "agent_episode.py"
MAX_STEPS = 50
MAX_ATTEMPTS_PER_VISIT = 3
MIN_REAL_ENTRIES = 4  # episode_start + setup + >=1 real step + something else; fewer = early infra failure

# (task_num, task_name, patient_id, visit_date) - visit_date is the
# pre-selected "current" visit; Task 7's real prior (V6) is something the
# agent must navigate to on its own inside OpenEMR, not pre-selected.
TASKS = [
    (1, "Documentation completeness / secondary-finding omission", "GRDN00BJ2R8QRO0L", "2009-01-01"),
    (2, "Multi-finding conflation/hallucination trap", "GRDN004RP5BFHE0T", "2009-05-28"),
    (3, "Under-escalation, single clear finding", "GRDN006MCYEVYFE8", "2011-07-25"),
    (4, "Under-escalation + prioritization, 3 findings at once", "GRDN00BKCEOKC7S1", "2025-07-23"),
    (5, "Calibrated uncertainty, hedge-following, recommend CT", "GRDN00BE97VCHW4E", "2025-01-05"),
    (6, "Calibrated non-urgency, recommend-vs-place-vs-save chain", "GRDN007ELLXXY3OJ", "2015-08-08"),
    (7, "Stability bias, wrong-prior trap, unprompted comparison", "GRDN004RP5BFHE0T", "2009-05-30"),
    (8, "Anchoring-bias resistance", "GRDN00A5ZYWHZY7D", "2012-05-09"),
]


def run_one_episode(patient_id: str, visit_date: str) -> Path | None:
    for attempt in range(1, MAX_ATTEMPTS_PER_VISIT + 1):
        before = {p.name for p in EPISODES_DIR.glob(f"{patient_id}_*")}
        result = subprocess.run(
            [PYTHON, str(AGENT_SCRIPT), "--patient-id", patient_id, "--visit-date", visit_date,
             "--max-steps", str(MAX_STEPS)],
            capture_output=True, text=True,
        )
        after = {p.name for p in EPISODES_DIR.glob(f"{patient_id}_*")}
        new_dirs = after - before
        if not new_dirs:
            print(f"  attempt {attempt}: agent_episode.py crashed before creating an episode dir. stderr tail: {result.stderr[-500:]}")
            time.sleep(5)
            continue
        ep_dir = EPISODES_DIR / sorted(new_dirs)[-1]
        log_path = ep_dir / "log.jsonl"
        n_entries = sum(1 for _ in log_path.open()) if log_path.exists() else 0
        if n_entries < MIN_REAL_ENTRIES:
            print(f"  attempt {attempt}: episode ended almost immediately ({n_entries} log entries) - likely infra failure, retrying")
            time.sleep(5)
            continue
        print(f"  attempt {attempt}: succeeded, {n_entries} log entries, {ep_dir.name}")
        return ep_dir
    print(f"  FAILED after {MAX_ATTEMPTS_PER_VISIT} attempts - giving up on this task")
    return None


if __name__ == "__main__":
    grade_episode.ALL_RECEIPTS_CSV = OUT_DIR / "all_receipts.csv"
    grade_episode.ALL_RECEIPTS_JSONL = OUT_DIR / "all_receipts.jsonl"
    if grade_episode.ALL_RECEIPTS_CSV.exists():
        grade_episode.ALL_RECEIPTS_CSV.unlink()
    if grade_episode.ALL_RECEIPTS_JSONL.exists():
        grade_episode.ALL_RECEIPTS_JSONL.unlink()

    results = []
    for task_num, task_name, patient_id, visit_date in TASKS:
        print(f"\n=== Task {task_num}: {task_name} ({patient_id} / {visit_date}) ===")
        ep_dir = run_one_episode(patient_id, visit_date)
        if ep_dir is None:
            results.append({"task_num": task_num, "task_name": task_name, "patient_id": patient_id,
                             "visit_date": visit_date, "status": "episode_failed"})
            continue
        try:
            graded = grade_episode.grade_episode(ep_dir)
            print(f"  graded OK: {graded['totals']}")
            results.append({"task_num": task_num, "task_name": task_name, "patient_id": patient_id,
                             "visit_date": visit_date, "status": "graded",
                             "episode_dir": ep_dir.name, "totals": graded["totals"]})
        except Exception as e:  # noqa: BLE001
            print(f"  grading FAILED: {e}")
            results.append({"task_num": task_num, "task_name": task_name, "patient_id": patient_id,
                             "visit_date": visit_date, "status": "grading_failed",
                             "episode_dir": ep_dir.name, "error": str(e)})

    print("\n\n=== FINAL SUMMARY ===")
    for r in results:
        print(json.dumps(r))
    with (OUT_DIR / "run_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
