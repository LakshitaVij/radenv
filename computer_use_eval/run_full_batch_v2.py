"""
run_full_batch_v2.py

Runs a fresh batch of 10 brand-new agent episodes (same 10 patients/visits/
conditions as the original stratified batch) with every fix from tonight
applied: documentation-loss grading fallback, marker-only Findings/
Impression/Follow-up splitting, Z2.16 step efficiency, dual A3 grading
(actions vs. written), and the thought-signature retry mitigation.

Resilience: given real connectivity flakiness observed tonight (OpenRouter
TLS resets, mid-episode drops), each visit gets up to 3 total attempts if
an attempt ends almost immediately (<=3 real log entries beyond setup -
the signature of an early infra failure, not real model behavior).

Isolation: grading writes to run_full_batch_v2/all_receipts.csv (isolated,
via the same monkeypatch approach as regrade_full_batch_v2.py) - never
touches the original shipped all_receipts.csv or episodes/ receipts.
Agent episodes themselves land in the normal episodes/ dir (harmless,
timestamped, no collision with the original batch's episode dirs).

Real cost: 10 real agent episodes (OpenRouter calls) + 10 real gradings
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
OUT_DIR = Path(__file__).resolve().parent / "run_full_batch_v2"
OUT_DIR.mkdir(exist_ok=True)

PYTHON = "/opt/homebrew/bin/python3.10"
AGENT_SCRIPT = Path(__file__).resolve().parent / "agent_episode.py"
MAX_STEPS = 50
MAX_ATTEMPTS_PER_VISIT = 3
MIN_REAL_ENTRIES = 4  # episode_start + setup + >=1 real step + something else; fewer = early infra failure

VISITS = [
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


def latest_episode_dir(patient_id: str) -> Path | None:
    candidates = sorted(EPISODES_DIR.glob(f"{patient_id}_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


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
    print(f"  FAILED after {MAX_ATTEMPTS_PER_VISIT} attempts - giving up on this visit")
    return None


if __name__ == "__main__":
    grade_episode.ALL_RECEIPTS_CSV = OUT_DIR / "all_receipts.csv"
    grade_episode.ALL_RECEIPTS_JSONL = OUT_DIR / "all_receipts.jsonl"
    if grade_episode.ALL_RECEIPTS_CSV.exists():
        grade_episode.ALL_RECEIPTS_CSV.unlink()
    if grade_episode.ALL_RECEIPTS_JSONL.exists():
        grade_episode.ALL_RECEIPTS_JSONL.unlink()

    results = []
    for patient_id, visit_date in VISITS:
        print(f"\n=== {patient_id} / {visit_date} ===")
        ep_dir = run_one_episode(patient_id, visit_date)
        if ep_dir is None:
            results.append({"patient_id": patient_id, "visit_date": visit_date, "status": "episode_failed"})
            continue
        try:
            graded = grade_episode.grade_episode(ep_dir)
            print(f"  graded OK: {graded['totals']}")
            results.append({"patient_id": patient_id, "visit_date": visit_date, "status": "graded",
                             "episode_dir": ep_dir.name, "totals": graded["totals"]})
        except Exception as e:  # noqa: BLE001
            print(f"  grading FAILED: {e}")
            results.append({"patient_id": patient_id, "visit_date": visit_date, "status": "grading_failed",
                             "episode_dir": ep_dir.name, "error": str(e)})

    print("\n\n=== FINAL SUMMARY ===")
    for r in results:
        print(json.dumps(r))
    with (OUT_DIR / "run_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
