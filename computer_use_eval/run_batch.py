"""
run_batch.py

Runs agent_episode.py + grade_episode.py across a fixed batch of visits,
one after another. Real cost per visit (real API calls, ~6-14 min each
based on actual timing from this session's test runs) - PILOT_VISITS is
deliberately small (~1.5-2 hour budget = 10 visits) rather than the full
152 (which would take ~15-35+ hours run sequentially).

10 visits fits ALL 10 real clinical conditions in the dataset exactly
once each (Normal, Cardiomegaly, Atelectasis, Effusion, Consolidation,
Emphysema, Pneumothorax, Edema, Pneumonia, Lung opacity) - full condition
coverage, picked via oracle.py's gold index.

Each pick was checked against a real, confirmed data issue: some patients
have two separate visits on the same calendar date (e.g. GRDN00BKCEOKC7S1
has two on 2024-05-25) - since agent_episode.py's --visit-date only takes
a date, not a full timestamp, and the frontend's date dropdown shows both
same-day visits with an identical label, Playwright's label-based
select_option can't reliably disambiguate which one gets tested. Every
visit below was verified to have no such same-day collision for its
patient before being included.

If an episode fails to complete (crashes, never reaches select_action),
it's logged and the batch continues - a failed episode is itself real
signal (an execution failure), not something to silently skip, but one
bad run shouldn't stop the whole batch.

Usage:
    python run_batch.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PILOT_VISITS = [
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

MAX_STEPS = 50


def latest_episode_dir(patient_id: str) -> Path | None:
    matches = sorted(
        (HERE / "episodes").glob(f"{patient_id}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def run_visit(patient_id: str, visit_date: str) -> dict:
    print(f"\n=== {patient_id} / {visit_date} ===")
    result = {"patient_id": patient_id, "visit_date": visit_date, "episode_ok": False, "grade_ok": False}

    episode_proc = subprocess.run(
        [sys.executable, str(HERE / "agent_episode.py"),
         "--patient-id", patient_id, "--visit-date", visit_date, "--max-steps", str(MAX_STEPS)],
        cwd=HERE, capture_output=True, text=True,
    )
    if episode_proc.returncode != 0:
        print(f"  EPISODE FAILED (exit {episode_proc.returncode}): {episode_proc.stderr[-500:]}")
        result["error"] = episode_proc.stderr[-1000:]
        return result
    result["episode_ok"] = True

    episode_dir = latest_episode_dir(patient_id)
    if episode_dir is None:
        print("  No episode directory found after run - unexpected.")
        return result
    result["episode_dir"] = episode_dir.name
    print(f"  Episode complete: {episode_dir.name}")

    grade_proc = subprocess.run(
        [sys.executable, str(HERE / "grade_episode.py"), str(episode_dir)],
        cwd=HERE, capture_output=True, text=True,
    )
    if grade_proc.returncode != 0:
        print(f"  GRADING FAILED (exit {grade_proc.returncode}): {grade_proc.stderr[-500:]}")
        result["error"] = grade_proc.stderr[-1000:]
        return result
    result["grade_ok"] = True
    print("  Graded and appended to all_receipts.")
    return result


def main() -> None:
    results = [run_visit(pid, date) for pid, date in PILOT_VISITS]

    print("\n\n=== Batch summary ===")
    for r in results:
        status = "OK" if r["grade_ok"] else ("episode ok, grading failed" if r["episode_ok"] else "EPISODE FAILED")
        print(f"  {r['patient_id']}/{r['visit_date']}: {status}")

    n_ok = sum(1 for r in results if r["grade_ok"])
    print(f"\n{n_ok}/{len(results)} visits fully graded and in all_receipts.")


if __name__ == "__main__":
    main()
