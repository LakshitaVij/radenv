"""
regrade_full_batch_v2.py

Re-grades the same 10 episodes (same patients, same visits, same
conditions - no new agent runs) using grade_episode.py exactly as it
stands right now: documentation-loss fallback + Z2.16 step efficiency +
corrected saved-note Impression extraction + dual A3 grading (actions vs.
written) + the Findings label strip. Calls the REAL grade_episode()
directly (not a duplicated copy of its logic - that's what went stale
last time) so this can never drift out of sync with the actual grading
code again.

Isolation: copies each episode's log.jsonl into
regraded_full_batch_v2/episodes/<name>/log.jsonl and monkeypatches
grade_episode's ALL_RECEIPTS_CSV/JSONL globals to point inside
regraded_full_batch_v2/ for the duration of the run - the original
episodes/ folder and the original all_receipts.csv are never touched.
Real judge() LLM calls - not free, not instant.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grade_episode  # noqa: E402

SRC_EPISODES_DIR = Path(__file__).resolve().parent / "episodes"
OUT_DIR = Path(__file__).resolve().parent / "regraded_full_batch_v2"
OUT_EPISODES_DIR = OUT_DIR / "episodes"

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

if __name__ == "__main__":
    OUT_EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    # Redirect the combined-store output before any grading runs, so
    # _append_to_combined_store() (called inside the real grade_episode())
    # writes into the isolated folder instead of the shared all_receipts.csv.
    grade_episode.ALL_RECEIPTS_CSV = OUT_DIR / "all_receipts.csv"
    grade_episode.ALL_RECEIPTS_JSONL = OUT_DIR / "all_receipts.jsonl"
    if grade_episode.ALL_RECEIPTS_CSV.exists():
        grade_episode.ALL_RECEIPTS_CSV.unlink()
    if grade_episode.ALL_RECEIPTS_JSONL.exists():
        grade_episode.ALL_RECEIPTS_JSONL.unlink()

    results = []
    for ep_dir_name in EPISODE_DIRS:
        src_log = SRC_EPISODES_DIR / ep_dir_name / "log.jsonl"
        out_ep_dir = OUT_EPISODES_DIR / ep_dir_name
        out_ep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_log, out_ep_dir / "log.jsonl")

        try:
            result = grade_episode.grade_episode(out_ep_dir)
            print(f"{ep_dir_name}: totals={result['totals']}")
            results.append((ep_dir_name, result["totals"]))
        except Exception as e:  # noqa: BLE001
            print(f"{ep_dir_name}: FAILED - {e}")

    print("\n=== Summary ===")
    for ep_dir_name, totals in results:
        print(ep_dir_name, totals)
