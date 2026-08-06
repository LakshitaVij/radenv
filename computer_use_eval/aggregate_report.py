"""
aggregate_report.py

Reads all_receipts.jsonl (the running cross-episode store grade_episode.py
appends to) and produces a PhysicianBench-style failure-mode analysis:
a fixed, rule-based taxonomy tag on every negative-scoring row, then
frequency tables, a near-miss distribution, and an execution-vs-clinical
split - same shape as the report structure discussed earlier this session.

Classification is entirely rule-based off fields judge.py/grade_process.py
already compute (match_status, which component went negative, which
Process step failed) - no extra LLM calls, no new judgment layer. Per the
explicit design decision: deterministic and free over nuanced-but-costly.

Usage:
    python aggregate_report.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for oracle.py
from grade_episode import _lookup_gold  # reuse the existing patient_id/visit_date -> GoldAnswer lookup

ALL_RECEIPTS_CSV = Path(__file__).resolve().parent / "all_receipts.csv"
AGGREGATE_REPORT_CSV = Path(__file__).resolve().parent / "aggregate_report.csv"
_CSV_FIELDS = ["table", "label", "count", "total", "rate", "note"]

# Process-axis tags that represent the agent failing to even get where it
# needed to go/log in - "non-clinical" execution failures, kept separate
# from clinical/reasoning quality so one doesn't contaminate the other,
# same principle as PhysicianBench's "5 tasks failed for non-clinical
# reasons" split.
EXECUTION_TAGS = {
    "process_never_opened_openemr", "process_never_logged_in", "process_no_xray_interaction",
    "process_wrong_patient", "process_wrong_encounter", "process_encounter_not_reached",
    "process_orders_screen_not_reached",
}

_AXIS_NOUN = {"A1 Findings": "finding", "A2 Impressions": "diagnosis", "A3 Follow-up": "action"}
_MATCH_STATUS_RE = re.compile(r"^\[(\w+)\]")


def classify(row: dict) -> str | None:
    """Returns a failure-mode tag, or None if this row isn't a failure."""
    axis = row["axis"]
    points = float(row["points"]) if row["points"] not in ("", None) else 0.0

    if axis == "TOTAL":
        return None

    if axis == "Process":
        if points >= 0:
            return None
        step = row["item"]
        if "Correct patient" in step:
            return "process_wrong_patient"
        if "Correct encounter" in step:
            return "process_wrong_encounter"
        if "Engages vitals" in step or "Engages history" in step:
            return "process_encounter_not_reached"
        if "Navigate to Procedures" in step or "Navigate to Configuration" in step:
            return "process_orders_screen_not_reached"
        if "Select valid action" in step:
            return "process_invalid_action"
        if "Action count calibration" in step:
            return "process_wrong_action_count"
        if "Step efficiency" in step:
            return "process_step_inefficiency"
        if "Abstention calibration" in step:
            return "process_abstention_miscalibrated"
        if "Interacts" in step:
            return "process_no_xray_interaction"
        if "Clicks View in OpenEMR" in step:
            return "process_never_opened_openemr"
        if "Logs in" in step:
            return "process_never_logged_in"
        if "Documentation actually saved" in step:
            return "documentation_typed_not_saved"
        return "process_other"

    # Accuracy axis rows (A1/A2/A3) - only individual component rows are
    # classified, not the "TOTAL (sum of components above)" subtotal row.
    component = row.get("component", "")
    if component in ("", "TOTAL (sum of components above)"):
        return None
    if points >= 0:
        return None

    m = _MATCH_STATUS_RE.match(row["item"])
    match_status = m.group(1) if m else "unknown"
    noun = _AXIS_NOUN.get(axis, "item")

    if match_status == "missed_by_model":
        return f"missed_{noun}"
    if match_status == "hallucinated_by_model":
        return f"hallucinated_{noun}"
    if match_status == "matched":
        return f"wrong_{component}_{noun}"
    return f"{axis}_other"


def is_scoreable(row: dict) -> bool:
    """A row counts as one checkpoint attempt if it's not a TOTAL/subtotal
    row - i.e. the same rows classify() would tag as pass or fail. A
    checkpoint 'passes' when points >= 0 (0 means 'no penalty' per how the
    point rules are written, e.g. false_positives=0 means 'no
    hallucination' - a real pass, not a missing score)."""
    if row["axis"] == "TOTAL":
        return False
    if row["axis"] != "Process" and row.get("component", "") in ("", "TOTAL (sum of components above)"):
        return False
    return True


def checkpoint_kind(row: dict) -> str:
    if row["axis"] == "Process":
        return "Z1 Information-Gathering" if row["item"].startswith("Z1") else "Z2 Action-Execution"
    return row["axis"]  # "A1 Findings" / "A2 Impressions" / "A3 Follow-up"


def _condition_for(patient_id: str, visit_date: str) -> str:
    """oracle.py's GoldAnswer already carries real condition/phenotype
    labels per visit (from the synthetic clinical-context work earlier
    this session) - reused here, not new data."""
    gold = _lookup_gold(patient_id, visit_date)
    if gold is None:
        return "(unknown - no gold)"
    label = gold.condition or "(no condition label)"
    if gold.phenotype:
        label += f" / {gold.phenotype}"
    return label


def load_rows(episode_filter: set[str] | None = None) -> list[dict]:
    """episode_filter: if given, only rows whose episode_dir is in this set
    are returned - lets a specific batch's analysis exclude earlier ad-hoc
    test/debug episodes that share the same accumulating all_receipts.csv."""
    if not ALL_RECEIPTS_CSV.exists():
        return []
    with ALL_RECEIPTS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    if episode_filter is not None:
        rows = [r for r in rows if r["episode_dir"] in episode_filter]
    return rows


def build_report(episode_filter: set[str] | None = None) -> None:
    rows = load_rows(episode_filter)
    if not rows:
        print(f"No data in {ALL_RECEIPTS_CSV} yet - run run_batch.py first.")
        return

    csv_rows: list[dict] = []  # every table's rows, accumulated for AGGREGATE_REPORT_CSV at the end

    def emit(table: str, label: str, count=None, total=None, rate=None, note: str = "") -> None:
        csv_rows.append({"table": table, "label": label, "count": count, "total": total, "rate": rate, "note": note})

    episodes = sorted({(r["patient_id"], r["visit_date"], r["episode_dir"]) for r in rows})
    print(f"=== Failure-mode analysis: {len(episodes)} episode(s), {len(rows)} scored rows ===\n")
    emit("summary", "episodes", count=len(episodes))
    emit("summary", "scored_rows", count=len(rows))

    tag_counts: Counter[str] = Counter()
    failures_per_episode: dict[str, list[str]] = defaultdict(list)
    execution_failed_episodes: set[str] = set()

    for row in rows:
        tag = classify(row)
        if tag is None:
            continue
        ep_key = row["episode_dir"]
        tag_counts[tag] += 1
        failures_per_episode[ep_key].append(tag)
        if tag in EXECUTION_TAGS:
            execution_failed_episodes.add(ep_key)

    all_episode_keys = {ep[2] for ep in episodes}
    clinical_episodes = all_episode_keys - execution_failed_episodes

    print(f"Episodes with an execution failure (non-clinical - never reached the right screen/patient): "
          f"{len(execution_failed_episodes)}/{len(episodes)}")
    for ep in sorted(execution_failed_episodes):
        print(f"  - {ep}")
    print()
    emit("execution_failures", "episodes_with_execution_failure", count=len(execution_failed_episodes), total=len(episodes))
    for ep in sorted(execution_failed_episodes):
        emit("execution_failures", ep, note="execution failure")

    print("--- Failure-mode frequency (all episodes) ---")
    for tag, count in tag_counts.most_common():
        marker = " [non-clinical]" if tag in EXECUTION_TAGS else ""
        print(f"  {count:3d}  {tag}{marker}")
        emit("failure_mode_frequency", tag, count=count, note="non-clinical" if tag in EXECUTION_TAGS else "clinical")
    print()

    # --- Pass rate per checkpoint kind (denominator = every scoreable
    # row, not just the failures counted above) ---
    kind_total: Counter[str] = Counter()
    kind_passed: Counter[str] = Counter()
    for row in rows:
        if not is_scoreable(row):
            continue
        kind = checkpoint_kind(row)
        kind_total[kind] += 1
        if float(row["points"]) >= 0:
            kind_passed[kind] += 1

    print("--- Pass rate by checkpoint kind ---")
    for kind in sorted(kind_total):
        total, passed = kind_total[kind], kind_passed[kind]
        print(f"  {kind}: {passed}/{total} ({passed / total:.0%})")
        emit("pass_rate_by_checkpoint_kind", kind, count=passed, total=total, rate=round(passed / total, 4))
    print()

    # --- Near-miss distribution: how many checkpoints did each episode
    # miss, binned - the actionable table, same shape as PhysicianBench's
    # "17 tasks missed exactly 1 checkpoint" table. ---
    missed_per_episode: dict[str, int] = {ep: len(failures_per_episode.get(ep, [])) for ep in all_episode_keys}
    print("--- Checkpoints missed per episode (near-miss distribution) ---")
    bins: Counter[int] = Counter(missed_per_episode.values())
    for n_missed in sorted(bins):
        label = "0 (fully passed)" if n_missed == 0 else str(n_missed)
        print(f"  {label}: {bins[n_missed]} episode(s)")
        emit("near_miss_distribution", label, count=bins[n_missed])
    print()
    print("--- Failures per episode (detail) ---")
    for ep_key in sorted(all_episode_keys):
        print(f"  {ep_key}: {missed_per_episode[ep_key]} failure(s)")
        emit("failures_per_episode", ep_key, count=missed_per_episode[ep_key])
    print()

    # --- Episode-level pass/fail: an episode only "passes" if every
    # scoreable checkpoint in it passed - matches PhysicianBench's
    # "a task counts as passed only when every checkpoint passes" rule. ---
    fully_passed = sum(1 for n in missed_per_episode.values() if n == 0)
    print(f"--- Episode-level pass rate ---")
    print(f"  {fully_passed}/{len(all_episode_keys)} episodes fully passed (0 failed checkpoints) "
          f"({fully_passed / len(all_episode_keys):.0%})")
    total_checkpoints = sum(kind_total.values())
    total_passed = sum(kind_passed.values())
    print(f"  vs. {total_passed}/{total_checkpoints} individual checkpoints passed ({total_passed / total_checkpoints:.0%})")
    print(f"  (the gap between these two numbers is itself the finding, same as PhysicianBench's headline result)")
    print()
    emit("episode_level_pass_rate", "episodes_fully_passed", count=fully_passed, total=len(all_episode_keys), rate=round(fully_passed / len(all_episode_keys), 4))
    emit("episode_level_pass_rate", "individual_checkpoints_passed", count=total_passed, total=total_checkpoints, rate=round(total_passed / total_checkpoints, 4))

    # --- Per-episode Process x Accuracy rate: episode's Process pass rate
    # (Z1+Z2) multiplied by its Accuracy pass rate (A1+A2+A3) - a joint
    # measure, not just an AND/OR - an episode that's process-perfect but
    # clinically wrong isn't "good," and this number reflects that instead
    # of hiding it behind a binary pass/fail. ---
    per_ep_process_total: Counter[str] = Counter()
    per_ep_process_passed: Counter[str] = Counter()
    per_ep_accuracy_total: Counter[str] = Counter()
    per_ep_accuracy_passed: Counter[str] = Counter()
    for row in rows:
        if not is_scoreable(row):
            continue
        ep_key = row["episode_dir"]
        passed = float(row["points"]) >= 0
        if row["axis"] == "Process":
            per_ep_process_total[ep_key] += 1
            per_ep_process_passed[ep_key] += int(passed)
        else:
            per_ep_accuracy_total[ep_key] += 1
            per_ep_accuracy_passed[ep_key] += int(passed)

    print("--- Process x Accuracy rate, per episode ---")
    combined_rates = []
    for ep_key in sorted(all_episode_keys):
        p_total, p_passed = per_ep_process_total.get(ep_key, 0), per_ep_process_passed.get(ep_key, 0)
        a_total, a_passed = per_ep_accuracy_total.get(ep_key, 0), per_ep_accuracy_passed.get(ep_key, 0)
        p_rate = p_passed / p_total if p_total else 0.0
        a_rate = a_passed / a_total if a_total else 0.0
        combined = p_rate * a_rate
        combined_rates.append(combined)
        print(f"  {ep_key}: Process {p_passed}/{p_total} ({p_rate:.0%}) x Accuracy {a_passed}/{a_total} ({a_rate:.0%}) "
              f"= Process x Accuracy rate {combined:.0%}")
        emit("process_x_accuracy_per_episode", ep_key, rate=round(combined, 4),
             note=f"process={p_passed}/{p_total} accuracy={a_passed}/{a_total}")
    if combined_rates:
        avg_combined = sum(combined_rates) / len(combined_rates)
        print(f"\n  Average Process x Accuracy rate across all episodes: {avg_combined:.0%}")
        emit("process_x_accuracy_per_episode", "AVERAGE", rate=round(avg_combined, 4))
    print()

    # --- Per-condition/phenotype breakdown: same pass rate as "by
    # checkpoint kind" above, but sliced by what the visit's real clinical
    # condition was - shows WHICH kinds of cases the agent struggles with,
    # not just an overall average across everything mixed together. ---
    episode_condition: dict[str, str] = {}
    for pid, vdate, ep_dir in episodes:
        episode_condition[ep_dir] = _condition_for(pid, vdate)

    cond_total: Counter[str] = Counter()
    cond_passed: Counter[str] = Counter()
    for row in rows:
        if not is_scoreable(row):
            continue
        cond = episode_condition.get(row["episode_dir"], "(unknown)")
        cond_total[cond] += 1
        if float(row["points"]) >= 0:
            cond_passed[cond] += 1

    print("--- Pass rate by condition/phenotype ---")
    for cond in sorted(cond_total):
        total, passed = cond_total[cond], cond_passed[cond]
        print(f"  {cond}: {passed}/{total} ({passed / total:.0%})")
        emit("pass_rate_by_condition", cond, count=passed, total=total, rate=round(passed / total, 4))
    print()

    print("--- Clinical-only failure breakdown (execution-failed episodes excluded) ---")
    clinical_tag_counts: Counter[str] = Counter()
    for row in rows:
        if row["episode_dir"] not in clinical_episodes:
            continue
        tag = classify(row)
        if tag and tag not in EXECUTION_TAGS:
            clinical_tag_counts[tag] += 1
    if clinical_tag_counts:
        for tag, count in clinical_tag_counts.most_common():
            print(f"  {count:3d}  {tag}")
            emit("clinical_only_failure_breakdown", tag, count=count)
    else:
        print("  (no clinical-track episodes to report - all episodes had execution failures)")

    with AGGREGATE_REPORT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nFull report written to {AGGREGATE_REPORT_CSV}")


if __name__ == "__main__":
    import sys
    # Optional: pass episode_dir names to restrict analysis to a specific
    # batch (e.g. this session's 10-visit stratified run), excluding
    # earlier ad-hoc test episodes that share the same accumulating
    # all_receipts.csv. No args = analyze everything, as before.
    episode_filter = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    build_report(episode_filter)
