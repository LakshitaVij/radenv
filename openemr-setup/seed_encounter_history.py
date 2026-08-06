"""
seed_encounter_history.py

Writes each visit's real HistoryText into that SPECIFIC encounter's own
`reason` field (form_encounter.reason) - genuinely per-visit, not the
patient-level "Medical Problems" list (lists.type='medical_problem'), which
was tried first and is wrong for this: Medical Problems is a longitudinal
problem list (begdate/enddate range) that shows on every encounter until
resolved, not a per-visit field. That's why "Sudden loss of vision" showed
up on all 10 of a patient's visits instead of just the one it came from.

form_encounter.reason is the correct home: it's a plain per-encounter text
field, already exists, and is what actually displays only when viewing that
specific encounter.

Idempotent: only writes when reason is currently empty or still the seed
placeholder ("X-ray gold-set seed visit"), so re-running doesn't clobber
manual edits made in the OpenEMR UI.
"""

import csv
import re
import subprocess
from pathlib import Path

REPORTS_CSV = Path("/Users/lakshitavij/synthetic pipeline/csv/consolidated_real_reports.csv")
COMPOSE_DIR = Path("/Users/lakshitavij/synthetic pipeline/openemr/docker/development-easy")
PLACEHOLDER = "X-ray gold-set seed visit"

_COMPARISON_LINE_RE = re.compile(r"(?im)^\s*comparison\s*:.*$")


def clean_history(raw_history: str) -> str:
    """The HistoryText column content as-is, minus any 'Comparison: ...'
    line (that's a report metadata note, not part of the clinical history).
    Returns "" if nothing remains, so no reason is written for that visit."""
    text = _COMPARISON_LINE_RE.sub("", raw_history)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines).strip()


def run_sql(sql: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "mysql", "mariadb", "-uroot", "-proot", "openemr", "-e", sql],
        cwd=COMPOSE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def sql_escape(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def format_datetime(study_date: str, study_time: str) -> str | None:
    if not study_date or len(study_date) != 8:
        return None
    date_part = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
    hhmmss = "00:00:00"
    if study_time:
        digits = study_time.split(".")[0].zfill(6)
        hhmmss = f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"
    return f"{date_part} {hhmmss}"


def main() -> None:
    # patient_id (lname) -> pid, since form_encounter keys on the internal pid.
    pid_lookup_raw = run_sql("SELECT pid, lname FROM patient_data;")
    pid_by_patient_id = {}
    for line in pid_lookup_raw.strip().splitlines()[1:]:
        pid, lname = line.split("\t")
        pid_by_patient_id[lname] = int(pid)

    with REPORTS_CSV.open() as f:
        rows = list(csv.DictReader(f))

    written, skipped_no_history, skipped_no_patient, skipped_no_encounter = 0, 0, 0, 0

    for row in rows:
        history = clean_history((row.get("HistoryText") or "").strip())
        if not history:
            skipped_no_history += 1
            continue

        pid = pid_by_patient_id.get(row["PatientID"])
        if pid is None:
            skipped_no_patient += 1
            continue

        dt = format_datetime(row["StudyDate"], row.get("StudyTime", ""))
        if not dt:
            skipped_no_encounter += 1
            continue

        existing = run_sql(
            f"SELECT reason FROM form_encounter WHERE pid={pid} AND date='{dt}';"
        )
        lines = [l for l in existing.strip().splitlines()[1:] if l.strip() or l == ""]
        if not lines:
            skipped_no_encounter += 1
            continue

        # Correction pass: every existing reason came from this same script's
        # prior (buggy) run, not a manual OpenEMR edit, so it's safe to
        # overwrite unconditionally here rather than only when empty/placeholder.
        run_sql(
            f"UPDATE form_encounter SET reason='{sql_escape(history[:255])}' "
            f"WHERE pid={pid} AND date='{dt}';"
        )
        written += 1
        print(f"  {row['PatientID']} {dt}: reason set")

    print(f"\nDone. {written} encounters updated.")
    print(f"Skipped: {skipped_no_history} no history text, "
          f"{skipped_no_patient} patient not found, "
          f"{skipped_no_encounter} encounter not found.")


if __name__ == "__main__":
    main()
