"""
seed_xray_viewer_forms.py

Pre-attaches the "X-ray Viewer" form (interface/forms/xray_viewer/) to
every OpenEMR encounter that has a matching real DICOM study in Orthanc -
same forms/form_xray_viewer row creation new.php does lazily on first
open, just done once up front for every real patient/encounter so the
form already shows up in each encounter's form list instead of requiring
a manual "Add Form" click 152 times.

Matches encounters to studies the same way the viewer itself does at
runtime (confirmed against real seeded data, not guessed):
  - patient_data.lname            == DICOM PatientID
  - form_encounter.date           == DICOM StudyDateTime

Only attaches the form where a real matching Orthanc study actually
exists - an encounter with no imaging gets no X-ray Viewer form, same as
new.php would find no data if opened for it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import requests

ORTHANC_URL = "http://localhost:8042"
MYSQL_CMD = ["docker", "compose", "exec", "-T", "mysql", "mariadb", "-uroot", "-proot", "openemr"]
# Relative to this file's own location (openemr-setup/../openemr/docker/...)
# instead of hardcoded to one machine's checkout path - this script needs to
# run correctly from any clone of the repo, not just the one it was written on.
COMPOSE_DIR = str(Path(__file__).resolve().parent.parent / "openemr" / "docker" / "development-easy")


def sql(query: str, params: list | None = None) -> str:
    if params:
        for p in params:
            query = query.replace("?", f"'{p}'", 1)
    result = subprocess.run(
        MYSQL_CMD + ["-e", query],
        cwd=COMPOSE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def has_matching_study(patient_id: str, study_date_yyyymmdd: str) -> bool:
    resp = requests.post(
        f"{ORTHANC_URL}/tools/find",
        json={
            "Level": "Study",
            "Query": {"PatientID": patient_id, "StudyDate": study_date_yyyymmdd},
        },
        timeout=10,
    )
    resp.raise_for_status()
    return len(resp.json()) > 0


def main() -> None:
    encounters_raw = sql(
        "SELECT fe.pid, fe.encounter, pd.lname, DATE_FORMAT(fe.date, '%Y%m%d') "
        "FROM form_encounter fe JOIN patient_data pd ON pd.pid = fe.pid ORDER BY fe.pid, fe.date;"
    )
    rows = [line.split("\t") for line in encounters_raw.strip().splitlines()[1:]]
    print(f"Found {len(rows)} encounters to check.")

    attached, skipped_no_imaging, skipped_already_attached = 0, 0, 0

    for pid, encounter, patient_id, study_date in rows:
        existing = sql(
            "SELECT form_id FROM forms WHERE formdir='xray_viewer' AND pid=? AND encounter=? AND deleted=0 LIMIT 1;",
            [pid, encounter],
        )
        if existing.strip().splitlines()[1:]:
            skipped_already_attached += 1
            continue

        if not has_matching_study(patient_id, study_date):
            skipped_no_imaging += 1
            continue

        # Mirrors new.php's own create-on-first-open logic exactly (same
        # two inserts, same column values) - just done here for all
        # matching encounters up front instead of one at a time on demand.
        #
        # INSERT and SELECT LAST_INSERT_ID() must run in the SAME `mariadb
        # -e` invocation, not two separate sql() calls - LAST_INSERT_ID() is
        # scoped to a single database connection, and each `docker compose
        # exec ... mariadb -e` call opens a brand new one. Two separate
        # calls meant the id lookup never saw the real insert at all,
        # silently returning 0 instead - confirmed the hard way: it
        # produced 151 broken forms rows (form_id=0, pointing at nothing)
        # across the whole patient set on the first run of this script.
        insert_and_id = sql(
            "INSERT INTO form_xray_viewer (date, pid, encounter, user, groupname, authorized, activity) "
            "VALUES (NOW(), ?, ?, 'admin', 'Default', 1, 1); SELECT LAST_INSERT_ID();",
            [pid, encounter],
        )
        new_id = insert_and_id.strip().splitlines()[1]
        assert new_id and new_id != "0", f"LAST_INSERT_ID() returned {new_id!r} for pid={pid} encounter={encounter}"
        sql(
            "INSERT INTO forms (date, encounter, form_name, form_id, pid, user, groupname, authorized, formdir) "
            "VALUES (NOW(), ?, 'X-ray Viewer', ?, ?, 'admin', 'Default', 1, 'xray_viewer');",
            [encounter, new_id, pid],
        )
        attached += 1
        print(f"  pid={pid} encounter={encounter} ({patient_id}/{study_date}): X-ray Viewer attached")

    print(
        f"\nDone: {attached} attached, {skipped_already_attached} already attached, "
        f"{skipped_no_imaging} skipped (no matching Orthanc study)."
    )


if __name__ == "__main__":
    main()
