"""
seed_action_library.py

Track A: registers the action_ids that ACTUALLY appear in
generated_followups_fin.csv's real output (34 of the 52 in the full Action
Library) as real OpenEMR order types, via direct SQL insert into
procedure_type - the same table the "Configure Orders and Results" UI
(Procedures -> Configuration) reads/writes.

Idempotent: re-running updates existing rows (matched by procedure_code)
rather than duplicating them.
"""

import csv
import subprocess
from pathlib import Path

import openpyxl

FOLLOWUPS_CSV = Path("/Users/lakshitavij/synthetic pipeline/generated_followups_fin.csv")
ACTION_LIBRARY_XLSX = Path(
    "/Users/lakshitavij/ChartR-ClinicalRL/pulmonary_action_library_with_2026_guideline_rewards.xlsx"
)
COMPOSE_DIR = Path("/Users/lakshitavij/synthetic pipeline/openemr/docker/development-easy")
CATEGORY_NAME = "Pulmonology/Radiology Follow-Up"


def get_used_action_ids() -> set[str]:
    used = set()
    with FOLLOWUPS_CSV.open() as f:
        for row in csv.DictReader(f):
            for aid in (row.get("generated_action_ids") or "").split(";"):
                aid = aid.strip()
                if aid:
                    used.add(aid)
    return used


def load_action_library() -> dict[str, dict]:
    wb = openpyxl.load_workbook(ACTION_LIBRARY_XLSX, data_only=True)
    ws = wb["Action Library"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = rows[0]
    catalog = {}
    for r in rows[1:]:
        d = dict(zip(header, r))
        if d.get("action_id"):
            catalog[d["action_id"]] = d
    return catalog


def sql_escape(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def run_sql(sql: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "mysql", "mariadb", "-uroot", "-proot", "openemr", "-e", sql],
        cwd=COMPOSE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    used_ids = get_used_action_ids()
    catalog = load_action_library()

    missing = used_ids - set(catalog.keys())
    if missing:
        print(f"WARNING: {len(missing)} used action_ids not found in the Action Library xlsx: {missing}")
    used_ids &= set(catalog.keys())
    print(f"Registering {len(used_ids)} actually-used action_ids under category {CATEGORY_NAME!r}")

    # Category (top-level parent, parent=0). Idempotent: reuse if it already exists.
    existing = run_sql(
        f"SELECT procedure_type_id FROM procedure_type WHERE name='{sql_escape(CATEGORY_NAME)}' AND parent=0;"
    )
    lines = [l for l in existing.strip().splitlines()[1:] if l.strip()]
    if lines:
        category_id = int(lines[0])
        print(f"Reusing existing category, procedure_type_id={category_id}")
    else:
        # NOTE: each run_sql() call is a separate `docker compose exec` -> a fresh
        # DB connection, so LAST_INSERT_ID() in a follow-up call would always see 0
        # (it's connection-scoped). Do the insert and the id lookup in one call.
        run_sql(
            "INSERT INTO procedure_type "
            "(parent, name, procedure_type, procedure_code, description, activity) "
            f"VALUES (0, '{sql_escape(CATEGORY_NAME)}', 'grp', '', "
            f"'Top-level category for real chest X-ray follow-up actions actually used "
            f"across the 152-case gold dataset.', 1); SELECT LAST_INSERT_ID();"
        )
        lookup = run_sql(
            f"SELECT procedure_type_id FROM procedure_type WHERE name='{sql_escape(CATEGORY_NAME)}' AND parent=0;"
        )
        category_id = int([l for l in lookup.strip().splitlines()[1:] if l.strip()][0])
        print(f"Created category, procedure_type_id={category_id}")

    for action_id in sorted(used_ids):
        info = catalog[action_id]
        name = sql_escape(info.get("action_name") or action_id)
        description = sql_escape((info.get("appropriate_use") or "")[:255])
        code = sql_escape(action_id)

        existing_row = run_sql(
            f"SELECT procedure_type_id FROM procedure_type WHERE procedure_code='{code}' AND parent={category_id};"
        )
        row_lines = [l for l in existing_row.strip().splitlines()[1:] if l.strip()]
        if row_lines:
            run_sql(
                f"UPDATE procedure_type SET name='{name}', description='{description}' "
                f"WHERE procedure_type_id={int(row_lines[0])};"
            )
            print(f"  updated  {action_id}")
        else:
            run_sql(
                "INSERT INTO procedure_type "
                "(parent, name, procedure_type, procedure_code, description, activity) "
                f"VALUES ({category_id}, '{name}', 'ord', '{code}', '{description}', 1);"
            )
            print(f"  inserted {action_id}")

    print("Done.")


if __name__ == "__main__":
    main()
