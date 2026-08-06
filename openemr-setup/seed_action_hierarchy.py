"""
seed_action_hierarchy.py

Registers the full 3-level action/tool-calling hierarchy (see the plan doc
"Three-level action/tool-calling hierarchy") into OpenEMR's real order-type
catalog (procedure_type table, the same table "Configure Orders and
Results" - Procedures -> Configuration - reads/writes). NOT "Load
Compendium", which is specifically a lab-vendor test-code importer, not a
fit for this.

Structure: Level 1 (16 tool signatures) as top-level parent categories,
Level 2 (55 actions) nested under their Level-1 tool, Level 3 (30 specific
options across 8 actions that have them) nested under their Level-2 action.
All names are human-readable labels (no ACT_ prefix, no underscores, no
snake_case) - e.g. "ACT_REASSESS_BEDSIDE" -> "Reassess Bedside".

This REPLACES the flat 34-action registration from seed_action_library.py
(Track A) - that one is left in place but superseded; the real, current
structure lives under these new Level-1 categories.

Idempotent: matched by (parent, name) - re-running updates rather than
duplicates.
"""

import subprocess
from pathlib import Path

import openpyxl

COMPOSE_DIR = Path("/Users/lakshitavij/synthetic pipeline/openemr/docker/development-easy")
ACTION_LIBRARY_XLSX = Path(
    "/Users/lakshitavij/ChartR-ClinicalRL/pulmonary_action_library_with_2026_guideline_rewards.xlsx"
)

# New actions not in the xlsx (added during the Level-3 review) - name only,
# matching the format load_action_library() below expects.
NEW_ACTIONS = {
    "ACT_REPOSITION_LINE": "Reposition Or Retract Central Line/Catheter",
    "ACT_MANAGE_PLEURAL_CATHETER": "Manage Indwelling Pleural Catheter",
    "ACT_OPTIMIZE_BP": "Optimize Blood Pressure/Hemodynamics",
}


def humanize(action_id: str) -> str:
    """ACT_REASSESS_BEDSIDE -> Reassess Bedside"""
    words = action_id.replace("ACT_", "").split("_")
    return " ".join(w.capitalize() for w in words)


def humanize_opt(opt: str) -> str:
    """with_contrast -> With Contrast; PE_workup -> PE Workup.
    Leaves tokens that already have internal mixed case alone (e.g.
    "Swan-Ganz" must stay as-is - .capitalize() would mangle it to
    "Swan-ganz"). Opts with no underscore are already a finished label
    (e.g. the real-interval options below) - passed through unchanged."""
    if "_" not in opt:
        return opt
    def fix(w: str) -> str:
        if w.isupper() or not w.islower():
            return w
        return w.capitalize()
    return " ".join(fix(w) for w in opt.split("_"))


# Real intervals extracted from the 30 rows in generated_followups_fin.csv
# that actually use ACT_TIMED_FOLLOWUP / ACT_REASSESS_DEFINED_INTERVAL -
# every distinct numeric interval mentioned in the real recommendation text.
# No "days" option: zero rows in the real data mention a days-scale interval,
# so one isn't invented (matches the project's no-fabrication rule).
REAL_INTERVAL_OPTIONS = [
    "4-6 Hours (ED Reassessment)",
    "6-24 Hours (Short-Interval Recheck)",
    "6-8 Weeks (Post-Treatment)",
    "2 Months (Post-Op Surveillance)",
]


# --- The full tree: Level 1 -> [(Level 2 action_id, [Level 3 options])] ---
# Matches the plan file's confirmed flow exactly.
TREE: dict[str, list[tuple[str, list[str]]]] = {
    "Get Current Exam": [],
    "Get Prior Exam": [("ACT_COMPARE_PRIORS", [])],
    "Get Clinical Context": [
        ("ACT_REQUEST_OXYGEN_STATUS", []),
        ("ACT_REQUEST_RESPIRATORY_HISTORY", []),
    ],
    "Get Relevant Labs": [
        ("ACT_REQUEST_RELEVANT_LABS", ["renal_electrolyte", "heart_failure", "infectious", "PE_workup", "COPD_ABG"]),
    ],
    "Get Current Medications": [],
    "Get Allergies": [],
    "Request Bedside Reassessment": [
        ("ACT_REASSESS_BEDSIDE", ["general_reassessment", "ventilator_and_airway_review"]),
    ],
    "Order Repeat CXR": [("ACT_REPEAT_CXR", [])],
    "Order CT Chest": [("ACT_CT_CHEST", ["with_contrast", "without_contrast"])],
    "Order CTPA": [("ACT_CTPA", [])],
    "Order Thoracic Ultrasound": [
        ("ACT_THORACIC_US", []),
        ("ACT_LUNG_US_FOR_CAP", []),
    ],
    "Request Specialist Review": [
        ("ACT_PULMONOLOGY_FOLLOWUP", ["routine_outpatient", "interventional_pulmonology"]),
        ("ACT_THORACIC_SURGERY_FOLLOWUP", []),
        ("ACT_ONCOLOGY_FOLLOWUP", []),
        ("ACT_CARDIOLOGY_FOLLOWUP", ["general_cardiology", "ep_device_clinic"]),
    ],
    "Schedule Followup": [
        ("ACT_OUTPATIENT_SURVEILLANCE", []),
        ("ACT_TIMED_FOLLOWUP", REAL_INTERVAL_OPTIONS),
        ("ACT_REASSESS_DEFINED_INTERVAL", REAL_INTERVAL_OPTIONS),
        ("ACT_STOP_SURVEILLANCE", []),
    ],
    "Notify Urgent Clinician": [("ACT_URGENT_ASSESSMENT", [])],
    "Activate Rapid Response": [("ACT_RAPID_RESPONSE", [])],
    "Escalate Level Of Care": [("ACT_ESCALATE_CARE", [])],
    "Recommend Action": [
        ("ACT_ABSTAIN", []),
        ("ACT_CLINICAL_CORRELATION", []),
        ("ACT_OBSERVE", []),
        ("ACT_INCENTIVE_SPIROMETRY", []),
        ("ACT_DEEP_BREATHING_COUGH", []),
        ("ACT_EARLY_MOBILIZATION", []),
        ("ACT_OPTIMIZE_ANALGESIA", []),
        ("ACT_AIRWAY_CLEARANCE", []),
        ("ACT_ALBUTEROL", []),
        ("ACT_IPRATROPIUM", []),
        ("ACT_SYSTEMIC_STEROID_COPD", []),
        ("ACT_ANTIBIOTIC_COPD", []),
        ("ACT_OXYGEN_TITRATION", []),
        ("ACT_HFNC", []),
        ("ACT_NIV", []),
        ("ACT_INTUBATION_PATHWAY", []),
        ("ACT_FUROSEMIDE_IV", []),
        ("ACT_ESCALATE_DIURETIC", []),
        ("ACT_NITROGLYCERIN", []),
        ("ACT_CAP_OUTPATIENT_ABX", []),
        ("ACT_CAP_INPATIENT_ABX", []),
        ("ACT_HOLD_ANTIBIOTICS", []),
        ("ACT_SYSTEMIC_STEROID_SEVERE_CAP", []),
        ("ACT_DIAGNOSTIC_THORACENTESIS", []),
        ("ACT_THERAPEUTIC_THORACENTESIS", []),
        ("ACT_CHEST_TUBE_PLEURAL_INFECTION", []),
        ("ACT_CHEST_TUBE_PTX", ["insert", "water_seal_trial", "remove"]),
        ("ACT_EMERGENT_DECOMPRESSION", []),
        ("ACT_BRONCHOSCOPY", []),
        ("ACT_START_ANTICOAGULATION_PE", []),
        ("ACT_ESCALATE_PE_REPERFUSION", []),
        ("ACT_REPOSITION_LINE", ["PICC", "Swan-Ganz", "central_line_other"]),
        ("ACT_MANAGE_PLEURAL_CATHETER", ["restore_drainage", "reposition", "replace"]),
        ("ACT_OPTIMIZE_BP", []),
    ],
}


def load_action_names() -> dict[str, str]:
    wb = openpyxl.load_workbook(ACTION_LIBRARY_XLSX, data_only=True)
    ws = wb["Action Library"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = rows[0]
    names = {}
    for r in rows[1:]:
        d = dict(zip(header, r))
        if d.get("action_id"):
            names[d["action_id"]] = d["action_name"]
    names.update(NEW_ACTIONS)
    return names


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


CHARTR_LAB_ID = 1  # procedure_providers.ppid for the dummy "ChartR Clinical Actions"
# lab, registered once via `INSERT INTO procedure_providers (name, ...)
# VALUES ('ChartR Clinical Actions', ...)`. Without a real lab_id, our
# catalog rows never appear as selectable in OpenEMR's real per-patient
# Procedure Order form (interface/forms/procedure_order), which filters
# procedure_type by lab_id - confirmed via source, not an assumption.


def find_or_create(parent: int, name: str, procedure_type: str, code: str = "") -> int:
    existing = run_sql(
        f"SELECT procedure_type_id FROM procedure_type "
        f"WHERE parent={parent} AND name='{sql_escape(name)}';"
    )
    lines = [l for l in existing.strip().splitlines()[1:] if l.strip()]
    if lines:
        return int(lines[0])
    run_sql(
        "INSERT INTO procedure_type (parent, name, procedure_type, procedure_code, lab_id, activity) "
        f"VALUES ({parent}, '{sql_escape(name)}', '{procedure_type}', '{sql_escape(code)}', {CHARTR_LAB_ID}, 1);"
    )
    lookup = run_sql(
        f"SELECT procedure_type_id FROM procedure_type "
        f"WHERE parent={parent} AND name='{sql_escape(name)}';"
    )
    return int([l for l in lookup.strip().splitlines()[1:] if l.strip()][0])


def main() -> None:
    action_names = load_action_names()
    l1_count, l2_count, l3_count = 0, 0, 0

    for level1_label, actions in TREE.items():
        l1_id = find_or_create(0, level1_label, "grp")
        l1_count += 1
        print(f"L1: {level1_label} (id={l1_id})")

        for action_id, level3_opts in actions:
            l2_label = action_names.get(action_id, humanize(action_id))
            l2_id = find_or_create(l1_id, l2_label, "ord", action_id)
            l2_count += 1
            print(f"  L2: {l2_label} (id={l2_id})")

            for opt in level3_opts:
                l3_label = humanize_opt(opt)
                find_or_create(l2_id, l3_label, "ord", opt)
                l3_count += 1
                print(f"    L3: {l3_label}")

    print(f"\nDone. {l1_count} Level-1, {l2_count} Level-2, {l3_count} Level-3 entries.")


if __name__ == "__main__":
    main()
