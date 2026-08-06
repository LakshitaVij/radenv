"""
seed_patients_and_vitals.py

Track B: seeds all 10 real X-ray patients into OpenEMR as FHIR Patients (same
mechanism already proven manually for the first test patient), one Encounter
per real visit (152 total - vitals in this OpenEMR version are strictly
encounter-scoped, confirmed the hard way: the docs' "POST .../vital" example
without an encounter 404s; the real route is
"POST /api/patient/:pid/encounter/:eid/vital"), then one Vitals entry per
encounter, using the clinical context merged into consolidated_real_reports.csv
earlier.

Also note: Standard API scopes use a different naming convention than FHIR
scopes (lowercase, e.g. "user/encounter.crus" vs "user/Encounter.cruds") -
a client registered with only FHIR-style scopes gets a silent 401 on Standard
API routes even with a valid token. Both scope styles are requested here.

Idempotent: patients are looked up by family name before creating (FHIR
identifier search doesn't match custom identifier systems in this OpenEMR
version - confirmed directly); encounters are looked up by (patient, date)
before creating; vitals are looked up by encounter before creating.

Field mapping (see plan file for the "why"):
  bps                <- systolic_bp_mmHg
  temperature        <- temperature_C, converted to Fahrenheit (confirmed the
                         hard way: OpenEMR's temperature field is Fahrenheit-
                         native - posting a raw Celsius number got labeled and
                         displayed as if it were already Fahrenheit)
  respiration        <- respiratory_rate_per_min
  oxygen_saturation  <- oxygen_saturation_percent
  note               <- mental_status / work_of_breathing / oxygen_device /
                         current_care_setting (no native form_vitals fields
                         exist for these)
  pulse              <- not set; not present in our generated clinical context

Output: openemr/patient_uuid_map.csv (PatientID -> OpenEMR uuid), needed for
any later write (orders, notes) since OpenEMR's API keys off its own uuid,
not our PatientID.
"""

import csv
from collections import defaultdict
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://localhost:9300"
CLIENT_ID = "Ay4VOL6aq1jM8UATLvWZbSCA4wiJeXDJr0Pzy3Cdz0I"
CLIENT_SECRET = "kNti0zni8fmpGGHhlq_7vtoQCkOeBmzfY0zL_H8wn9w_tc0hvFYtu2I9ESdjSjiPjT1J4KGVvh76WPzYY4XJlQ"
SCOPE = (
    "openid offline_access api:oemr api:fhir "
    "user/Patient.cruds user/Encounter.cruds user/DocumentReference.cruds user/ServiceRequest.cruds "
    "user/patient.crus user/encounter.crus user/vital.crus"
)
OFFICE_VISIT_CATID = 5  # openemr_postcalendar_categories.pc_catid for "Office Visit"

REPORTS_CSV = Path("/Users/lakshitavij/synthetic pipeline/csv/consolidated_real_reports.csv")
OUT_MAP_CSV = Path("/Users/lakshitavij/synthetic pipeline/openemr/patient_uuid_map.csv")

IDENTIFIER_SYSTEM = "urn:chartr:xray-gold-set"


def get_token() -> str:
    resp = requests.post(
        f"{BASE}/oauth2/default/token",
        data={
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": SCOPE,
            "user_role": "users",
            "username": "admin",
            "password": "pass",
        },
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def find_patient_uuid(token: str, patient_id: str) -> str | None:
    # NOTE: FHIR identifier search doesn't match custom identifier systems in
    # this OpenEMR version (confirmed directly - returns total:0 even for a
    # patient created with that exact identifier). family-name search does
    # work reliably, and since we control the family name format (= patient_id
    # exactly), it's a solid substitute for exact lookup here.
    resp = requests.get(
        f"{BASE}/apis/default/fhir/Patient",
        params={"family": patient_id},
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
    )
    resp.raise_for_status()
    bundle = resp.json()
    if bundle.get("total", 0) > 0:
        return bundle["entry"][0]["resource"]["id"]
    return None


def parse_age_years(age_str: str) -> int | None:
    # e.g. "046Y" -> 46
    if not age_str or not age_str.endswith("Y"):
        return None
    try:
        return int(age_str[:-1])
    except ValueError:
        return None


def create_patient(token: str, patient_id: str, age_years: int | None, sex: str, first_visit_year: int) -> str:
    birth_year = first_visit_year - age_years if age_years is not None else first_visit_year - 40
    gender = {"M": "male", "F": "female"}.get(sex, "unknown")
    body = {
        "resourceType": "Patient",
        "identifier": [{"system": IDENTIFIER_SYSTEM, "value": patient_id}],
        # No given name - Patient Finder should just show the real PatientID
        # (family), not a fake first name.
        "name": [{"use": "official", "family": patient_id, "given": [""]}],
        "gender": gender,
        "birthDate": f"{birth_year}-01-01",
    }
    resp = requests.post(
        f"{BASE}/apis/default/fhir/Patient",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/fhir+json"},
        verify=False,
    )
    resp.raise_for_status()
    # The FHIR create response here returns {"pid": ..., "uuid": ...} per what
    # we saw manually - re-fetch by identifier to get the canonical FHIR id
    # consistently regardless of response shape.
    uuid = find_patient_uuid(token, patient_id)
    if not uuid:
        raise RuntimeError(f"Created patient {patient_id} but could not find it back by identifier")
    return uuid


def format_datetime(study_date: str, study_time: str) -> str | None:
    # "20080330" + "011543.934395" -> "2008-03-30 01:15:43"
    # StudyTime is included (not just date) because a patient can have two
    # real, distinct visits on the same calendar day - confirmed in this
    # dataset (10 same-day collisions found the hard way: date-only keying
    # collapsed them into one encounter with two conflicting vitals rows).
    # oracle.py already keys gold lookups on (StudyDate, StudyTime) for the
    # same reason.
    if not study_date or len(study_date) != 8:
        return None
    date_part = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
    hhmmss = "00:00:00"
    if study_time:
        digits = study_time.split(".")[0].zfill(6)
        hhmmss = f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"
    return f"{date_part} {hhmmss}"


def find_encounter(token: str, uuid: str, date: str) -> int | None:
    resp = requests.get(
        f"{BASE}/apis/default/api/patient/{uuid}/encounter",
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
    )
    if resp.status_code != 200:
        return None
    for enc in resp.json().get("data", []):
        if (enc.get("date") or "") == date:
            return enc["eid"]
    return None


def create_encounter(token: str, uuid: str, date: str) -> int:
    resp = requests.post(
        f"{BASE}/apis/default/api/patient/{uuid}/encounter",
        json={
            "date": date,
            "reason": "X-ray gold-set seed visit",
            "class_code": "AMB",
            "pc_catid": OFFICE_VISIT_CATID,
            "facility_id": "3",
        },
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        verify=False,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("validationErrors"):
        raise RuntimeError(f"Encounter validation failed for {uuid}/{date}: {body['validationErrors']}")
    return body["data"]["eid"]


def vital_exists(token: str, uuid: str, eid: int) -> bool:
    resp = requests.get(
        f"{BASE}/apis/default/api/patient/{uuid}/encounter/{eid}/vital",
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
    )
    if resp.status_code != 200:
        return False
    return len(resp.json().get("data", [])) > 0


def post_vital(token: str, uuid: str, eid: int, row: dict, date: str) -> None:
    note_parts = []
    for label, field in [
        ("Mental status", "mental_status"),
        ("Work of breathing", "work_of_breathing"),
        ("Oxygen device", "oxygen_device"),
        ("Care setting", "current_care_setting"),
    ]:
        val = row.get(field, "").strip()
        if val:
            note_parts.append(f"{label}: {val}")

    body = {"date": date}
    if row.get("systolic_bp_mmHg"):
        body["bps"] = row["systolic_bp_mmHg"]
    if row.get("temperature_C"):
        fahrenheit = float(row["temperature_C"]) * 9 / 5 + 32
        body["temperature"] = round(fahrenheit, 1)
    if row.get("respiratory_rate_per_min"):
        body["respiration"] = row["respiratory_rate_per_min"]
    if row.get("oxygen_saturation_percent"):
        body["oxygen_saturation"] = row["oxygen_saturation_percent"]
    if note_parts:
        body["note"] = "; ".join(note_parts)[:255]

    resp = requests.post(
        f"{BASE}/apis/default/api/patient/{uuid}/encounter/{eid}/vital",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        verify=False,
    )
    resp.raise_for_status()


COMPOSE_DIR = Path("/Users/lakshitavij/synthetic pipeline/openemr/docker/development-easy")


def backfill_vitals_pid() -> None:
    # Confirmed the hard way: OpenEMR's POST .../vital route leaves forms.pid
    # and form_vitals.pid at 0 regardless of what's in the request body (even
    # passing "pid" explicitly in the body doesn't help) - the real, correct
    # patient link only exists via forms.encounter -> form_encounter.pid.
    # Without this backfill, the patient dashboard's Vitals widget shows
    # "No vitals have been documented" even though the data is really there.
    import subprocess

    subprocess.run(
        [
            "docker", "compose", "exec", "-T", "mysql", "mariadb", "-uroot", "-proot", "openemr", "-e",
            "UPDATE forms f JOIN form_encounter fe ON fe.encounter = f.encounter "
            "SET f.pid = fe.pid WHERE f.form_name = 'Vitals' AND f.pid = 0; "
            "UPDATE form_vitals fv JOIN forms f ON f.form_id = fv.id AND f.form_name = 'Vitals' "
            "SET fv.pid = f.pid WHERE fv.pid = 0;",
        ],
        cwd=COMPOSE_DIR,
        check=True,
    )


def main() -> None:
    with REPORTS_CSV.open() as f:
        rows = list(csv.DictReader(f))

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_patient[row["PatientID"]].append(row)

    token = get_token()
    uuid_map = {}
    vitals_written = 0
    vitals_skipped_existing = 0
    vitals_skipped_no_date = 0

    for patient_id, visits in by_patient.items():
        visits.sort(key=lambda r: r["StudyDate"])
        first = visits[0]

        existing_uuid = find_patient_uuid(token, patient_id)
        if existing_uuid:
            print(f"{patient_id}: reusing existing patient uuid={existing_uuid}")
            uuid = existing_uuid
        else:
            age = parse_age_years(first["PatientAge"])
            first_year = int(first["StudyDate"][:4])
            uuid = create_patient(token, patient_id, age, first["PatientSex"], first_year)
            print(f"{patient_id}: created patient uuid={uuid}")
        uuid_map[patient_id] = uuid

        for row in visits:
            date = format_datetime(row["StudyDate"], row.get("StudyTime", ""))
            if not date:
                vitals_skipped_no_date += 1
                continue

            eid = find_encounter(token, uuid, date)
            if eid is None:
                eid = create_encounter(token, uuid, date)

            if vital_exists(token, uuid, eid):
                vitals_skipped_existing += 1
                continue
            post_vital(token, uuid, eid, row, date)
            vitals_written += 1
            print(f"  {patient_id} {date}: vital written (eid={eid})")

    with OUT_MAP_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PatientID", "openemr_uuid"])
        for pid, uuid in uuid_map.items():
            writer.writerow([pid, uuid])

    backfill_vitals_pid()

    print(f"\nDone. {len(uuid_map)} patients. "
          f"Vitals: {vitals_written} written, {vitals_skipped_existing} already existed, "
          f"{vitals_skipped_no_date} skipped (bad date).")
    print(f"Patient UUID map written to {OUT_MAP_CSV}")


if __name__ == "__main__":
    main()
