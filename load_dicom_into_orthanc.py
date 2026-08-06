"""
load_dicom_into_orthanc.py

Uploads every real DICOM file in dicom_by_visit/ into the local Orthanc
server via its REST API (POST /instances, raw DICOM bytes) - the real
data source OHIF/Cornerstone will read from via DICOMweb.

Read-only against the source files; only writes to Orthanc's own storage.
"""

import sys
from pathlib import Path

import requests

DICOM_ROOT = Path("/Users/lakshitavij/synthetic pipeline/dicom_by_visit")
ORTHANC_URL = "http://localhost:8042"

if __name__ == "__main__":
    files = sorted(DICOM_ROOT.rglob("*.dcm"))
    print(f"Found {len(files)} DICOM files to upload.")

    ok, failed = 0, 0
    for i, f in enumerate(files, 1):
        with f.open("rb") as fh:
            data = fh.read()
        resp = requests.post(f"{ORTHANC_URL}/instances", data=data, headers={"Content-Type": "application/dicom"})
        if resp.status_code in (200, 201):
            ok += 1
        else:
            failed += 1
            print(f"  FAILED ({resp.status_code}): {f} - {resp.text[:200]}")
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} processed...")

    print(f"\nDone. {ok} uploaded OK, {failed} failed.")

    # Sanity check: ask Orthanc how many studies it now has
    stats = requests.get(f"{ORTHANC_URL}/statistics").json()
    print(f"Orthanc now reports: {stats.get('CountStudies')} studies, {stats.get('CountInstances')} instances.")
