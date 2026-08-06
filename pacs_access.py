"""
pacs_access.py

Direct data-access layer for Task 2 (diagnose from organized patient
folders, no UI navigation). Reads DICOM files out of the
dicom_by_visit/{PatientID}/visit_{StudyDate}_{suffix}/{SeriesDescription}_{SeriesUID}/
folder structure produced by organize_dicom_by_patient.py.

Two functions, meant to become tool calls for an agent:
    list_visits(patient_id)              -> list of visits + their series
    get_image(patient_id, visit_folder, series_name) -> PNG bytes (base64)
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

DICOM_ROOT = Path("/Users/lakshitavij/synthetic pipeline/dicom_by_visit")


@dataclass
class SeriesInfo:
    series_name: str          # e.g. "PA Chest Landscape" (folder name, minus the UID suffix)
    folder_name: str          # full folder name, e.g. "PA Chest Landscape_1.2.826.0.1.3680043."
    dcm_path: str              # path to the .dcm file inside


@dataclass
class VisitInfo:
    visit_folder: str          # e.g. "visit_20100415_87206065"
    study_date: str            # e.g. "20100415"
    study_time: str | None      # e.g. "111732.415117", read from the DICOM header -
                                # StudyDate alone isn't unique (a patient can have two
                                # visits same day); StudyTime disambiguates and matches
                                # the gold CSV's StudyTime column exactly.
    series: list[SeriesInfo]


def _read_study_time(dcm_path: str) -> str | None:
    """Metadata-only read (no pixel data) just to get StudyTime for disambiguation."""
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return str(ds.get("StudyTime")) or None
    except Exception:
        return None


def list_visits(patient_id: str) -> list[VisitInfo]:
    """Return every visit for a patient, sorted chronologically, each with its series."""
    patient_dir = DICOM_ROOT / patient_id
    if not patient_dir.is_dir():
        raise ValueError(f"No such patient: {patient_id!r}")

    visits = []
    for visit_dir in sorted(patient_dir.iterdir()):
        if not visit_dir.is_dir() or not visit_dir.name.startswith("visit_"):
            continue

        # visit_{StudyDate}_{suffix}
        parts = visit_dir.name.split("_")
        study_date = parts[1] if len(parts) > 1 else "unknown"

        series_list = []
        study_time = None
        for series_dir in sorted(visit_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            dcm_files = list(series_dir.glob("*.dcm"))
            if not dcm_files:
                continue
            # folder name is "{SeriesDescription}_{SeriesUID[:20]}" - strip the UID
            # suffix back off by cutting at the last underscore-separated numeric run
            series_name = series_dir.name.rsplit("_", 1)[0]
            series_list.append(SeriesInfo(
                series_name=series_name,
                folder_name=series_dir.name,
                dcm_path=str(dcm_files[0]),
            ))
            if study_time is None:
                study_time = _read_study_time(str(dcm_files[0]))

        visits.append(VisitInfo(
            visit_folder=visit_dir.name,
            study_date=study_date,
            study_time=study_time,
            series=series_list,
        ))

    visits.sort(key=lambda v: (v.study_date, v.study_time or ""))
    return visits


def _dicom_to_png_bytes(dcm_path: str) -> bytes:
    """Load a DICOM file and render it to 8-bit grayscale PNG bytes."""
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # Apply VOI LUT / windowing if present, otherwise min-max normalize.
    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        pass

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    img = Image.fromarray(arr.astype(np.uint8), mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_image(patient_id: str, visit_folder: str, series_name: str) -> str:
    """
    Return a base64-encoded PNG for the requested series, ready to drop into
    an API `image` content block:
        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/png", "data": get_image(...)}}
    """
    visits = list_visits(patient_id)
    visit = next((v for v in visits if v.visit_folder == visit_folder), None)
    if visit is None:
        raise ValueError(f"No such visit {visit_folder!r} for patient {patient_id!r}")

    series = next((s for s in visit.series if s.series_name == series_name), None)
    if series is None:
        available = [s.series_name for s in visit.series]
        raise ValueError(
            f"No series named {series_name!r} in visit {visit_folder!r}; "
            f"available: {available}"
        )

    png_bytes = _dicom_to_png_bytes(series.dcm_path)
    return base64.standard_b64encode(png_bytes).decode("utf-8")


if __name__ == "__main__":
    import sys

    patient_id = sys.argv[1] if len(sys.argv) > 1 else "GRDN003M0HXJ2YHT"
    visits = list_visits(patient_id)
    print(f"{patient_id}: {len(visits)} visits")
    for v in visits:
        series_names = [s.series_name for s in v.series]
        print(f"  {v.visit_folder} ({v.study_date}): {series_names}")

    if visits and visits[0].series:
        first = visits[0]
        series_name = first.series[0].series_name
        b64 = get_image(patient_id, first.visit_folder, series_name)
        print(f"\nFetched {series_name!r} from {first.visit_folder}: "
              f"{len(b64)} base64 chars")
