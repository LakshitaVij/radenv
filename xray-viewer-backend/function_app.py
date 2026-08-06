import azure.functions as func
import json
import logging
import os
import sys
import tempfile
import uuid
import io
import numpy as np
import pydicom
from PIL import Image
from datetime import datetime, timezone
from datetime import timedelta
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.data.tables import TableServiceClient
from openai import OpenAI

# Bridge to the standalone eval pipeline (oracle.py, pacs_access.py, judge.py)
# which lives one directory up, in the "synthetic pipeline" parent folder -
# both read the same dicom_by_visit/{patient}/{visit}/{series} convention from
# local disk, so this works the same way locally as the standalone scripts do.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pacs_access import list_visits  # noqa: E402
from oracle import has_gold, oracle_answer  # noqa: E402
from judge import judge_and_save  # noqa: E402
# READERS = real model-calling functions (MedGemma via Modal, Gemini via API),
# NO_PRIOR_FULL = the default reading prompt - reused as-is so the Model
# Testing tab calls the exact same code path run_model_eval.py's CLI runs use,
# not a re-implementation.
from run_model_eval import READERS, NO_PRIOR_FULL  # noqa: E402

# Only models with real, working credentials wired into THIS backend process
# are live; the rest are real entries in READERS (usable from the CLI) but
# not yet exposed here (missing API keys in local.settings.json).
LIVE_MODEL_TESTING_MODELS = {"medgemma", "gemini-3.6-flash", "gemini-3.5-flash"}

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

CONSOLIDATED_REPORTS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "csv", "consolidated_real_reports.csv"
)
BUNDLE_ACTION_TIERS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "bundle_action_tiers.csv"
)

_patient_demographics_cache = None


def get_patient_demographics(patient_id: str):
    """PatientAge/PatientSex per patient, read once from consolidated_real_reports.csv
    and cached - real data (not placeholder), just not currently exposed by any route."""
    global _patient_demographics_cache
    if _patient_demographics_cache is None:
        import csv
        cache = {}
        try:
            with open(CONSOLIDATED_REPORTS_CSV) as f:
                for row in csv.DictReader(f):
                    pid = row.get("PatientID")
                    if pid and pid not in cache:
                        cache[pid] = {
                            "age": row.get("PatientAge", ""),
                            "sex": row.get("PatientSex", ""),
                        }
        except FileNotFoundError:
            cache = {}
        _patient_demographics_cache = cache
    return _patient_demographics_cache.get(patient_id, {"age": "", "sex": ""})


_action_tiers_cache = None


def get_action_placeholders():
    """Real action_ids from bundle_action_tiers.csv for UI placeholders - not
    invented values. Returns (structured_action_picks, order_panel_picks), 5 each."""
    global _action_tiers_cache
    if _action_tiers_cache is None:
        import csv
        rows = []
        try:
            with open(BUNDLE_ACTION_TIERS_CSV) as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError:
            rows = []
        _action_tiers_cache = rows
    rows = _action_tiers_cache

    essential = [r for r in rows if r.get("tier") == "essential" and not r["action_id"].startswith("(conflict)")]
    other = [r for r in rows if r.get("tier") != "essential" and not r["action_id"].startswith("(conflict)")]

    structured_picks = essential[:5]
    order_panel_picks = other[:5]

    def _shape(r):
        return {"action_id": r["action_id"], "action_name": r["action_name"], "tier": r["tier"]}

    return [_shape(r) for r in structured_picks], [_shape(r) for r in order_panel_picks]

STORAGE_ACCOUNT_NAME = "chartrstorage"
CONTAINER_NAME = "dicom-patients"
TABLE_NAME = "PatientFindings"

PATIENTS = [
    "GRDN003M0HXJ2YHT",
    "GRDN004RP5BFHE0T",
    "GRDN006MCYEVYFE8",
    "GRDN006VFINFR877",
    "GRDN006WOMKG4AVY",
    "GRDN007ELLXXY3OJ",
    "GRDN00A5ZYWHZY7D",
    "GRDN00BE97VCHW4E",
    "GRDN00BJ2R8QRO0L",
    "GRDN00BKCEOKC7S1",
]

_credential = None


def get_credential():
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential


def get_blob_service_client():
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=get_credential())


_user_delegation_key = None
_user_delegation_key_expiry = None


def get_user_delegation_key(blob_service_client):
    """Cache the user delegation key for up to ~50 minutes (max validity is 7 days,
    but shorter-lived keys are safer and this avoids re-fetching on every request)."""
    global _user_delegation_key, _user_delegation_key_expiry

    now = datetime.now(timezone.utc)
    if _user_delegation_key and _user_delegation_key_expiry and now < _user_delegation_key_expiry:
        return _user_delegation_key

    key_start_time = now
    key_expiry_time = now + timedelta(hours=1)

    _user_delegation_key = blob_service_client.get_user_delegation_key(
        key_start_time=key_start_time,
        key_expiry_time=key_expiry_time,
    )
    _user_delegation_key_expiry = now + timedelta(minutes=50)

    return _user_delegation_key


def generate_sas_url_for_blob(blob_service_client, blob_name, expiry_minutes=60):
    user_delegation_key = get_user_delegation_key(blob_service_client)

    sas_token = generate_blob_sas(
        account_name=STORAGE_ACCOUNT_NAME,
        container_name=CONTAINER_NAME,
        blob_name=blob_name,
        user_delegation_key=user_delegation_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes),
    )

    return f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}?{sas_token}"


def get_table_client():
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.table.core.windows.net"
    table_service = TableServiceClient(endpoint=account_url, credential=get_credential())
    try:
        table_service.create_table(TABLE_NAME)
    except Exception:
        pass
    return table_service.get_table_client(TABLE_NAME)


@app.route(route="patients", methods=["GET"])
def get_patients(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /patients called")
    return func.HttpResponse(
        json.dumps({"patients": PATIENTS}),
        mimetype="application/json",
        status_code=200
    )


@app.route(route="patient/{patient_id}/visits", methods=["GET"])
def get_patient_visits(req: func.HttpRequest) -> func.HttpResponse:
    patient_id = req.route_params.get("patient_id")
    logging.info(f"GET /patient/{patient_id}/visits called")

    try:
        blob_service = get_blob_service_client()
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        prefix = f"dicom_by_visit/{patient_id}/"
        blobs = container_client.list_blobs(name_starts_with=prefix)

        visits = {}

        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) < 5:
                continue
            _, _, visit_folder, series_folder, filename = parts[0], parts[1], parts[2], parts[3], parts[4]

            if visit_folder not in visits:
                visits[visit_folder] = {}
            if series_folder not in visits[visit_folder]:
                visits[visit_folder][series_folder] = []

            visits[visit_folder][series_folder].append(blob.name)

        sorted_visits = []
        for visit_folder in sorted(visits.keys()):
            series_list = []
            for series_folder, blob_names in visits[visit_folder].items():
                image_urls = [
                    generate_sas_url_for_blob(blob_service, name)
                    for name in blob_names
                ]
                series_list.append({
                    "series_name": series_folder,
                    "images": image_urls
                })
            sorted_visits.append({
                "visit_folder": visit_folder,
                "series": series_list
            })

        demographics = get_patient_demographics(patient_id)

        return func.HttpResponse(
            json.dumps({
                "patient_id": patient_id,
                "age": demographics["age"],
                "sex": demographics["sex"],
                "visits": sorted_visits,
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error fetching visits: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="transcribe", methods=["POST"])
def transcribe_audio(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /transcribe called")

    try:
        audio_bytes = req.get_body()

        if not audio_bytes:
            return func.HttpResponse(
                json.dumps({"error": "No audio data received"}),
                mimetype="application/json",
                status_code=400
            )

        content_type = req.headers.get("Content-Type", "audio/webm")
        ext = "webm"
        if "wav" in content_type:
            ext = "wav"
        elif "mp3" in content_type or "mpeg" in content_type:
            ext = "mp3"
        elif "ogg" in content_type:
            ext = "ogg"
        elif "m4a" in content_type:
            ext = "m4a"

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        os.remove(tmp_path)

        return func.HttpResponse(
            json.dumps({"text": transcript.text}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="findings", methods=["POST"])
def save_findings(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /findings called")

    try:
        body = req.get_json()

        patient_id = body.get("patient_id")
        visit_folder = body.get("visit_folder")
        series_name = body.get("series_name", "")
        findings = body.get("findings", "")
        impressions = body.get("impressions", "")
        follow_up = body.get("follow_up", "")
        reviewer = body.get("reviewer", "unknown")

        if not patient_id or not visit_folder:
            return func.HttpResponse(
                json.dumps({"error": "patient_id and visit_folder are required"}),
                mimetype="application/json",
                status_code=400
            )

        table_client = get_table_client()

        entity = {
            "PartitionKey": patient_id,
            "RowKey": str(uuid.uuid4()),
            "visit_folder": visit_folder,
            "series_name": series_name,
            "findings": findings,
            "impressions": impressions,
            "follow_up": follow_up,
            "reviewer": reviewer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        table_client.create_entity(entity=entity)

        return func.HttpResponse(
            json.dumps({"status": "saved", "entry": entity}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error saving findings: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="findings/{patient_id}", methods=["GET"])
def get_findings(req: func.HttpRequest) -> func.HttpResponse:
    patient_id = req.route_params.get("patient_id")
    logging.info(f"GET /findings/{patient_id} called")

    try:
        table_client = get_table_client()
        entities = table_client.query_entities(f"PartitionKey eq '{patient_id}'")

        results = []
        for e in entities:
            results.append(dict(e))

        return func.HttpResponse(
            json.dumps({"patient_id": patient_id, "findings": results}, default=str),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error fetching findings: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="export/csv", methods=["GET"])
def export_findings_csv(req: func.HttpRequest) -> func.HttpResponse:
    """
    Export all findings across all patients as a downloadable CSV.
    Optional query param ?patient_id=X to export just one patient.
    """
    import csv
    import io

    patient_filter = req.params.get("patient_id")
    logging.info(f"GET /export/csv called (patient_id={patient_filter})")

    try:
        table_client = get_table_client()

        if patient_filter:
            entities = list(table_client.query_entities(f"PartitionKey eq '{patient_filter}'"))
        else:
            entities = list(table_client.list_entities())

        output = io.StringIO()
        fieldnames = [
            "PartitionKey", "RowKey", "visit_folder", "series_name",
            "findings", "impressions", "follow_up", "reviewer", "timestamp"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for e in entities:
            writer.writerow(dict(e))

        csv_content = output.getvalue()
        filename = f"findings_{patient_filter or 'all'}.csv"

        return func.HttpResponse(
            csv_content,
            mimetype="text/csv",
            status_code=200,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        logging.error(f"Error exporting CSV: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="thumbnail", methods=["GET"])
def get_thumbnail(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate a small PNG thumbnail from a DICOM blob.
    Query params:
      - blob_name: full path to the .dcm blob (e.g. dicom_by_visit/PATIENT/visit_.../series/file.dcm)
      - size: max width/height in pixels (default 150)
    Applies proper windowing and MONOCHROME1 inversion so it displays correctly.
    """
    blob_name = req.params.get("blob_name")
    size = int(req.params.get("size", 150))

    if not blob_name:
        return func.HttpResponse(
            json.dumps({"error": "blob_name query param is required"}),
            mimetype="application/json",
            status_code=400
        )

    try:
        blob_service = get_blob_service_client()
        container_client = blob_service.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)

        dicom_bytes = blob_client.download_blob().readall()

        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = ds.pixel_array.astype(np.float64)

        # Apply rescale slope/intercept (modality LUT)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        pixel_array = pixel_array * slope + intercept

        # Apply windowing (VOI LUT) - use DICOM's own window if present, else min/max
        window_center = getattr(ds, 'WindowCenter', None)
        window_width = getattr(ds, 'WindowWidth', None)

        if window_center is not None and window_width is not None:
            # These can be lists (multi-value) - take first value
            wc = float(window_center[0] if hasattr(window_center, '__iter__') else window_center)
            ww = float(window_width[0] if hasattr(window_width, '__iter__') else window_width)
        else:
            wc = (pixel_array.min() + pixel_array.max()) / 2
            ww = pixel_array.max() - pixel_array.min()

        lower = wc - ww / 2
        upper = wc + ww / 2

        pixel_array = np.clip(pixel_array, lower, upper)
        pixel_array = (pixel_array - lower) / (upper - lower) * 255.0
        pixel_array = pixel_array.astype(np.uint8)

        # MONOCHROME1 means inverted display (higher value = darker)
        photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
        if photometric == 'MONOCHROME1':
            pixel_array = 255 - pixel_array

        img = Image.fromarray(pixel_array, mode='L')
        img.thumbnail((size, size))

        output = io.BytesIO()
        img.save(output, format='PNG')
        output.seek(0)

        return func.HttpResponse(
            output.read(),
            mimetype="image/png",
            status_code=200,
            headers={
                "Cache-Control": "public, max-age=3600"
            }
        )

    except Exception as e:
        logging.error(f"Error generating thumbnail: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="ehr/action-placeholders", methods=["GET"])
def get_action_placeholders_route(req: func.HttpRequest) -> func.HttpResponse:
    """Real action_id placeholders (not invented) for the EHR-mode structured
    follow-up + order panel sections, pulled from bundle_action_tiers.csv."""
    logging.info("GET /ehr/action-placeholders called")
    try:
        structured_picks, order_panel_picks = get_action_placeholders()
        return func.HttpResponse(
            json.dumps({
                "structured_follow_up": structured_picks,
                "order_panel": order_panel_picks,
            }),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error fetching action placeholders: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


def _find_local_visit(patient_id: str, visit_folder: str):
    """Look up the matching VisitInfo (which has the real study_time, not encoded
    in visit_folder's name alone) via the eval pipeline's own local-disk reader -
    reuses pacs_access.py's logic rather than re-deriving StudyTime here."""
    visits = list_visits(patient_id)
    return next((v for v in visits if v.visit_folder == visit_folder), None)


@app.route(route="oracle/{patient_id}/{visit_folder}", methods=["GET"])
def get_oracle(req: func.HttpRequest) -> func.HttpResponse:
    patient_id = req.route_params.get("patient_id")
    visit_folder = req.route_params.get("visit_folder")
    logging.info(f"GET /oracle/{patient_id}/{visit_folder} called")

    try:
        visit = _find_local_visit(patient_id, visit_folder)
        if visit is None or not has_gold(patient_id, visit):
            return func.HttpResponse(
                json.dumps({"has_gold": False}),
                mimetype="application/json",
                status_code=200
            )

        # Anti-leakage: only ever tell the caller whether gold EXISTS for this
        # case, never the actual gold text itself. Gold is used server-side
        # only, inside /api/judge's scoring - it should never cross the wire
        # to the browser, otherwise the reviewer can just read the answer.
        return func.HttpResponse(
            json.dumps({"has_gold": True}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error fetching oracle answer: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="judge", methods=["POST"])
def post_judge(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /judge called")

    try:
        body = req.get_json()
        patient_id = body.get("patient_id")
        visit_folder = body.get("visit_folder")
        model_name = body.get("model_name", "clinician-review")
        findings = body.get("findings", "")
        impressions = body.get("impressions", "")
        had_prior_context = body.get("had_prior_context", False)

        if not patient_id or not visit_folder:
            return func.HttpResponse(
                json.dumps({"error": "patient_id and visit_folder are required"}),
                mimetype="application/json",
                status_code=400
            )

        visit = _find_local_visit(patient_id, visit_folder)
        if visit is None or not has_gold(patient_id, visit):
            return func.HttpResponse(
                json.dumps({"error": "No gold answer available for this case - cannot score."}),
                mimetype="application/json",
                status_code=404
            )

        gold = oracle_answer(patient_id, visit)
        record = judge_and_save(
            patient_id=patient_id,
            visit_folder=visit_folder,
            model_name=model_name,
            model_findings=findings,
            model_impressions=impressions,
            gold_findings=gold.findings,
            gold_impressions=gold.impressions,
            had_prior_context=had_prior_context,
        )

        # Anti-leakage: judge_and_save()'s record includes gold_findings/
        # gold_impressions (for server-side audit storage in judge_results.jsonl)
        # - strip those before returning to the browser. Only the score/
        # reasoning should ever cross the wire, never the actual gold text.
        safe_record = {
            k: v for k, v in record.items()
            if k not in ("gold_findings", "gold_impressions")
        }

        return func.HttpResponse(
            json.dumps(safe_record, default=str),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error judging findings: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="model-eval", methods=["POST"])
def post_model_eval(req: func.HttpRequest) -> func.HttpResponse:
    """Model Testing tab: actually run a real model on the image (reusing
    run_model_eval.py's READERS - the same code path the CLI eval uses), then
    score its real output against gold. Unlike /api/judge (clinician's own
    typed entry), the model's generated findings/impressions ARE returned
    here - that's the point, you're inspecting what the model produced.
    Gold text is also returned here (unlike /api/judge): the model already
    ran blind, before this response is built, so showing gold now doesn't
    leak anything to the thing being tested - it's just letting the human
    reviewer see the model's answer next to the real one."""
    logging.info("POST /model-eval called")

    try:
        body = req.get_json()
        patient_id = body.get("patient_id")
        visit_folder = body.get("visit_folder")
        model_key = body.get("model")

        if not patient_id or not visit_folder or not model_key:
            return func.HttpResponse(
                json.dumps({"error": "patient_id, visit_folder, and model are required"}),
                mimetype="application/json",
                status_code=400
            )

        if model_key not in LIVE_MODEL_TESTING_MODELS:
            return func.HttpResponse(
                json.dumps({"error": f"{model_key} is not yet wired up in this backend "
                                      f"(missing API key) - coming soon."}),
                mimetype="application/json",
                status_code=501
            )

        visit = _find_local_visit(patient_id, visit_folder)
        if visit is None or not has_gold(patient_id, visit):
            return func.HttpResponse(
                json.dumps({"error": "No gold answer available for this case - cannot score."}),
                mimetype="application/json",
                status_code=404
            )
        if not visit.series:
            return func.HttpResponse(
                json.dumps({"error": "No image series found for this visit."}),
                mimetype="application/json",
                status_code=404
            )

        model_name, read_fn = READERS[model_key]
        series_name = visit.series[0].series_name

        report = read_fn(patient_id, visit_folder, series_name, NO_PRIOR_FULL)

        gold = oracle_answer(patient_id, visit)
        record = judge_and_save(
            patient_id=patient_id,
            visit_folder=visit_folder,
            model_name=model_name,
            model_findings=report.findings,
            model_impressions=report.impressions,
            gold_findings=gold.findings,
            gold_impressions=gold.impressions,
            had_prior_context=False,
        )

        safe_record = dict(record)
        # Unlike /api/judge, the model's own generated text IS returned - it's
        # not the clinician's input, it's the thing being tested. Gold is also
        # returned here (see docstring) so the reviewer can see it next to the
        # model's answer.
        safe_record["model_generated_findings"] = report.findings
        safe_record["model_generated_impressions"] = report.impressions

        return func.HttpResponse(
            json.dumps(safe_record, default=str),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error running model eval: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
