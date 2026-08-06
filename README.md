# radenv

Computer-use AI agent evaluation pipeline: an agent reads chest X-rays
inside OpenEMR (via an Orthanc/DICOMweb-backed viewer) and is graded
against real gold reports.

## Components

- `openemr-setup/` — patch + seed scripts for the OpenEMR instance
- `xray-viewer-frontend/` — React/Vite X-ray viewer (Cornerstone3D), embedded in OpenEMR
- `xray-viewer-backend/` — Azure Functions API the frontend calls (patients/visits)
- `oracle.py`, `judge.py`, `pacs_access.py`, `load_dicom_into_orthanc.py`, `run_model_eval.py` — shared pipeline core
- `computer_use_eval/` — agent execution (`agent_episode.py`) + grading (`grade_episode.py`, `grade_process.py`), orchestrated by `run_task_batch.py`
- `dicom_by_visit/`, `csv/consolidated_real_reports.csv`, `generated_followups_fin.csv` — real data

## Setup

```bash
pip install -r requirements.txt
pip install -r xray-viewer-backend/requirements.txt
cd xray-viewer-frontend && npm install
```

Playwright (used by `agent_episode.py` to drive the browser):

```bash
pip install playwright && playwright install chromium
```

Secrets (git-ignored, must be created locally):
- `computer_use_eval/.openrouter_key` — OpenRouter API key (used to call Gemini 3.1 Pro for agent episodes)
- `OPENAI_API_KEY` env var — used by `judge.py` for grading

## Run order

1. **OpenEMR** — bring up your OpenEMR instance, apply `openemr-setup/orthanc-addition.patch`, then run the seed scripts in order:
   ```bash
   python openemr-setup/seed_patients_and_vitals.py
   python openemr-setup/seed_encounter_history.py
   python openemr-setup/seed_action_library.py
   python openemr-setup/seed_action_hierarchy.py
   ```
   OpenEMR is expected at `https://localhost:9300`.

2. **Orthanc** — start Orthanc (port `8042`), then load the real DICOM data:
   ```bash
   python load_dicom_into_orthanc.py
   ```

3. **Backend** (Azure Functions):
   ```bash
   cd xray-viewer-backend && func start
   ```
   Expected at `http://localhost:7071`.

4. **Frontend**:
   ```bash
   cd xray-viewer-frontend && npm run dev
   ```
   Expected at `http://localhost:5173`, proxies `/dicom-web` to Orthanc.

5. **Run the agent evaluation** (real OpenRouter + OpenAI calls — has real cost):
   ```bash
   cd computer_use_eval && python run_task_batch.py
   ```
   Runs the 8 designed evaluation tasks end to end (agent episode + grading per task) and writes results to `computer_use_eval/task_batch_v1/`.
