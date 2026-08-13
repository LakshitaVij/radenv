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

1. **OpenEMR** — apply `openemr-setup/orthanc-addition.patch` against
   `openemr/` first (it edits `docker/development-easy/docker-compose.yml`
   to add the Orthanc service, so it has to land before containers start,
   not after):
   ```bash
   cd openemr && git apply ../openemr-setup/orthanc-addition.patch && cd ..
   ```
   Then bring the stack up (`docker compose up` in
   `openemr/docker/development-easy`) and run the base seed scripts in
   order:
   ```bash
   python openemr-setup/seed_patients_and_vitals.py
   python openemr-setup/seed_encounter_history.py
   python openemr-setup/seed_action_library.py
   python openemr-setup/seed_action_hierarchy.py
   ```
   OpenEMR is expected at `https://localhost:9300` (login `admin`/`pass`).

2. **X-ray Viewer form** (the embedded viewer patients/agents actually use
   inside OpenEMR - separate from the standalone dev frontend in step 4
   below) - one command applies its patches, seeds its form attachments,
   and builds + deploys the React frontend into OpenEMR:
   ```bash
   bash openemr-setup/setup_xray_viewer.sh
   ```
   Requires the OpenEMR docker stack already up (step 1) and `python3`/`npm`
   on PATH. Safe to re-run - patch application and the DB seed are both
   idempotent.

3. **Orthanc** — load the real DICOM data (Orthanc itself is already up
   from step 1):
   ```bash
   python load_dicom_into_orthanc.py
   ```

4. **Standalone frontend dev server** (optional - only needed if you're
   iterating on `xray-viewer-frontend/` source directly; the embedded
   build from step 2 is what OpenEMR/the agent actually serve):
   ```bash
   cd xray-viewer-frontend && npm run dev
   ```
   Expected at `http://localhost:5173`, proxies `/dicom-web` to Orthanc.

5. **Backend** (Azure Functions, only used by the standalone frontend
   above):
   ```bash
   cd xray-viewer-backend && func start
   ```
   Expected at `http://localhost:7071`.

6. **Run the agent evaluation** (real OpenRouter + OpenAI calls — has real cost):
   ```bash
   cd computer_use_eval && python run_task_batch.py
   ```
   Runs the 8 designed evaluation tasks end to end (agent episode + grading per task) and writes results to `computer_use_eval/task_batch_v1/`.
