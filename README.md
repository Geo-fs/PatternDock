## Quickstart (Codespaces / Linux)

```bash
cd /workspaces/PatternDock

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
### If you have pyproject.toml and want editable installs:
```bash
pip install -e . || true
```
### Run the full pipeline (ingest → infer → cluster)
```bash
python -m mn_event_ai.jobs.run_all
```

### Optional: run forecasting too
```bash
python -m mn_event_ai.jobs.run_all --forecast
```
Run the API
```bash
uvicorn mn_event_ai.api.app:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

GET /health

GET /articles?limit=50

GET /incidents?limit=50

GET /search?q=keywords

POST /run/pipeline

**This makes the repo “clone → run” instead of “clone → interpret runes.”**

---

### 4) Commit + push (so GitHub and Codespaces stop gaslighting you)
```bash
git add src/mn_event_ai/jobs/run_all.py .gitignore README.md data/.gitkeep models/.gitkeep
git commit -m "Add run_all orchestrator, improve gitignore, simplify quickstart"
git push
```
---
---

## Windows Setup (ZIP Release)

*These steps assume you downloaded the project as a .zip file from GitHub Releases and extracted it to a folder on your PC.*

### 0) Requirements

**You need:**

- Python 3.10+ (3.11 or 3.12 is fine)

*Git is NOT required for ZIP installs*

-  terminal: **PowerShell** or Windows Terminal recommended

- Check Python works:
```bash
python --version
```

If that doesn’t work, install Python from python.org and *make sure “Add Python to PATH” is checked.*

### 1) Extract the ZIP

1: Download the ZIP release from GitHub

2: Right-click → Extract All

3: Move the extracted folder somewhere sane like:
```bash
C:\Projects\PatternDock\
```

### 2) Open PowerShell in the Project Folder

#### Option A: In File Explorer, open the folder, click the address bar, type:
```bash
powershell
```

#### Option B: Start menu → PowerShell → cd into it:

```bash
cd C:\Projects\PatternDock
```

### 3) Create and Activate a Virtual Environment (venv)

**Create venv:**
```bash
python -m venv .venv
```

Activate it (PowerShell):
```bash
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation with an execution policy error, run this once:
```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then try activation again:
```bash
.\.venv\Scripts\Activate.ps1
```

*You should now see (.venv) at the start of your terminal line.*

### 4) Install Dependencies

**Upgrade pip tools:**
```bash
python -m pip install --upgrade pip setuptools wheel
```

Install requirements:
```bash
pip install -r requirements.txt
```

### 5) Make the Package Importable

**Recommended (if pyproject.toml is included (It is most of the time...))**

This makes the package installable in editable mode:
```bash
pip install -e .
```
**Fallback (if editable install fails)**

Set PYTHONPATH:
```bash
$env:PYTHONPATH = "$PWD\src"
```
Run the Full Pipeline (One Command)

This runs:

- RSS ingest → normalize/dedupe

- TextCNN inference (if models/textcnn.pt exists)

- Incident clustering (if predictions exist)

```bash
python -m mn_event_ai.jobs.run_all
```

#### Optional: include forecasting (DRN baseline):
```bash
python -m mn_event_ai.jobs.run_all --forecast
```
Run the API (Optional)

Start the FastAPI server:
```bash
uvicorn mn_event_ai.api.app:app --reload --host 127.0.0.1 --port 8000
```

Then open in your browser:
```text
http://127.0.0.1:8000/health

http://127.0.0.1:8000/docs (interactive API docs)
```
**Useful endpoints:**

- GET /articles?limit=50

- GET /incidents?limit=50

- GET /search?q=keywords

- POST /run/pipeline

### Common Windows Issues
- “python is not recognized”

  - Python isn’t on PATH. Reinstall Python and check Add Python to PATH.

- “Activate.ps1 cannot be loaded”

  - PowerShell execution policy is blocking scripts:
```bash
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
- Slow or failing installs

    - Try upgrading pip first:
```bash
      python -m pip install --upgrade pip setuptools wheel
```
#### Notes for ZIP Releases

Generated outputs are written to:

- data/raw/

- data/processed/

- models/

*If a ZIP release includes a pretrained model (models/textcnn.pt), inference will run automatically.*

**If it does not, you’ll need to train first:**
```bash
python -m mn_event_ai.nlp.weak_label
python -m mn_event_ai.nlp.train_textcnn
python -m mn_event_ai.nlp.infer_textcnn
python -m mn_event_ai.graph.cluster_incidents
```
