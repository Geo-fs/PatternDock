# PatternDock

Notes: 

Install from scratch: 
```bash
cd /workspaces/PatternDock
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# If you have pyproject.toml:
pip install -e . || true

# If editable install fails, fallback:
export PYTHONPATH="$(pwd)/src"

python -m mn_event_ai.jobs.run_pipeline
python -m mn_event_ai.nlp.infer_textcnn
python -m mn_event_ai.graph.cluster_incidents
```
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # if you want editable install as primary
python -m mn_event_ai.jobs.run_all
```

