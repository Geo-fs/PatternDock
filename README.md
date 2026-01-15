# PatternDock

Notes: 

After install of requirements.txt, please run 
```bash
python -m spacy download en_core_web_sm
```
```bash
pip install fastapi uvicorn scikit-learn
```
```bash
export PYTHONPATH="$(pwd)/src"
python -m mn_event_ai.graph.cluster_incidents
```
```bash
export PYTHONPATH="$(pwd)/src"
python -m mn_event_ai.predict.forecaster
```
```bash
export PYTHONPATH="$(pwd)/src"
uvicorn mn_event_ai.api.app:app --reload --host 0.0.0.0 --port 8000
```
```bash
pip install -e .
```

