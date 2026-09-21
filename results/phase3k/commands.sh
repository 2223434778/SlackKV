python -m pytest -q tests/test_phase3k.py
PYTHONPATH=. python experiments/phase3k/run_phase3k.py
PYTHONPATH=. python experiments/phase3k/postprocess.py
