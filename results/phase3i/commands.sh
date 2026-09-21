python -m pytest -q tests/test_phase3i.py
PYTHONPATH=. python experiments/phase3i/run_phase3i.py --stage baseline
PYTHONPATH=. python experiments/phase3i/run_phase3i.py --stage integration
PYTHONPATH=. python experiments/phase3i/run_phase3i.py --stage full
PYTHONPATH=. python experiments/phase3i/postprocess.py
