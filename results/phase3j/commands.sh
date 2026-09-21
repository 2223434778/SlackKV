python -m pytest -q tests/test_phase3j.py tests/test_phase3i.py
PYTHONPATH=. python experiments/phase3j/run_phase3j.py --stage calibration
PYTHONPATH=. python experiments/phase3j/run_phase3j.py --stage validation
PYTHONPATH=. python experiments/phase3j/postprocess.py
