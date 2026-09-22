python -m pytest -q tests/test_phase3n.py
for ms in 5 20 30 40 70; do python -m experiments.phase3n.run_phase3n --offset-ms "$ms"; done
python -m experiments.phase3n.postprocess
