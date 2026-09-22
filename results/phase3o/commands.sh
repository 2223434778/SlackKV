python -m pytest -q tests/test_phase3o.py
python -m experiments.phase3o.run_phase3o --stage integration
for policy in OnDemand-K2 Blocking70-K2 Async40-K2; do python -m experiments.phase3o.run_phase3o --stage full --policy "$policy"; done
python -m experiments.phase3o.postprocess
