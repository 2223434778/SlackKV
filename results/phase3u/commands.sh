PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase3u.py
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3u/run_phase3u.py --stage x-only
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3u/run_phase3u.py --stage integration
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3u/run_phase3u.py --stage full
PYTHONPATH=. python3 experiments/phase3u/postprocess.py
