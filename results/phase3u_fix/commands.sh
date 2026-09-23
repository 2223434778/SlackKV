PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase3u_fix.py
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3u_fix/run_phase3u_fix.py --stage x-only
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3u_fix/run_phase3u_fix.py --stage stage-a
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3u_fix/run_phase3u_fix.py --stage stage-b
