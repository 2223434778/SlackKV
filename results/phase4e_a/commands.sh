cd /media/a6000/D/ZhaoBinHe/SlackKV
git rev-parse HEAD
git status --short
sha256sum results/phase4c/hybrid_workload.csv results/phase4d_b/vllm_phase4d_b_delta.patch
PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4e_a.py
PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase4e_a.run_phase4e_a
# Driver stopped at the first invalid Stage-B run; preserve diagnostics, do not resume a partial matrix.
PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase4e_a.finalize_invalid
