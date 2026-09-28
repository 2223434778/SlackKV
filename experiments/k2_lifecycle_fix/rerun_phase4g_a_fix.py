"""Run the unchanged Phase 4G-A-FIX matrix against the repaired K2 runtime."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from experiments.phase4g_a_fix import run_phase4g_a_fix as harness

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4g_a_fix_k2repair"
START_COMMIT = "84c75c880fddf5e98cefb70d0b99da43cfe9cb5b"
VLLM_DIFF_SHA = "4ec03c5f7bcb331485becea471d6d46b28b2b9ec3f8b0a2997f62e7e8bf15618"


def main() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != START_COMMIT:
        raise RuntimeError(f"rerun must start at K2 fix commit {START_COMMIT}, found {head}")
    source = ROOT / "results/phase4c/hybrid_workload.csv"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105":
        raise RuntimeError(f"frozen workload mismatch: {sha}")
    if harness.vllm_diff_sha() != VLLM_DIFF_SHA:
        raise RuntimeError("repaired vLLM source fingerprint mismatch")
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite rerun output directory: {OUT}")

    harness.OUT = OUT
    harness.START_COMMIT = START_COMMIT
    harness.VLLM_EXPECTED_DIFF_SHA = VLLM_DIFF_SHA
    save_artifacts = harness.save_initial_artifacts

    def save_rerun_artifacts():
        result = save_artifacts()
        (OUT / "commands.sh").write_text(
            "set -euo pipefail\n"
            f"cd {ROOT}\n"
            "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python "
            "-u -m experiments.k2_lifecycle_fix.rerun_phase4g_a_fix\n",
            encoding="utf-8",
        )
        return result

    harness.save_initial_artifacts = save_rerun_artifacts
    harness.run_all()


if __name__ == "__main__":
    main()
