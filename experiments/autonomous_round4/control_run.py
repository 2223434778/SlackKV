"""Common control runtime: OFF, program-order only, or physical fixed-TTL control."""
import os
import sys
import json
from pathlib import Path

from experiments.phase4d_a import runtime
from experiments.autonomous_admission.run import main

if __name__ == "__main__":
    enabled = "--retention-control" in sys.argv
    pin = "--physical-pin" in sys.argv
    for option in ("--retention-control", "--physical-pin"):
        if option in sys.argv:
            sys.argv.remove(option)
    if pin and not enabled:
        raise ValueError("physical pin requires control")
    os.environ["SLACKKV_RETENTION_CONTROL"] = str(int(enabled))
    os.environ["SLACKKV_RETENTION_PIN"] = str(int(pin))
    sys.argv.extend(["--round-directory", "autonomous_round4", "--execution", "default",
                     "--runtime-source", "/media/a6000/D/ZhaoBinHe/slackkv-round4-control-source",
                     "--gate-off"])
    try:
        main()
    finally:
        name = sys.argv[sys.argv.index("--name") + 1]
        directory = Path(__file__).resolve().parents[2] / "results/autonomous_round4" / name
        if directory.exists():
            (directory / "control_config.json").write_text(json.dumps({
                "program_order": enabled, "physical_pin": pin, "ttl_seconds": 1.0,
                "age_guard_seconds": 1.0, "gate_enabled": False,
                "label": "mechanism-matched control; not full Continuum reproduction",
                "args": sys.argv,
            }, indent=2) + "\n")
