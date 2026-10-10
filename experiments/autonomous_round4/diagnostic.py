"""Resource trace calibration uses a common isolated source with instrumentation off/on."""
import os
import sys

from experiments.autonomous_admission.run import main

if __name__ == "__main__":
    enabled = "--instrumented" in sys.argv
    if enabled:
        sys.argv.remove("--instrumented")
        sys.argv.append("--trace")
    os.environ["SLACKKV_RESOURCE_TRACE"] = "1" if enabled else "0"
    sys.argv.extend(["--round-directory", "autonomous_round4", "--execution", "default",
                     "--runtime-source", "/media/a6000/D/ZhaoBinHe/slackkv-round4-diagnostic-v2-source"])
    main()
