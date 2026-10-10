"""Round4 uses the same isolated runtime and compiled execution for every method."""
import sys

from experiments.autonomous_admission.run import main

if __name__ == "__main__":
    sys.argv.extend([
        "--round-directory", "autonomous_round4", "--execution", "default",
        "--runtime-source", "/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source",
    ])
    main()
