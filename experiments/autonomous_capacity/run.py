"""Independent budget ledger, compiled OnDemand by default."""
import sys

from experiments.autonomous_admission.run import main

if __name__ == "__main__":
    sys.argv.extend(["--round-directory", "autonomous_capacity", "--execution", "default"])
    main()
