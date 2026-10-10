"""After all frozen screens finish, calibrate diagnostics and run correctness smoke."""
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.autonomous_round4.analyze import schedule


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "results/autonomous_round4"
    deadline = time.monotonic() + 1800
    while True:
        paths = [out / r["name"] / "execution.json" for r in schedule()]
        if all(p.exists() and json.loads(p.read_text())["status"] != "RUNNING" for p in paths):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("baseline ledger incomplete; no follow-up server launched")
        time.sleep(5)
    common = ["--kv-gib", "2", "--gate-off", "--replay-inputs",
              "results/autonomous_capacity/capacity-2/inputs.jsonl.gz"]
    commands = [
        ("resource-off-repaired", ["-m", "experiments.autonomous_round4.diagnostic", "--name", "resource-off-repaired", *common]),
        ("resource-on-repaired", ["-m", "experiments.autonomous_round4.diagnostic", "--name", "resource-on-repaired", "--instrumented", *common]),
        ("retention-smoke", ["-m", "experiments.autonomous_round4.smoke"]),
    ]
    for name, args in commands:
        print(json.dumps({"launch": name, "args": args}), flush=True)
        with (out / (name + ".launch.log")).open("w") as log:
            proc = subprocess.Popen([sys.executable, *args], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            code = proc.wait(timeout=1800)
        execution = out / name / "execution.json"
        status = json.loads(execution.read_text())["status"] if execution.exists() else "NO_LEDGER"
        print(json.dumps({"finished": name, "exit_code": code, "status": status}), flush=True)
        if code or status != "VALID":
            raise RuntimeError("follow-up failure retained; inspect before further launches")
    subprocess.run([sys.executable, "-m", "experiments.autonomous_round4.diagnose",
                    str(out / "resource-on-repaired")], cwd=root, check=True)


if __name__ == "__main__":
    main()
