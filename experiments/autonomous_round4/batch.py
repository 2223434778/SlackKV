"""Sequential frozen screen; no replacement runs or concurrent owned servers."""
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.autonomous_round4.analyze import schedule


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "results/autonomous_round4"
    first = out / "gate-2-p1-k2/execution.json"
    deadline = time.monotonic() + 1800
    while json.loads(first.read_text())["status"] == "RUNNING":
        if time.monotonic() > deadline:
            raise TimeoutError("first owned run did not finalize; no additional launch")
        time.sleep(5)
    for row in schedule()[1:]:
        command = [sys.executable, "-m", "experiments.autonomous_round4.run",
                   "--name", row["name"], "--kv-gib", str(row["budget"]),
                   "--replay-inputs", "results/autonomous_capacity/capacity-2/inputs.jsonl.gz"]
        if row["method"] == "off":
            command.append("--gate-off")
        print(json.dumps({"launch": row["name"], "command": command}), flush=True)
        with (out / (row["name"] + ".launch.log")).open("w") as log:
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                # The run watchdog owns server cleanup; do not start another server.
                raise TimeoutError(f"owned runner {proc.pid} exceeded budget; inspect cleanup")
        execution = out / row["name"] / "execution.json"
        print(json.dumps({"finished": row["name"], "exit_code": code,
                          "status": json.loads(execution.read_text())["status"] if execution.exists() else "NO_LEDGER"}), flush=True)
        if not execution.exists() or json.loads(execution.read_text())["status"] == "RUNNING":
            raise RuntimeError("unfinalized run; refuse subsequent launch")


if __name__ == "__main__":
    main()
