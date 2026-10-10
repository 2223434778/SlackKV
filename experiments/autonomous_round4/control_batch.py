"""Preregistered control triplet; reverse replication only after a passing screen."""
import json
import subprocess
import sys
from pathlib import Path

from experiments.autonomous_round4.control_summary import summarize


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "results/autonomous_round4"
    if json.loads((out / "retention-smoke/execution.json").read_text())["status"] != "VALID":
        raise RuntimeError("retention safety smoke did not pass")
    def triplet(prefix, methods):
        for method in methods:
            name = f"{prefix}-{method}"
            args = [sys.executable, "-m", "experiments.autonomous_round4.control_run",
                    "--name", name, "--kv-gib", "2", "--replay-inputs",
                    "results/autonomous_capacity/capacity-2/inputs.jsonl.gz"]
            if method != "off":
                args.append("--retention-control")
            if method == "pin":
                args.append("--physical-pin")
            print(json.dumps({"launch": name, "command": args}), flush=True)
            with (out / (name + ".launch.log")).open("w") as log:
                proc = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                code = proc.wait(timeout=1800)
            execution = out / name / "execution.json"
            status = json.loads(execution.read_text())["status"] if execution.exists() else "NO_LEDGER"
            print(json.dumps({"finished": name, "exit_code": code, "status": status}), flush=True)
            if code or status != "VALID":
                raise RuntimeError("control failure retained; inspect before further runs")
    triplet("control", ["off", "order", "pin"])
    screen = summarize(out)
    (out / "control_screen_decision.json").write_text(json.dumps(screen["comparisons"], indent=2) + "\n")
    promising = any(screen["comparisons"][k]["pass_screen"] for k in ("program_order", "physical_retention"))
    if promising:
        triplet("reverse", ["pin", "order", "off"])
    else:
        print("neither control passed; no TTL/age tuning or replacement", flush=True)
    subprocess.run([sys.executable, "-m", "experiments.autonomous_round4.control_summary", str(out)], cwd=root, check=True)


if __name__ == "__main__":
    main()
