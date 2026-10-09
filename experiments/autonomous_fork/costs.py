"""Reproducible conditional opportunity screen using archived Phase 0 bandwidth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.autonomous_fork.model import fork_footprint, streaming_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="keep exported original unchanged")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    table = root / "results/phase0/summary_metrics.csv"
    records = list(csv.DictReader(table.read_text().splitlines()))
    def bandwidth(direction):
        return float(next(r["bandwidth_GBps_p50"] for r in records
                          if r["direction"] == direction and int(r["size_bytes"]) == 2**30))
    cpu, peer = bandwidth("pinned_cpu_to_gpu0"), bandwidth("gpu1_to_gpu0")
    config = json.loads((args.directory / "environment.json").read_text())["model_config"]
    qwen7b = dict(num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4, hidden_size=3584)
    result = dict(phase0_reference_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                  reference_note="historical Phase 0 CUDA12.1 measurement, not fresh Torch2.11 timing or universal bandwidth limit",
                  pinned_cpu_GBps=cpu, peer_gpu1_to_gpu0_GBps=peer,
                  transfer_only_peer_improvement_percent=100 * (1 - cpu / peer),
                  qwen1_5b_fork=fork_footprint(config, 24576, 256, 128, 8),
                  qwen1_5b_second_turn_fork=fork_footprint(config, 24576, 448, 128, 8),
                  qwen7b_config_source="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json",
                  qwen7b_fork_estimate=fork_footprint(qwen7b, 24576, 256, 128, 8),
                  qwen1_5b_active_root_streaming=streaming_cost(config, 24576, 256, cpu),
                  scope="synthetic exact full-attention fork; shared root fetched once per attention iteration, not per branch; no claim about CPU attention or compression")
    (args.output or args.directory / "cost_model.json").write_text(json.dumps(result, indent=2) + "\n", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
