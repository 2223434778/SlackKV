"""Deterministic resource arithmetic; no proxy-to-Qwen or latency extrapolation."""
import argparse
import json
from pathlib import Path

from experiments.deployment_opportunity.analyze import kv_bytes_per_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    inventory = json.loads((root / 'server_inventory.json').read_text())
    large = json.loads((root / 'qwen32b_metadata/config.json').read_text())
    index = json.loads((root / 'qwen32b_metadata/model.safetensors.index.json').read_text())
    small_bytes = kv_bytes_per_token(inventory['cached_model_config'])
    large_bytes = kv_bytes_per_token(large)
    total_vram = 2 * 49140 * 2**20
    weights = index['metadata']['total_size']
    result = dict(dtype_bytes=2, context_tokens=32768,
                  qwen1p5b_kv_bytes_per_token=small_bytes,
                  qwen32b_kv_bytes_per_token=large_bytes,
                  qwen32b_tp2_per_rank_kv_bytes_per_token=large_bytes // 2,
                  qwen32b_weight_payload_bytes=weights,
                  qwen1p5b_measured_gpu_pool_tokens_per_replica=1355712,
                  qwen1p5b_measured_pool_whole_32k_inputs_per_replica=1355712 // 32768,
                  qwen32b_tp2_85pct_vram_minus_weights_upper_bytes=int(total_vram * .85) - weights,
                  gpu_hours_added=0,
                  caveats=['Measured 1.5B pool from previous compiled run, not remeasured now.',
                           '32B weight-only subtraction is an optimistic KV capacity upper bound, not a runtime measurement.',
                           'Ignoring output, activation, CUDA graphs, temporary buffers, CPU model loading and service headroom.',
                           'Full-host duplicate copies differ from exclusive GPU+host placement; do not sum demand twice.',
                           'Session counts below are feasibility scenarios, not observed production demand.'])
    result['scenarios'] = [dict(model=model, sessions=sessions,
                                distinct_32k_input_kv_bytes=sessions * 32768 * token_bytes,
                                full_host_copies_fit_idle_memavailable=(sessions * 32768 * token_bytes <=
                                                                       inventory['meminfo_bytes']['MemAvailable']))
                           for model, token_bytes in [('Qwen2.5-1.5B', small_bytes), ('Qwen2.5-32B', large_bytes)]
                           for sessions in (4, 8, 16, 32, 64)]
    (root / 'capacity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
