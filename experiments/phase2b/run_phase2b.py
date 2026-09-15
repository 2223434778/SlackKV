#!/usr/bin/env python3
"""Phase 2B: real Hugging Face Transformer KV offload/restore validation."""
from __future__ import annotations

import csv
import gc
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import transformers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from slackkv.simulator.transfer_model import TransferModel
RESULTS = ROOT / "results" / "phase2b"
DEVICE = torch.device("cuda:0")
PREFERRED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXT_TOKENS = 4096
DECODE_TOKENS = 16
WARMUP_RUNS = 1
MEASURED_RUNS = 3
DTYPE = torch.float16


LegacyCache = Tuple[Tuple[torch.Tensor, ...], ...]


def kv_byte_count(legacy_cache: LegacyCache) -> int:
    return sum(tensor.numel() * tensor.element_size() for layer in legacy_cache for tensor in layer)


def map_legacy_cache(
    legacy_cache: LegacyCache, mapper: Callable[[torch.Tensor], torch.Tensor]
) -> LegacyCache:
    return tuple(tuple(mapper(tensor) for tensor in layer) for layer in legacy_cache)


def legacy_shapes(legacy_cache: LegacyCache) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(tuple(tensor.shape) for tensor in layer) for layer in legacy_cache)


def token_sequences_equal(first: Iterable[int], second: Iterable[int]) -> bool:
    return list(first) == list(second)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cuda_timed_copy(
    destination: torch.Tensor, source: torch.Tensor, stream: torch.cuda.Stream
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        destination.copy_(source, non_blocking=True)
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end))


def cuda_timed_call(call: Callable[[], Any]) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def deterministic_input(tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    paragraph = (
        "SlackKV validates deterministic Transformer cache offload and restoration "
        "during an ordinary coding-agent pause. The same prefix must support an "
        "identical greedy continuation after the cache returns from pinned CPU memory. "
    )
    text = paragraph * 1024
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=CONTEXT_TOKENS,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] != CONTEXT_TOKENS:
        raise RuntimeError(f"expected {CONTEXT_TOKENS} tokens, got {input_ids.shape[1]}")
    return input_ids.to(DEVICE), torch.ones_like(input_ids, device=DEVICE)


def make_cpu_snapshot(legacy_cache: LegacyCache) -> tuple[LegacyCache, float]:
    stream = torch.cuda.Stream(device=DEVICE)
    cpu_cache = map_legacy_cache(
        legacy_cache,
        lambda tensor: torch.empty(
            tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
        ),
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for gpu_layer, cpu_layer in zip(legacy_cache, cpu_cache):
            for gpu_tensor, cpu_tensor in zip(gpu_layer, cpu_layer):
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        end.record(stream)
    end.synchronize()
    return cpu_cache, float(start.elapsed_time(end))


def restore_gpu_cache(cpu_cache: LegacyCache) -> tuple[LegacyCache, float]:
    stream = torch.cuda.Stream(device=DEVICE)
    gpu_cache = map_legacy_cache(
        cpu_cache,
        lambda tensor: torch.empty(tensor.shape, dtype=tensor.dtype, device=DEVICE),
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for cpu_layer, gpu_layer in zip(cpu_cache, gpu_cache):
            for cpu_tensor, gpu_tensor in zip(cpu_layer, gpu_layer):
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        end.record(stream)
    end.synchronize()
    return gpu_cache, float(start.elapsed_time(end))


def greedy_continue(
    model: Any, cache: DynamicCache, first_token: torch.Tensor
) -> dict[str, Any]:
    generated: list[int] = []
    current = first_token
    first_logits: torch.Tensor | None = None
    first_step_ms = 0.0
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()

    for step in range(DECODE_TOKENS):
        outputs, step_ms = cuda_timed_call(
            lambda: model(input_ids=current, past_key_values=cache, use_cache=True)
        )
        if step == 0:
            first_step_ms = step_ms
            first_logits = outputs.logits[:, -1, :].float().cpu()
        generated.append(int(current.item()))
        current = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = outputs.past_key_values

    total_end.record()
    total_end.synchronize()
    if first_logits is None:
        raise RuntimeError("no continuation logits captured")
    return {
        "token_ids": generated,
        "first_logits": first_logits,
        "first_token_decode_ms": first_step_ms,
        "decode_16_ms": float(total_start.elapsed_time(total_end)),
        "cache": cache,
    }


def load_model_and_tokenizer() -> tuple[str, Any, Any, str | None]:
    error: str | None = None
    for model_id in (PREFERRED_MODEL, FALLBACK_MODEL):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE)
            model.to(DEVICE)
            model.eval()
            return model_id, model, tokenizer, error
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            gc.collect()
    raise RuntimeError(f"could not load preferred or fallback Qwen model: {error}")


def one_full_run(model: Any, tokenizer: Any, model_id: str) -> dict[str, Any]:
    input_ids, attention_mask = deterministic_input(tokenizer)
    torch.cuda.synchronize(DEVICE)
    allocated_before_prefill = int(torch.cuda.memory_allocated(DEVICE))
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    torch.cuda.synchronize(DEVICE)

    prefill_cache = prefill.past_key_values
    legacy = (
        prefill_cache.to_legacy_cache()
        if hasattr(prefill_cache, "to_legacy_cache")
        else prefill_cache
    )
    kv_bytes = kv_byte_count(legacy)
    shapes = legacy_shapes(legacy)
    first_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    allocated_after_prefill = int(torch.cuda.memory_allocated(DEVICE))
    cpu_cache, d2h_ms = make_cpu_snapshot(legacy)
    with torch.inference_mode():
        baseline = greedy_continue(
            model, DynamicCache.from_legacy_cache(legacy), first_token
        )
    baseline_token_ids = baseline["token_ids"]
    baseline_text = tokenizer.decode(baseline_token_ids, skip_special_tokens=False)

    del baseline["cache"], prefill_cache, legacy
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    allocated_after_release = int(torch.cuda.memory_allocated(DEVICE))
    reserved_after_release = int(torch.cuda.memory_reserved(DEVICE))
    freed_bytes = allocated_after_prefill - allocated_after_release

    replacement_bytes = int(kv_bytes * 0.8)
    replacement_numel = (replacement_bytes + 1) // 2
    replacement = torch.empty(replacement_numel, dtype=DTYPE, device=DEVICE)
    replacement.fill_(1)
    torch.cuda.synchronize(DEVICE)
    replacement_ok = replacement.numel() * replacement.element_size() >= replacement_bytes
    del replacement
    torch.cuda.synchronize(DEVICE)

    restored_legacy, h2d_ms = restore_gpu_cache(cpu_cache)
    restored_cache = DynamicCache.from_legacy_cache(restored_legacy)
    with torch.inference_mode():
        restored = greedy_continue(model, restored_cache, first_token)
    restored_token_ids = restored["token_ids"]
    restored_text = tokenizer.decode(restored_token_ids, skip_special_tokens=False)

    logit_diff = (baseline["first_logits"] - restored["first_logits"]).abs()
    max_abs_logit_diff = float(logit_diff.max().item())
    mean_abs_logit_diff = float(logit_diff.mean().item())
    top1_equal = bool(
        baseline["first_logits"].argmax(dim=-1).item()
        == restored["first_logits"].argmax(dim=-1).item()
    )
    continuation_ok = token_sequences_equal(baseline_token_ids, restored_token_ids)

    row = {
        "model": model_id,
        "context_tokens": int(input_ids.shape[1]),
        "layer_count": len(shapes),
        "kv_tensor_shapes": json.dumps(shapes),
        "kv_size_bytes": kv_bytes,
        "kv_size_mib": kv_bytes / 2**20,
        "d2h_ms": d2h_ms,
        "h2d_ms": h2d_ms,
        "d2h_gbps": (kv_bytes / 1e9) / (d2h_ms / 1000),
        "h2d_gbps": (kv_bytes / 1e9) / (h2d_ms / 1000),
        "allocated_before_prefill_bytes": allocated_before_prefill,
        "allocated_after_prefill_bytes": allocated_after_prefill,
        "allocated_after_release_bytes": allocated_after_release,
        "reserved_after_release_bytes": reserved_after_release,
        "freed_gpu_memory_mib": freed_bytes / 2**20,
        "replacement_bytes": replacement_bytes,
        "memory_reuse": "PASS"
        if freed_bytes >= int(kv_bytes * 0.8) and replacement_ok
        else "FAIL",
        "baseline_first_token_ms": baseline["first_token_decode_ms"],
        "baseline_decode_16_ms": baseline["decode_16_ms"],
        "restored_first_token_ms": restored["first_token_decode_ms"],
        "restored_decode_16_ms": restored["decode_16_ms"],
        "cold_resume_latency_ms": h2d_ms + restored["first_token_decode_ms"],
        "warm_resume_latency_ms": restored["first_token_decode_ms"],
        "baseline_token_ids": json.dumps(baseline_token_ids),
        "restored_token_ids": json.dumps(restored_token_ids),
        "baseline_text": baseline_text,
        "restored_text": restored_text,
        "max_abs_logit_diff": max_abs_logit_diff,
        "mean_abs_logit_diff": mean_abs_logit_diff,
        "top1_equal": top1_equal,
        "continuation_correctness": "PASS" if continuation_ok else "FAIL",
    }

    del restored["cache"], restored_cache, restored_legacy, cpu_cache
    del input_ids, attention_mask, first_token
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    return row


def transfer_estimates_ms(kv_bytes: int) -> dict[str, float]:
    phase0_d2h = TransferModel.from_phase0_direction(ROOT, "gpu0_to_pinned_cpu").piecewise_service_time_s(kv_bytes) * 1000
    phase0_h2d = TransferModel.from_phase0_direction(ROOT, "pinned_cpu_to_gpu0").piecewise_service_time_s(kv_bytes) * 1000
    rows = list(csv.DictReader((ROOT / "results/phase2a/transfer_metrics.csv").open()))
    smallest = next(row for row in rows if int(row["size_bytes"]) == 256 * 2**20)
    scale = kv_bytes / int(smallest["size_bytes"])
    return {
        "phase0_expected_d2h_ms": phase0_d2h,
        "phase0_expected_h2d_ms": phase0_h2d,
        "phase2a_scaled_d2h_ms": float(smallest["d2h_median_ms"]) * scale,
        "phase2a_scaled_h2d_ms": float(smallest["h2d_median_ms"]) * scale,
    }


def median_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    numeric = {
        key
        for key, value in first.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    summary: dict[str, Any] = {}
    for key, value in first.items():
        if key in numeric:
            summary[key] = statistics.median(float(row[key]) for row in rows)
        elif key in {"memory_reuse", "continuation_correctness"}:
            summary[key] = "PASS" if all(row[key] == "PASS" for row in rows) else "FAIL"
        else:
            summary[key] = value
    return summary


def environment(model_id: str) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0]
    return {
        "gpu_index": 0,
        "gpu_name": properties.name,
        "total_gpu_memory_bytes": properties.total_memory,
        "driver_version": driver,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version,
        "model": model_id,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2B requires CUDA")
    torch.cuda.set_device(0)
    torch.manual_seed(20260915)
    RESULTS.mkdir(parents=True, exist_ok=True)

    model_id, model, tokenizer, preferred_error = load_model_and_tokenizer()
    with torch.inference_mode():
        _ = one_full_run(model, tokenizer, model_id)
    gc.collect()
    torch.cuda.synchronize(DEVICE)

    raw_rows = [one_full_run(model, tokenizer, model_id) for _ in range(MEASURED_RUNS)]
    summary = median_row(raw_rows)
    estimates = transfer_estimates_ms(int(summary["kv_size_bytes"]))
    summary.update(estimates)
    summary["d2h_relative_error_vs_phase0"] = abs(summary["d2h_ms"] - estimates["phase0_expected_d2h_ms"]) / estimates["phase0_expected_d2h_ms"]
    summary["h2d_relative_error_vs_phase0"] = abs(summary["h2d_ms"] - estimates["phase0_expected_h2d_ms"]) / estimates["phase0_expected_h2d_ms"]
    summary["d2h_relative_error_vs_phase2a"] = abs(summary["d2h_ms"] - estimates["phase2a_scaled_d2h_ms"]) / estimates["phase2a_scaled_d2h_ms"]
    summary["h2d_relative_error_vs_phase2a"] = abs(summary["h2d_ms"] - estimates["phase2a_scaled_h2d_ms"]) / estimates["phase2a_scaled_h2d_ms"]
    summary["phase2a_consistency"] = "approximately consistent" if max(summary["d2h_relative_error_vs_phase2a"], summary["h2d_relative_error_vs_phase2a"]) <= 0.25 else "materially different"

    write_csv(RESULTS / "raw_runs.csv", raw_rows)
    write_csv(RESULTS / "metrics.csv", [summary])
    (RESULTS / "config.json").write_text(
        json.dumps(
            {
                "base_commit": "11c26582e86d20c7edfa67b775523033b1f69734",
                "preferred_model": PREFERRED_MODEL,
                "fallback_model": FALLBACK_MODEL,
                "selected_model": model_id,
                "preferred_model_load_error": preferred_error,
                "context_tokens": CONTEXT_TOKENS,
                "decode_tokens": DECODE_TOKENS,
                "warmup_runs": WARMUP_RUNS,
                "measured_runs": MEASURED_RUNS,
                "dtype": str(DTYPE),
                "gpu": 0,
                "phase2a_calibration_commit": "11c26582e86d20c7edfa67b775523033b1f69734",
                "no_empty_cache_during_primary_experiment": True,
            },
            indent=2,
        )
        + "\n"
    )
    (RESULTS / "environment.json").write_text(
        json.dumps(environment(model_id), indent=2) + "\n"
    )
    (RESULTS / "commands.sh").write_text(
        "python -m pytest -q tests/test_phase2b.py\n"
        "python experiments/phase2b/run_phase2b.py\n"
    )
    print(f"model={model_id} runs={len(raw_rows)} kv_mib={summary['kv_size_mib']:.2f}")


if __name__ == "__main__":
    main()
