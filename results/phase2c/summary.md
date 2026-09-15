# Phase 2C — vLLM selective-offload capability gate

## Execution status

Phase 2C execution status: **INCOMPLETE**.

The required A/B experiment was not started.  The comparison requires Session B to
send `kv_transfer_params={"max_offload_tokens": 0}` so that it is demonstrably
kept local.  Running without that per-request control would not test selective
offload and would make cache/TTFT attribution invalid.

## Environment and capability evidence

The isolated compatible runtime imports vLLM 0.11.2 on Python 3.11 with the host
CUDA 12.4 PyTorch stack.  It contains `OffloadingConnector` but no
`max_offload_tokens` implementation anywhere in its installed source.  vLLM
0.29.0 was separately probed because it contains the required per-request cap,
but it cannot import on this host: its wheel requires `libcudart.so.13`, while
the installed CUDA runtime is 12.4.  The machine therefore has no tested runtime
that can both import and enforce the required A/B admission distinction.

`environment.json` and `server_logs/environment_probe.txt` preserve the
machine-readable capability result.  The source harness exits before server
startup when this gate fails.  `resume_metrics.csv` deliberately records `NA`
instead of fabricated TTFT, cache-hit, or GPU-capacity values.

## Validation

`python -m pytest -q tests/test_phase2c.py` passes four pure-harness checks:
exact 4096-token construction, distinct pressure context, Prometheus counter
delta parsing, and streaming TTFT calculation.  No vLLM request, GPU-pressure
eviction validation, generation-validity comparison, or selective-offload
validation was run.

## Result

The Phase 2C A/B/C runtime conclusion is **not evaluated**.  For the current
deployment capability gate, this is Result C: the required selective-offload
control cannot be enforced.  It is an integration-environment blocker, not a
negative result for SlackKV's offload mechanism.

## Reproduction

Run the test command and then
`/home/a6000/.venvs/slackkv-phase2c/bin/python experiments/phase2c/run_phase2c.py`.
The latter is expected to exit nonzero until a CUDA-compatible vLLM release with
both `OffloadingConnector` and `max_offload_tokens` is available.
