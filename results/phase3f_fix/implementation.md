# Phase 3F-Fix implementation

The measured gate is implemented inside the pinned vLLM scheduler, not in the
client. The Phase 3B connector-first implementation is reused unchanged for
native prefetch requests.

Pinned vLLM source: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.

## Modified scheduler functions

- `Scheduler.__init__`: adds `_slackkv_ft_gate_owner`, first-model/first-token
  bookkeeping, blocked-step counters, and the monotonic scheduler step ID.
- `Scheduler.add_request`: records `RESUME_ARRIVED` for marked requests.
- `Scheduler.schedule`: increments the real step counter, skips only marked
  gated resumes while an owner exists, leaves unmarked traffic schedulable,
  assigns ownership only when a request is actually admitted, and records
  `GATE_BLOCKED`, `RESUME_ADMITTED`, and `FIRST_MODEL_SCHEDULED`.
- `Scheduler.update_from_output`: detects the first actual generated token from
  model output and releases the owner only then; no client event or timer is
  involved.
- remote-KV promotion and cleanup paths: record
  `WAITING_FOR_REMOTE_KVS`, `REMOTE_KV_READY`, `REQUEST_FINISHED`, and release
  stale ownership on abnormal termination.

The gate marker is carried in research-only `kv_transfer_params` fields
`slackkv_trace_resume`, `slackkv_session`, and
`slackkv_first_token_gate`. It does not alter cache identity or native
prefetch-only requests.

## State flow

```text
marked request arrives
        |
        v
WAITING -- owner busy --> GATE_BLOCKED (remains waiting)
   |
   +-- owner empty --> RESUME_ADMITTED / owner=request
                              |
                              v
                    actual model output token
                              |
                              v
                         FIRST_TOKEN
                              |
                              v
                         owner cleared
```

The scheduler uses the existing skipped-waiting queue so ordinary unmarked
requests continue to be considered. No CUDA kernel is preempted and no
`max_num_seqs` global restriction is introduced. With `max_tokens=1`, first
token release is approximately request completion release in this workload;
multi-token decode overlap is intentionally not evaluated here.

The exact source diff is preserved in `vllm_phase3f_fix.patch`. The patch is
relative to the currently patched Phase 3B source tree and includes only the
research instrumentation/gate delta on top of that tree.
