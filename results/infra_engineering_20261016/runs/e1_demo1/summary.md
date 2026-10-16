# SlackKV real-chat demo

- Requests: 6; sessions: 2; overlapping requests: True.
- Client-driven local JSON tool; actual assistant history retained.
- Missing values: UNKNOWN. Global counters do not attribute concurrent requests.
- Free generation demo is not a fair throughput benchmark.

```json
{
  "status": "COMPLETE",
  "requests": 6,
  "sessions": 2,
  "cross_session_request_overlap": true,
  "session_e2e_ms": {
    "cedar": 1196.464372,
    "maple": 1186.219097
  },
  "prompt_sources": {
    "local_compute": 291.0,
    "local_cache_hit": 352.0,
    "external_kv_transfer": 0.0
  },
  "transfer_bytes": {
    "GPU_to_CPU": null,
    "CPU_to_GPU": null
  },
  "missing": "null means UNKNOWN; absent transfer labels are not filled with zero",
  "attribution": "global counters over this demo; no per-request attribution during overlap",
  "timing": "client first nonempty SSE content; interval proxy is not a TPOT distribution",
  "physical_host_occupancy": "UNKNOWN; configured capacity is not actual occupancy",
  "comparison": "free generation histories can differ by policy; not a strict throughput ranking"
}
```
