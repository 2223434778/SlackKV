# Frozen-work engineering smoke

```json
{
  "status": "PASS",
  "scenario": "hot",
  "policy": "nohost",
  "source_tokens": {
    "local_compute": 2304.0,
    "local_cache_hit": 4224.0,
    "external_kv_transfer": 0.0
  },
  "actual_gpu_tokens": 18720,
  "request_metrics": [
    {
      "label": "cedar0",
      "ttft_first_content_ms": 160.402407,
      "response_ms": 263.075259,
      "first_content_to_end_mean_ms": 6.8448568
    },
    {
      "label": "maple0",
      "ttft_first_content_ms": 178.628777,
      "response_ms": 267.179284,
      "first_content_to_end_mean_ms": 5.903367133333333
    },
    {
      "label": "cedar1",
      "ttft_first_content_ms": 33.469946,
      "response_ms": 117.132566,
      "first_content_to_end_mean_ms": 5.577508
    },
    {
      "label": "maple1",
      "ttft_first_content_ms": 34.271578,
      "response_ms": 121.255222,
      "first_content_to_end_mean_ms": 5.7989096
    },
    {
      "label": "cedar2",
      "ttft_first_content_ms": 35.593096,
      "response_ms": 119.473052,
      "first_content_to_end_mean_ms": 5.591997066666667
    },
    {
      "label": "maple2",
      "ttft_first_content_ms": 36.693716,
      "response_ms": 123.627958,
      "first_content_to_end_mean_ms": 5.795616133333334
    }
  ],
  "transfer_bytes": {
    "GPU_to_CPU": 0,
    "CPU_to_GPU": 0
  },
  "session_e2e_ms": {
    "cedar": 512.578778,
    "maple": 516.55628
  },
  "cross_session_overlap": true,
  "total_requests": 6,
  "input_tokens": 6528,
  "output_tokens": 96,
  "sequence_ms": 517.361182,
  "prompt_identity": {
    "cedar0": "a628ec5741ebc17f6040a755731cfe91c35ef645ae086591b17865e33ccfccd2",
    "maple0": "f74bbcf5b6bdceb1828e70dda1781742403a7fb6d9f9c3676751c306d4b84a13",
    "cedar1": "f1a664223f1cd8133283a43a652ad8ed11068328ade2016fb449d4a7c7de302c",
    "maple1": "5d26dec8384012d106ac3cbd97ada7ab63558d205cadd1f15ea12d4be14c8b27",
    "cedar2": "265430ee4508e6671244c01d7e42a31365d69688837429d432e981eb49e2028c",
    "maple2": "d6e8a919dc3f97dd6d4f0068b182cd814c3203b5626522bdd11fbad871bcdf3b"
  },
  "qualification": "one frozen-work engineering smoke, no statistical speedup or task-quality claim; Full/Fair same connector, NoHost absent connector; null traffic UNKNOWN, NoHost0 by config"
}
```
