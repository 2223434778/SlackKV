# Frozen-work engineering smoke

```json
{
  "status": "PASS",
  "scenario": "host-enough",
  "policy": "fair",
  "source_tokens": {
    "local_compute": 49408.0,
    "local_cache_hit": 0.0,
    "external_kv_transfer": 32896.0
  },
  "actual_gpu_tokens": 18720,
  "request_metrics": [
    {
      "label": "cedar0",
      "ttft_first_content_ms": 319.182973,
      "response_ms": 710.745393,
      "first_content_to_end_mean_ms": 26.104161333333334
    },
    {
      "label": "maple0",
      "ttft_first_content_ms": 630.493312,
      "response_ms": 720.443907,
      "first_content_to_end_mean_ms": 5.996706333333333
    },
    {
      "label": "pressure0",
      "ttft_first_content_ms": 746.575458,
      "response_ms": 835.905567,
      "first_content_to_end_mean_ms": 5.9553406
    },
    {
      "label": "cedar1",
      "ttft_first_content_ms": 62.129886,
      "response_ms": 157.948061,
      "first_content_to_end_mean_ms": 6.387878333333333
    },
    {
      "label": "maple1",
      "ttft_first_content_ms": 77.289649,
      "response_ms": 167.416189,
      "first_content_to_end_mean_ms": 6.008436
    },
    {
      "label": "pressure1",
      "ttft_first_content_ms": 749.707062,
      "response_ms": 838.357031,
      "first_content_to_end_mean_ms": 5.909997933333333
    },
    {
      "label": "cedar2",
      "ttft_first_content_ms": 63.03288,
      "response_ms": 160.004087,
      "first_content_to_end_mean_ms": 6.464747133333334
    },
    {
      "label": "maple2",
      "ttft_first_content_ms": 79.262413,
      "response_ms": 169.798114,
      "first_content_to_end_mean_ms": 6.0357134
    }
  ],
  "transfer_bytes": {
    "GPU_to_CPU": 473432064.0,
    "CPU_to_GPU": 943194112.0
  },
  "session_e2e_ms": {
    "cedar": 2744.104302,
    "maple": 2753.701263
  },
  "cross_session_overlap": true,
  "total_requests": 8,
  "input_tokens": 82304,
  "output_tokens": 128,
  "sequence_ms": 2755.488642,
  "prompt_identity": {
    "cedar0": "788f85dd91bffc94999de65be99ee35c0a2dfa73aa0d07c457c1e01ce3468c29",
    "maple0": "9a639d5c435e95e44165f2406400e2b803b5d2e272a438cc3cd985748fd0ed29",
    "pressure0": "1215225a8230c477f95cd76479c584ab6bae072b35af519d202e12d361d8baa1",
    "cedar1": "11c60d752392f516fd8b62ae3ce9f75abd3f2291586617c6ee0c1d84cfd11c6e",
    "maple1": "02eb071654df7f1b78bd481fd39196b24010e5803e4704857b47072b4d3ebbbd",
    "pressure1": "1215225a8230c477f95cd76479c584ab6bae072b35af519d202e12d361d8baa1",
    "cedar2": "72fb5525137118a6392052ef8c1bf52fcf8de92527ad1f223b195e0d91b82e99",
    "maple2": "5a8a085b2342736f84cfaae21394875dab60da9efc49e21135246c8f05956146"
  },
  "qualification": "one frozen-work engineering smoke, no statistical speedup or task-quality claim; Full/Fair same connector, NoHost absent connector; null traffic UNKNOWN, NoHost0 by config"
}
```
