# Phase 3G-CorrectnessDiag

## Objective
Determine whether the earlier C continuation mismatch is isolated KV corruption or batch/scheduler-sensitive greedy decoding.

## Setup
vLLM 0.23.0, pinned source 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665, Qwen2.5-1.5B-Instruct, RTX A6000, 4096-token prefixes, max_tokens=32, temperature=0, ignore_eos=true. Four cases × three fresh server restarts were executed.

## Sequence results
S1 LocalKeep-Solo-C: 1 unique; hashes=cf435ed0b932bca3725234458655236af84c9f1d760badbb25b23df3b0db3a97; first divergence=none
S2 Prefetch-Solo-C: 1 unique; hashes=cf435ed0b932bca3725234458655236af84c9f1d760badbb25b23df3b0db3a97; first divergence=none
S3 LocalKeep-Gated-ABC: C 1 unique; hashes=16e8765313daf52d920de250fecf810c5c7e827895ad9aeb7cea35aacce08b6e; first divergence=26
S4 Prefetch-Gated-ABC: C 1 unique; hashes=16e8765313daf52d920de250fecf810c5c7e827895ad9aeb7cea35aacce08b6e; first divergence=26

S3 gate runs: rep0 pre=1 post=3 gate=PASS; rep1 pre=1 post=3 gate=PASS; rep2 pre=1 post=3 gate=PASS
S4 gate runs: rep0 pre=1 post=3 gate=PASS; rep1 pre=1 post=3 gate=PASS; rep2 pre=1 post=3 gate=PASS

## Required comparisons
S1-vs-S2 C: exact_match=True, matching_prefix=32, first_divergence=, interpretation=same stable branch
S1-vs-S3 C: exact_match=False, matching_prefix=25, first_divergence=26, interpretation=batch/scheduler-sensitive branch
S3-vs-S4 A: exact_match=True, matching_prefix=32, first_divergence=, interpretation=same stable branch
S3-vs-S4 B: exact_match=True, matching_prefix=32, first_divergence=, interpretation=same stable branch
S3-vs-S4 C: exact_match=True, matching_prefix=32, first_divergence=, interpretation=same stable branch

S1 is deterministic across three restarts and S2 is deterministic. S1 and S2 match exactly, so isolated prefetch correctness passes. S3 C differs from solo S1 at token 26 (one-based), while S4 reproduces S3 for A, B, and C under the matched gated schedule. This supports batch/scheduler-sensitive greedy decoding rather than an isolated KV transport error.

## Cache transport
S1/S3 local cases retained local prefixes. S2/S4 offload/prefetch logs show 4096 external tokens loaded, zero native model forwards, zero native generated tokens, and zero mixed prefetch/model steps. Transport validation is reported separately from sequence equality.

## Optional logprobs
Unsupported/not collected in this pinned endpoint; see logprobs.json.

## Interpretation
The Phase 3G blocker is most consistent with BATCH_SENSITIVITY. The matched concurrent LocalKeep and Prefetch cases produce the same continuation branch, while the solo C branch differs at token 26.

## Limitations
This is a correctness diagnostic with three fresh restarts per case, not a performance matrix. max_tokens=32 is used only for diagnosis; no deterministic CUDA mode or teacher-forced scoring was added.

Phase 3G-CorrectnessDiag execution status: COMPLETE
