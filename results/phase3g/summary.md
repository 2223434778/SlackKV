# Phase 3G — STOPPED at integration gate

## Configuration

Phase 3G used SlackKV base `c6ffb80b90b8d363dfa62f15dcbc5a1b145d9de5`, the
pinned vLLM source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, the unchanged
Phase 3F-Fix in-engine gate, A/B/C-only integration, 4096-token prefixes,
oracle deadlines 300/320/340 ms, `max_tokens=32`, `temperature=0`, and
`ignore_eos=true`. Six controls each generated exactly 32 tokens.

## Integration evidence

The real scheduler trace passed the admission mechanism checks:

| session | admitted step | first-token step | finished step | full sequence |
|---|---:|---:|---:|---|
| A | 71 | 72 | 103 | PASS |
| B | 73 | 74 | 105 | PASS |
| C | 75 | 76 | 107 | **FAIL** |

Maximum admitted-but-no-first-token gate ownership was 1. A→B→C ordering
held. B was admitted after A's first token and before A finished; C was
admitted after B's first token and before B finished. Post-first-token decode
therefore overlapped. Native promotion had zero model forwards/generated
tokens and the cache/offload path itself completed.

The C measured 32-token sequence diverged from its deterministic control
sequence, despite having exactly 32 generated tokens. The first 24 tokens
matched and the remaining continuation differed. Since Phase 3G requires exact
full-sequence equality, this is a hard correctness failure, not a performance
result.

## Decision

Per the execution order, the experiment stopped immediately after the failed
A/B/C integration gate. The nine-run policy matrix was **not** run, so no
OnDemand/ResumeAware/FirstTokenGate performance comparison is reported and no
Result A/B/C mechanism claim is made.

Final classification: **Result D — invalid implementation/measurement**.

The raw monotonic `t0`-aligned trace, control tokens, integration artifact,
and diagnostic tests are preserved in this directory. The most likely issue
to audit next is multi-token continuation correctness under concurrent local
KV resumes; Phase 3F-Fix's one-token checks did not expose it.
