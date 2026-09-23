# Phase 3V — Dynamic block reservation

## Objective and setup

This phase compares count-based two-slot reservation with a logical 512-block admission budget under three clusters of heterogeneous KV sizes, the frozen A–L deadlines, K=2 resume admission, and the 9500-token / 256-output-token background X. Runtime is vLLM 0.23.0 on RTX A6000, Qwen2.5-1.5B-Instruct FP16, GPU KV 1 GiB and CPU tier 2 GiB. X slowdown uses the median of the three fresh X-only controls (3683.740 ms decode duration).

The workload comprises 12 unique 1024/4096-token prompts (21,504 reusable prefix tokens total). Each cluster contains three 1024-token prefixes and one 4096-token prefix, requiring 448 blocks. The theoretical minimum H2D traffic is 588 MiB.

## Comparison

Medians across the three measured runs per policy:

| Policy | Useful promotions/run | Local source % | Peak reserved blocks | Budget use | AL P95 resume (ms) | Total H2D (MiB) | Duplicate H2D (MiB) | X slowdown | Combined tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-K2 | 0 | 0.00 | 0 | 0.0% | 107.75 | 588.00 | 0.00 | 1.047 | 162.59 |
| NaiveAsync40-K2 | 12 | 100.00 | 0 | 0.0% | 56.38 | 588.00 | 0.00 | 1.052 | 161.74 |
| FixedCount2-Async40-K2 | 6 | 28.57 | 128 | 25.0% | 79.99 | 588.00 | 0.00 | 1.031 | 165.12 |
| DynamicBlock512-Async40-K2 | 12 | 100.00 | 448 | 87.5% | 56.49 | 588.00 | 0.00 | 1.063 | 160.15 |

## Findings

- Uniform 4096-token regression: FixedCount2 promoted/skipped 6/6; DynamicBlock promoted/skipped 6/6. Both had 100% useful promotions, zero promoted remote waits, zero lease leaks, and a 512-block peak: PASS=True.
- Heterogeneous integration gate: 4/4 runs completed; gate PASS=True. Measured phase: 12/12 runs; exact 32-token request validation is 144/144.
- FixedCount2 promoted 6 and skipped 6 per run, using 128/512 blocks (25.0%). DynamicBlock promoted 12 and skipped 0, using 448/512 (87.5%). It admitted four jobs in a cluster because 64+64+64+256=448 blocks.
- Predicted versus physical lease blocks: DynamicBlock errors total 0 across measured runs; dynamic peak physical leased blocks median 448, maximum 448. FixedCount2 and DynamicBlock lease leaks total 0 and 0 respectively.
- Promotion usefulness: NaiveAsync 1.0; FixedCount2 1.0; DynamicBlock 1.0. Promoted remote waits totals were FixedCount2 0 and DynamicBlock 0.
- H2D accounting: OnDemand total/duplicate 588.00/0.00 MiB; NaiveAsync 588.00/0.00; FixedCount2 588.00/0.00; DynamicBlock 588.00/0.00. NaiveAsync's eviction/wasted-traffic weakness was not observed: duplicate H2D was 0.00 MiB in all measured repetitions.
- DynamicBlock versus FixedCount2: P50 resume change -24.00%; P95 change -29.38%; combined throughput change -3.01%; X slowdown difference +0.032; useful promotions/run 12 vs 6.
- Async40 service check: 0/36 DynamicBlock issue-to-T6 measurements exceeded 40 ms. Max measured issue-to-T6 was 34.02 ms. Max outstanding HTTP prefetch calls was 2; max internal T0–T6 overlap was 1.
- Source accounting totals per run: expected 21,504 tokens. FixedCount2 local/external = 6144/15360; DynamicBlock = 21504/0.
- X slowdown medians: OnDemand 1.047, FixedCount2 1.031, DynamicBlock 1.063; combined decode throughput medians were 162.59, 165.12, and 160.15 tok/s respectively.

## Interpretation and limitations

DynamicBlock's decisions are derived from EDF QueueJIT order, live logical reservations, and `ceil(prefix_tokens/16)` only. It uses no Prometheus or instantaneous free-memory oracle. The fixed-count baseline records block demand but admits based on an active-session limit of two. Both reservation policies use the unchanged Phase 3S physical lease.

The request's arithmetic line `448 + 64 > 512` is inconsistent with its stated `<= 512` admission rule: 448 + 64 equals 512. The implementation follows the explicit `<= 512` rule; the frozen workload has no fifth overlapping 64-block job in that cluster.

The earlier TraceLab-selection Phase 3V artifacts were retained under `results/phase3v_tracelab_incomplete/`; this phase does not overwrite them. No Phase 3W or prediction uncertainty was run.

## Result

**RESULT_A** — next direction: `DYNAMIC_BLOCK_RESERVATION_VALIDATED`.
