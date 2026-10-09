# Round 2 preregistration — 2026-10-09

## Second candidate, frozen after the first candidate failed

Continuation credit passed P95 but failed session median (+6.08%); stop it.
Two-call soft working-set epochs are an independent next candidate. Charge each
admitted sequential program its current prompt + declared max output, rounded to
16-token blocks. Charge non-program pending/running demand likewise, from actual
engine request objects. Admit a new program if this working-set estimate fits the
actual pool size minus background demand and null block. Existing-program demand
always has a progress path; an empty epoch set also falls back to real allocator
admission. Estimate is NOT physical free capacity, an atomic reservation or a
guarantee for unseen future context growth. All actual allocation checks remain.

After completion retain only a logical epoch for at most one second of idleness;
expire after two calls, explicit terminal call or cancellation. No physical pin,
lease or speculative restoration. During logical waiting skip new programs to
allow arrived existing-program demand through. Original client timestamps include
all new-program waits in E2E; never defer requests outside measurement. Background
has no cohort gate and all fixed work remains. Scope is sequential programs;
parallel program DAGs are not established. Safety tests cover missing next call,
expiration, cancel, double async allocation and output growth. Same acceptance
gates as above, first 2-GiB control/candidate; if passed, reverse-order pair and
4-GiB pair. If failed, do not tune lifetime or call count. TTL and admission are
existing research concepts; potential contribution needs independent comparison.

Start HEAD e239088b3891396cde654669d74020e8fce82ecd; no upstream increment.
Historical untracked server artifacts remain untouched. GPU0 exclusively; one
run <=1800 seconds; cumulative `*/execution.json` ledger <=4 GPU-hours including
load, compilation, oracles and teardown. Reserve a full 0.5 hours before each run.

First intervention: compiled OnDemand, K2, CPU 8 GiB, Qwen2.5-1.5B FP16,
the frozen 40-request/32-resume BGv2 workload and 8192 background output tokens.
Run 2, 4, 8 GiB GPU KV with allocator sampling enabled. These are capacity
diagnostics, not algorithm speedup or statistics. Each starts its own server.
Save all 40 inputs. For following runs replay the first run's actual inputs;
retain the causal finish-relative pause and fixed model work. This is an explicit
fixed-input resource replay: prior output feedback in the synthetic client is
replaced by captured inputs. Numerical/task equivalence remains separate and
UNKNOWN if hashes differ. Server responses and hashed inputs must reflect what
was actually sent. No run with mismatched prompts is eligible for a matched claim.

Before an algorithm screen freeze its independent variable and acceptance gates:
P95 >=10% reduction, session median no worse, background fixed-work wall and
TPOT <=5% worse, H2D <=5% worse, lifecycle checks passed. A single pair only
warrants expansion, never stability. Use a same-budget, same-execution OnDemand
control, with identical sampling in both arms. Preserve failed runs; never relax
the pause/background correctness gates or count capacity increases as innovation.

First candidate: one-turn continuation credit. Before a cold FCFS head, select
the oldest already-arrived, uncomputed, nonpreempted WAITING resume. One successful
allocation exhausts that cold head's credit across ticks (async restore and later
normal admission cannot spend twice). No allocation-result bypass, extra lease,
future pause information, resource-size inference or content change. On failure
stop as ordinary OnDemand. Do not reorder with physical leases. Cancelled head
state is removed at the next selection. Default disabled.

Run new-source disabled control and enabled candidate at 2 GiB, exact frozen inputs.
If it fails the gates above, stop this candidate without changing the credit count.
If it passes, run candidate/control in reverse order and test 4 GiB for a second
regime. This is an architectural screen, not a claim of novelty. Continuum uses
TTL-pinned priority and program FCFS; this probe asks whether bounded ordering of
actual demand alone can avoid needless eviction/queueing without physical pinning.
General continuity scheduling is existing work. Per-head credit boundedness is
not a proof of every queue interaction or universally bounded latency.

Reproduction: existing server venv Python, repository root:

```sh
python -m experiments.autonomous_capacity.run --name capacity-2 --kv-gib 2 --trace
python -m experiments.autonomous_capacity.run --name capacity-4 --kv-gib 4 --trace --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
python -m experiments.autonomous_capacity.run --name capacity-8 --kv-gib 8 --trace --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
```
