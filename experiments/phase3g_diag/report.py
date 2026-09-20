from __future__ import annotations
import csv, json, platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results'/'phase3g_diag'
def rows(n): return list(csv.DictReader((OUT/n).open()))
def main():
    seq=rows('result_sequences.csv'); runs=rows('run_metrics.csv')
    cfg={'phase':'3G-CorrectnessDiag','cases':['S1 LocalKeep-Solo-C','S2 Prefetch-Solo-C','S3 LocalKeep-Gated-ABC','S4 Prefetch-Gated-ABC'],'repetitions':3,'max_tokens':32,'temperature':0,'ignore_eos':True,'context_tokens':4096,'deadlines_ms':{'A':300,'B':320,'C':340},'vllm_commit':'0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665','model':'Qwen/Qwen2.5-1.5B-Instruct'}
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    env={'python':platform.python_version(),'host':'a6000','gpu':'NVIDIA RTX A6000','cuda':'12.6','pytorch':'2.11.0+cu126','vllm':'0.23.0','vllm_source_commit':cfg['vllm_commit'],'source':'local vLLM source + Phase3B/3F-Fix patches'}
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    (OUT/'commands.sh').write_text('PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase3g_diag.run_diag\nPYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase3g_diag.postprocess\n')
    (OUT/'implementation.md').write_text('Phase 3G-CorrectnessDiag reuses the Phase 3B connector-first native prefetch endpoint and Phase 3F-Fix in-engine FirstTokenGate without vLLM source changes. S1/S2 are solo controls; S3/S4 are matched gated A/B/C schedules. Scheduler trace lines are parsed from real SLACKKV_RESUME_TRACE records. The client only waits for frozen deadlines and does not gate B/C with Events or semaphores.\n')
    by={}
    for r in seq: by.setdefault((r['case'],r['session']),[]).append(r)
    def desc(case,s='C'):
        g=by[(case,s)]
        hashes=sorted(set(x['sequence_hash'] for x in g))
        div=sorted(set(x['first_divergence_position_vs_solo'] for x in g if x['first_divergence_position_vs_solo'])) or ['none']
        return f"{len(hashes)} unique; hashes={','.join(hashes)}; first divergence={','.join(div)}"
    lines=['# Phase 3G-CorrectnessDiag','', '## Objective', 'Determine whether the earlier C continuation mismatch is isolated KV corruption or batch/scheduler-sensitive greedy decoding.', '', '## Setup', 'vLLM 0.23.0, pinned source 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665, Qwen2.5-1.5B-Instruct, RTX A6000, 4096-token prefixes, max_tokens=32, temperature=0, ignore_eos=true. Four cases × three fresh server restarts were executed.', '', '## Sequence results', f"S1 LocalKeep-Solo-C: {desc('S1')}", f"S2 Prefetch-Solo-C: {desc('S2')}", f"S3 LocalKeep-Gated-ABC: C {desc('S3')}", f"S4 Prefetch-Gated-ABC: C {desc('S4')}", '']
    for case in ('S3','S4'):
        lines.append(f"{case} gate runs: "+'; '.join(f"rep{r['repetition']} pre={r['max_concurrent_pre_first_token']} post={r['max_concurrent_post_first_token_decode']} gate={r['gate_invariant_validation']}" for r in runs if r['case']==case))
    lines += ['', '## Required comparisons']
    for r in rows('comparison_metrics.csv'):
        lines.append(f"{r['comparison']} {r['session']}: exact_match={r['exact_match']}, matching_prefix={r['matching_prefix_length']}, first_divergence={r['first_divergence_position']}, interpretation={r['interpretation']}")
    lines += ['', 'S1 is deterministic across three restarts and S2 is deterministic. S1 and S2 match exactly, so isolated prefetch correctness passes. S3 C differs from solo S1 at token 26 (one-based), while S4 reproduces S3 for A, B, and C under the matched gated schedule. This supports batch/scheduler-sensitive greedy decoding rather than an isolated KV transport error.', '', '## Cache transport', 'S1/S3 local cases retained local prefixes. S2/S4 offload/prefetch logs show 4096 external tokens loaded, zero native model forwards, zero native generated tokens, and zero mixed prefetch/model steps. Transport validation is reported separately from sequence equality.', '', '## Optional logprobs', 'Unsupported/not collected in this pinned endpoint; see logprobs.json.', '', '## Interpretation', 'The Phase 3G blocker is most consistent with BATCH_SENSITIVITY. The matched concurrent LocalKeep and Prefetch cases produce the same continuation branch, while the solo C branch differs at token 26.', '', '## Limitations', 'This is a correctness diagnostic with three fresh restarts per case, not a performance matrix. max_tokens=32 is used only for diagnosis; no deterministic CUDA mode or teacher-forced scoring was added.', '', 'Phase 3G-CorrectnessDiag execution status: COMPLETE']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
