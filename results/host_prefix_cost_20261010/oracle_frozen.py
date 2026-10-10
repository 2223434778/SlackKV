"""Batch-invariant partial host restore versus independent salted cold prefill."""
import json
import time
import urllib.request
from experiments.host_prefix.checkpoint import FRONTIER


def run(out, runtime):
    from experiments.autonomous_fork.run import fixed_tokens
    root = fixed_tokens(28672, 773311)
    prompt = root + fixed_tokens(64, 1200)
    runtime.send(root, 16, stream=True, params={'max_offload_tokens': FRONTIER})
    time.sleep(2)
    reset = urllib.request.Request('http://127.0.0.1:8000/reset_prefix_cache?reset_external=false', data=b'', method='POST')
    with urllib.request.urlopen(reset, timeout=20) as r:
        assert r.status == 200
    before = runtime.metric_text()
    restored = runtime.send(prompt, 16, stream=True, params={'max_offload_tokens': FRONTIER})
    time.sleep(2)
    after = runtime.metric_text()
    cold = runtime.send(prompt, 16, stream=True, params={'max_offload_tokens': 0, 'cache_salt': 'host-prefix-independent-cold'})
    time.sleep(2)
    final = runtime.metric_text()
    for name, raw in (('before.prom', before), ('after_restore.prom', after), ('after_cold.prom', final)):
        (out / name).write_text(raw)
    result = dict(prompt=prompt, restored=restored, cold=cold,
                  token_match=restored['tokens'] == cold['tokens'], expected_external_tokens=FRONTIER,
                  qualification='one partial restore input; not all cancellation/preemption/quality proof')
    (out / 'oracle.json').write_text(json.dumps(result, indent=2) + '\n')
    if len(restored['tokens']) != 16 or not result['token_match']:
        raise RuntimeError('partial restore oracle failed')
