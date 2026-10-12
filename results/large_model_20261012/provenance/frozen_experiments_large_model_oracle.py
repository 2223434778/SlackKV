"""Sensitive text retrieval plus truly independent cold prefix identity."""
import json
import time
import urllib.request


def inputs(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    result = {}
    filler = tokenizer.encode('Tool log: file inspection completed; status ok; no project code in this entry.\n',
                              add_special_tokens=False)
    for label, marker in [('a', 'CEDAR17'), ('b', 'MAPLE42')]:
        messages = [dict(role='system', content='Retrieve the initial project code accurately. Reply only with that code.'),
                    dict(role='user', content=f'The initial project code is {marker}.\n__FILLER__\nWhat is the initial project code?')]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        head, tail = text.split('__FILLER__')
        prefix = tokenizer.encode(head, add_special_tokens=False)
        suffix = tokenizer.encode(tail, add_special_tokens=False)
        needed = 28672 - len(prefix) - len(suffix)
        if needed <= 0:
            raise ValueError('oracle template too long')
        result[label] = dict(marker=marker, prompt=prefix + (filler * (needed//len(filler)+1))[:needed] + suffix)
    return result, tokenizer


def send(runtime, prompt, limit, salt):
    payload = dict(model=str(runtime.MODEL), prompt=prompt, max_tokens=16, temperature=0,
                   ignore_eos=True, stream=True, return_token_ids=True, cache_salt=salt,
                   kv_transfer_params={'max_offload_tokens': limit})
    request = urllib.request.Request(runtime.API, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    issue = time.monotonic_ns()
    tokens, first = [], None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            part = line[6:].strip()
            if part == b'[DONE]':
                continue
            choice = (json.loads(part).get('choices') or [{}])[0]
            ids = choice.get('token_ids') or []
            if ids and first is None:
                first = time.monotonic_ns()
            tokens.extend(map(int, ids))
    if first is None:
        raise ValueError('oracle response has no tokens')
    return dict(issue_ns=issue, first_ns=first, done_ns=time.monotonic_ns(), tokens=tokens)
