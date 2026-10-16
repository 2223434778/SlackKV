"""Attach only to our live owner; test real disconnect without fake close inference."""
import json
from pathlib import Path
import threading
import time

from slackkv.infra.config import layout
from slackkv.infra.http import IncompleteResponse,get,post
from slackkv.infra.report import counter
from slackkv.infra.runtime import save
from slackkv.infra.service import birth
from slackkv.infra.sessions import SessionPool
from slackkv.infra.workloads import chat_tokens,tokenizer


def run(config,out):
    out=Path(out);execution=json.loads((out/'execution.json').read_text(encoding='utf-8'))
    if execution['status']!='RUNNING' or execution['config']!=config.snapshot() or birth(execution['parent_pid'])!=execution['parent_birth']:
        raise ValueError('reliability exercise requires matching live owned service')
    if (out/'reliability.json').exists():raise FileExistsError('exercise already recorded')
    cancel=threading.Event();observed=[];tok=tokenizer(config)
    def transport(url,body,timeout,event):
        return post(url,body,timeout,event,on_first=lambda info:(observed.append(info),cancel.set()))
    pool=SessionPool(config,layout(config)['quota_tokens'],transport=transport,
        token_count=lambda messages:len(chat_tokens(tok,messages)))
    old=pool.open('disconnect')
    try:pool.send('disconnect','Please count from one to one hundred, with one number on each line.',cancel=cancel)
    except IncompleteResponse:pass
    else:raise AssertionError('disconnect did not interrupt client stream')
    if not observed or len(old.messages)!=1 or old.state!='FAILED' or old.pending:
        raise AssertionError('failed stream committed history or retained client pending state')
    close=pool.close('disconnect');new=pool.open('disconnect');pool.transport=post
    followup=pool.send('disconnect','Say READY.',terminal=True)
    if new.generation!=old.generation+1 or new.salt==old.salt:raise AssertionError('generation identity not renewed')
    deadline=time.monotonic()+10
    while True:
        metrics=get(config.url+'/metrics')
        running=counter(metrics,'vllm:num_requests_running');waiting=counter(metrics,'vllm:num_requests_waiting')
        if running==waiting==0:break
        if time.monotonic()>=deadline:raise RuntimeError('scheduler idle gauges unconfirmed after disconnect/followup')
        time.sleep(.1)
    record=dict(status='PASS',cancel_after_first_content=observed,failed_history_messages=len(old.messages),
        old_generation=old.generation,new_generation=new.generation,close=close,followup=followup,
        observed_running=running,observed_waiting=waiting,
        limit='client transport really closed; server abort vs natural finish UNKNOWN; idle observed, no claim of immediate physical KV release')
    save(out/'reliability.json',record)
    return record
