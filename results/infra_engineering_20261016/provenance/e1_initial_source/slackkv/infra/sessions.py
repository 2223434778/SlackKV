"""Client-side sessions; logical closure is not immediate native physical release."""
from dataclasses import dataclass, field
import threading
import uuid

from slackkv.infra.http import post


class SessionError(RuntimeError):pass


@dataclass
class Session:
    session_id: str
    generation: int
    slot: int
    salt: str
    messages: list = field(default_factory=list)
    state: str = 'OPEN'
    pending: bool = False
    started_ns: int | None = None
    done_ns: int | None = None


class SessionPool:
    def __init__(self,config,quotas,transport=post,token_count=None):
        self.config=config;self.quotas=quotas;self.transport=transport;self.token_count=token_count
        self.sessions={};self.generations={};self.lock=threading.Lock();self.epoch=uuid.uuid4().hex

    def open(self,session_id,system='You are a concise helpful assistant.'):
        with self.lock:
            if not isinstance(session_id,str) or not session_id or len(session_id)>128:raise SessionError('invalid session_id')
            if session_id in self.sessions and self.sessions[session_id].state!='CLOSED':raise SessionError('session already exists; close it before a new generation')
            used={s.slot for s in self.sessions.values() if s.state!='CLOSED'}
            free=[i for i in range(self.config.session_slots) if i not in used]
            if not free:raise SessionError('session capacity exhausted; no implicit queue or borrowing')
            generation=self.generations.get(session_id,-1)+1;self.generations[session_id]=generation
            session=Session(session_id,generation,free[0],f'{self.epoch}:{uuid.uuid4().hex}:{generation}',
                            messages=[dict(role='system',content=system)])
            self.sessions[session_id]=session
            return session

    def close(self,session_id):
        with self.lock:
            s=self.sessions[session_id]
            if s.pending:raise SessionError('request in flight; close only after settlement/timeout')
            s.state='CLOSED'
            return dict(status='logical_closed',physical_release='native LRU and in-flight settlement; no ordered native close API')

    def send(self,session_id,user,terminal=False,cancel=None):
        if not isinstance(user,str) or not user:raise SessionError('nonempty user text required')
        with self.lock:
            s=self.sessions[session_id]
            if s.state!='OPEN' or s.pending:raise SessionError('session is closed/failed or already has an in-flight request')
            proposed=s.messages+[dict(role='user',content=user)]
            if self.token_count is not None and self.token_count(proposed)+self.config.max_output_tokens>self.config.max_model_len:
                raise SessionError('context limit exceeded; history unchanged')
            s.pending=True
        params={'slackkv_terminal':dict(independent=True,end_after_request=terminal)}
        if self.config.policy=='fair':params['max_offload_tokens']=self.quotas[s.slot]
        body=dict(model='slackkv',messages=proposed,max_tokens=self.config.max_output_tokens,
            temperature=0,stream=True,stream_options={'include_usage':True},cache_salt=s.salt)
        if self.config.policy!='nohost':body['kv_transfer_params']=params
        client_id=uuid.uuid4().hex
        try:
            result=self.transport(self.config.url+'/v1/chat/completions',body,self.config.request_timeout_seconds,cancel)
            with self.lock:
                s.messages=proposed+[dict(role='assistant',content=result['text'])]
                s.started_ns=s.started_ns or result['issue_ns'];s.done_ns=result['done_ns']
                if terminal:s.state='CLOSED'
            return dict(session_id=session_id,generation=s.generation,slot=s.slot,client_request_id=client_id,
                        messages_submitted=proposed,terminal=terminal,**result)
        except BaseException:
            # History is atomic. The server may still be settling cancellation;
            # require logical close + fresh salted generation before reuse.
            with self.lock:s.state='FAILED'
            raise
        finally:
            with self.lock:s.pending=False
