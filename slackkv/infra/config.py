"""Explicit fixed slots, TP1/full-attention only, loopback service."""
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model: str
    runtime: str
    policy: str = 'fair'
    gpu: int = 0
    host: str = '127.0.0.1'
    port: int = 8000
    gpu_kv_bytes: int = 536870912
    host_kv_bytes: int = 234881024
    max_model_len: int = 18432
    session_slots: int = 2
    max_output_tokens: int = 128
    request_timeout_seconds: int = 120
    ready_timeout_seconds: int = 240
    max_run_seconds: int = 600
    batch_invariant: bool = False

    def __post_init__(self):
        if self.policy not in ('fair','full','nohost'):raise ValueError('policy must be fair, full or nohost')
        if self.host != '127.0.0.1':raise ValueError('E1 supports loopback127.0.0.1 only; no authentication gateway')
        for name in ('port','gpu_kv_bytes','host_kv_bytes','max_model_len','session_slots',
                     'max_output_tokens','request_timeout_seconds','ready_timeout_seconds','max_run_seconds'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise ValueError(name+' must be positive int')
        if type(self.gpu) is not int or self.gpu<0:raise ValueError('gpu must be a nonnegative index')
        if self.port>65535 or self.session_slots>64:raise ValueError('port/slot limit exceeded')
        if self.max_output_tokens>=self.max_model_len:raise ValueError('output exceeds context')
        if not 60<=self.max_run_seconds<=1800:raise ValueError('run limit must be60..1800 seconds including cleanup')
        if type(self.batch_invariant) is not bool:raise ValueError('batch_invariant must be boolean')
        if not all(isinstance(x,str) and x for x in (self.model,self.runtime)):raise ValueError('model and runtime required')

    @classmethod
    def load(cls,path,policy=None):
        raw=json.loads(Path(path).read_text(encoding='utf-8'))
        unknown=raw.keys()-{f.name for f in fields(cls)}
        if unknown:raise ValueError('unknown configuration keys: '+','.join(sorted(unknown)))
        for key in ('model','runtime'):
            if key in raw:raw[key]=os.path.expandvars(raw[key])
        if policy is not None:raw['policy']=policy
        return cls(**raw)

    @property
    def url(self):return f'http://{self.host}:{self.port}'

    def snapshot(self):return asdict(self)


def layout(config):
    model=Path(config.model)
    if '$' in config.model or not model.is_dir():raise ValueError('set SLACKKV_MODEL to an existing local model; no automatic download')
    shape=json.loads((model/'config.json').read_text(encoding='utf-8'))
    if shape.get('use_sliding_window') or shape.get('sliding_window') is not None:
        if shape.get('use_sliding_window') is not False:raise ValueError('only full attention supported')
    if shape.get('model_type') not in ('qwen2',):raise ValueError('tested adapter supports Qwen2 full-attention layout only')
    if config.max_model_len>shape['max_position_embeddings']:raise ValueError('context exceeds model config')
    token_bytes=2*2*shape['num_hidden_layers']*shape['num_key_value_heads']*(shape['hidden_size']//shape['num_attention_heads'])
    if config.max_model_len*token_bytes>config.gpu_kv_bytes:raise ValueError('one full context cannot fit GPU KV budget')
    blocks=config.host_kv_bytes//(16*token_bytes)
    if blocks<config.session_slots:raise ValueError('host pool cannot provide one aligned block per slot')
    per,remainder=divmod(blocks,config.session_slots)
    quotas=[16*(per+(i<remainder)) for i in range(config.session_slots)]
    return dict(kv_bytes_per_token=token_bytes,host_blocks=blocks,quota_tokens=quotas,
        unused_host_bytes=config.host_kv_bytes-blocks*16*token_bytes,
        sharing='cache_salt isolates sessions/generations; no cross-session shared-root discount',
        idle_quota='unused slots and short histories do not lend quota; exhaustion rejects immediately')
