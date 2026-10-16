"""Small native spec: terminal suppresses optional writes, never bypasses reads."""
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.base import PrepareStoreOutput
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from slackkv.m1.native import terminal_independent


class TerminalManager(CPUOffloadingManager):
    def prepare_store(self,keys,ctx):
        if terminal_independent(ctx):
            return PrepareStoreOutput([],self._get_load_store_spec([],[]),[])
        return super().prepare_store(keys,ctx)


class InfraCPUSpec(CPUOffloadingSpec):
    def __init__(self,vllm_config,kv_cache_config):
        super().__init__(vllm_config,kv_cache_config)
        p=vllm_config.parallel_config;groups=kv_cache_config.kv_cache_groups
        if p.tensor_parallel_size!=1 or p.pipeline_parallel_size!=1 or p.decode_context_parallel_size!=1 or p.prefill_context_parallel_size!=1:
            raise ValueError('infra0.1 validates TP/PP/DCP/PCP=1 only')
        if len(groups)!=1 or not isinstance(groups[0].kv_cache_spec,FullAttentionSpec) or groups[0].is_eagle_group:
            raise ValueError('infra0.1 requires one full-attention group')
        s=groups[0].kv_cache_spec
        if s.sliding_window is not None or s.attention_chunk_size is not None or self.eviction_policy!='lru':
            raise ValueError('unsupported attention/policy layout')

    def get_manager(self):
        if self._manager is None:
            events=self.vllm_config.kv_events_config
            self._manager=TerminalManager(self.num_blocks,cache_policy=self.eviction_policy,
                enable_events=events is not None and events.enable_kv_cache_events)
        return self._manager
