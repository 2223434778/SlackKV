"""Load through the existing native OffloadingSpecFactory configuration."""
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from slackkv.m1.native import M1CPUManager, TerminalCPUManager


class TerminalCPUOffloadingSpec(CPUOffloadingSpec):
    def get_manager(self):
        if self._manager is None:
            events = self.vllm_config.kv_events_config
            self._manager = TerminalCPUManager(self.num_blocks, cache_policy=self.eviction_policy,
                enable_events=events is not None and events.enable_kv_cache_events)
        return self._manager


class SlackKVM1Spec(CPUOffloadingSpec):
    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)
        self.enabled = self.extra_config.get('slackkv_m1_enabled', True)
        if type(self.enabled) is not bool:
            raise ValueError('slackkv_m1_enabled must be a boolean')
        if self.enabled:
            parallel = vllm_config.parallel_config
            if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1 or getattr(parallel, 'pipeline_parallel_size', 1) != 1:
                raise ValueError('M1 validates tensor parallelism only, without context/pipeline parallelism')
            groups = kv_cache_config.kv_cache_groups
            if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec) or groups[0].is_eagle_group:
                raise ValueError('M1 supports a single full-attention group')
            spec = groups[0].kv_cache_spec
            if spec.sliding_window is not None or spec.attention_chunk_size is not None:
                raise ValueError('M1 does not support sliding/chunked attention')
            if self.eviction_policy != 'lru' or int(self.extra_config.get('store_threshold', 0)) != 0:
                raise ValueError('M1 requires LRU and no additional store-threshold filter')

    def get_manager(self):
        if not self.enabled:
            return super().get_manager()
        if self._manager is None:
            events = self.vllm_config.kv_events_config
            trace = self.extra_config.get('slackkv_m1_trace', False)
            if type(trace) is not bool:
                raise ValueError('slackkv_m1_trace must be boolean')
            terminal = self.extra_config.get('slackkv_terminal_store_gate', False)
            if type(terminal) is not bool:
                raise ValueError('slackkv_terminal_store_gate must be boolean')
            self._manager = M1CPUManager(self.num_blocks, self.gpu_block_size[0]*self.block_size_factor,
                                        self.kv_bytes_per_offloaded_block,
                                        enable_events=events is not None and events.enable_kv_cache_events, trace=trace,
                                        terminal_store_gate=terminal)
        return self._manager
