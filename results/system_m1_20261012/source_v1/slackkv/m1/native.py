"""Opt-in native CPU manager. Metadata only here; original workers do all DMA."""
from collections import Counter
from threading import RLock, get_ident

from vllm.v1.kv_offload.base import OffloadingEvent, PrepareStoreOutput
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from slackkv.m1.controller import PrefixController


class M1CPUManager(CPUOffloadingManager):
    def __init__(self, num_blocks: int, block_tokens: int, block_bytes: int, enable_events: bool = False):
        super().__init__(num_blocks, cache_policy='lru', enable_events=enable_events)
        if block_tokens <= 0 or block_bytes <= 0:
            raise ValueError('positive native block dimensions required')
        self.block_tokens, self.block_bytes = block_tokens, block_bytes
        self.controller = PrefixController(num_blocks)
        self._lock = RLock()
        self._owner_thread = get_ident()
        self._contexts = {}
        self._serial = 0
        self._writes = {}
        self._reads = Counter()
        self._finished = set()
        self.metrics = Counter()

    def on_new_request(self, ctx):
        with self._lock:
            if id(ctx) not in self._contexts:
                self._serial += 1
                self._contexts[id(ctx)] = (ctx, f'{ctx.req_id}@{self._serial}')
            return super().on_new_request(ctx)

    def slackkv_m1_observe(self, state):
        """Called by two opt-in scheduler hooks; never changes a GPU block table."""
        with self._lock:
            ctx = state.req_context
            self.on_new_request(ctx)
            handle = self._contexts[id(ctx)][1]
            if handle in self.controller.plans:
                return
            groups = state.config.kv_group_configs
            if len(groups) != 1 or groups[0].sliding_window_size_in_blocks is not None:
                raise ValueError('M1 requires one full-attention group')
            count = state.req.num_prompt_tokens // self.block_tokens
            path = tuple(state.group_states[0].offload_keys[:count])
            meta = (ctx.kv_transfer_params or {}).get('slackkv_m1', {})
            if not isinstance(meta, dict):
                meta = {}
            if len(path) != count:
                meta = {}  # Unknown native identity: suppress optional writes.
            plan = self.controller.observe(handle, meta.get('session_id'), meta.get('generation'), path)
            cap = plan.ceiling * self.block_tokens
            if state.max_offload_tokens is not None:
                cap = min(cap, state.max_offload_tokens)
            state.max_offload_tokens = cap
            self.metrics['observed_requests'] += 1
            if plan.reason:
                self.metrics['fallback_' + plan.reason] += 1
            self.reclaim_tails()

    def _plan(self, ctx):
        entry = self._contexts.get(id(ctx))
        return self.controller.plans.get(entry[1]) if entry else None

    def _empty(self):
        return PrepareStoreOutput([], self._get_load_store_spec([], []), [])

    def reclaim_tails(self, protected=()):
        """Manager-owned atomic policy/free-list/event update, deepest first."""
        with self._lock:
            protected = set(protected)
            removed = []
            for token, member in list(self.controller.members.items()):
                owned = [k for k in member.path if self.controller.key_owner.get(k) == token]
                depth = 0
                for key in owned:
                    if self._policy.get(key) is None:
                        break
                    depth += 1
                # A failed earlier write can leave in-flight orphan descendants.
                keep = min(member.target, depth)
                for key in reversed(owned[keep:]):
                    block = self._policy.get(key)
                    if block is None:
                        continue
                    if block.ref_cnt != 0 or key in protected:
                        break
                    self._policy.remove(key)
                    self._free_block(block)
                    removed.append(key)
                if not member.alive and not member.active and not any(self._policy.get(k) for k in owned):
                    self.controller.collect(token)
            if removed and self.events is not None:
                self.events.append(OffloadingEvent(removed, self.medium, removed=True))
            self.metrics['reclaimed_blocks'] += len(removed)
            return removed

    def prepare_store(self, keys, ctx):
        with self._lock:
            keys = list(keys)
            plan = self._plan(ctx)
            if plan is None or not (bound := self.controller.allowed(plan)):
                self.metrics['denied_store_batches'] += 1
                return self._empty()
            removed = self.reclaim_tails(protected=keys)
            existing = []
            new = []
            requested = set(keys)
            # Prefix order comes from native hashes, never client-provided keys.
            for key in plan.path[:bound]:
                block = self._policy.get(key)
                if block is not None:
                    if key in requested:
                        existing.append(key)
                    continue
                if key not in requested or len(new) >= self._get_num_free_blocks():
                    break
                new.append(key)
            result = super().prepare_store(existing + new, ctx)
            if result is None or result.evicted_keys:
                raise RuntimeError('M1 physical gate failed: native LRU eviction must not run')
            result.evicted_keys.extend(removed)
            if result.keys_to_store:
                ticket = (id(ctx), frozenset(result.keys_to_store))
                self._writes[ticket] = ctx
            self.metrics['admitted_store_blocks'] += len(result.keys_to_store)
            self.metrics['skipped_store_blocks'] += len(requested - set(existing + new))
            return result

    def lookup(self, key, ctx):
        with self._lock:
            return super().lookup(key, ctx)

    def touch(self, keys, ctx):
        with self._lock:
            return super().touch(keys, ctx)

    def prepare_load(self, keys, ctx):
        with self._lock:
            keys = list(keys)
            self.on_new_request(ctx)
            result = super().prepare_load(keys, ctx)
            self._reads[(id(ctx), frozenset(keys))] += 1
            return result

    def complete_load(self, keys, ctx):
        with self._lock:
            keys = list(keys)
            ticket = (id(ctx), frozenset(keys))
            if not self._reads[ticket]:
                self.metrics['ignored_duplicate_load_completion'] += 1
                return
            super().complete_load(keys, ctx)
            self._reads[ticket] -= 1
            if not self._reads[ticket]:
                del self._reads[ticket]
            self.reclaim_tails()
            self._collect_context(ctx)

    def complete_store(self, keys, ctx, success=True):
        with self._lock:
            keys = list(keys)
            ticket = (id(ctx), frozenset(keys))
            if ticket not in self._writes:
                self.metrics['ignored_duplicate_store_completion'] += 1
                return
            super().complete_store(keys, ctx, success=success)
            del self._writes[ticket]
            self.reclaim_tails()
            self._collect_context(ctx)

    def end_session(self, session: str, generation: int):
        if get_ident() != self._owner_thread:
            raise RuntimeError('membership events must be marshalled onto the scheduler thread')
        with self._lock:
            self.controller.end(session, generation)
            return self.reclaim_tails()

    def on_request_finished(self, ctx):
        with self._lock:
            entry = self._contexts.get(id(ctx))
            if entry:
                plan = self._plan(ctx)
                meta = (ctx.kv_transfer_params or {}).get('slackkv_m1', {})
                if isinstance(meta, dict) and meta.get('end_after_request') is True and plan and plan.token:
                    self.controller.end(*plan.token)
                self.controller.finish(entry[1])
                self._finished.add(id(ctx))
            self.reclaim_tails()
            self._collect_context(ctx)

    def _collect_context(self, ctx):
        identity = id(ctx)
        if identity in self._finished and not any(t[0] == identity for t in self._writes) and not any(t[0] == identity for t in self._reads):
            self._contexts.pop(identity, None)
            self._finished.discard(identity)

    def on_schedule_end(self):
        self.reclaim_tails()

    def reset_cache(self):
        # The native scheduler first flushes jobs and advances its stale threshold.
        with self._lock:
            super().reset_cache()
            self._writes.clear()
            self._reads.clear()
            self.reclaim_tails()
            for identity in list(self._finished):
                if identity in self._contexts:
                    self._collect_context(self._contexts[identity][0])

    def snapshot_metrics(self):
        with self._lock:
            ready = sum(b.ref_cnt >= 0 for b in self._policy.blocks.values())
            pending = sum(b.ref_cnt == -1 for b in self._policy.blocks.values())
            protected = sum(b.ref_cnt > 0 for b in self._policy.blocks.values())
            members = []
            for token, member in sorted(self.controller.members.items()):
                ready_depth = allocated_depth = 0
                for key in member.path:
                    block = self._policy.get(key)
                    if block is None:
                        break
                    allocated_depth += 1
                for key in member.path:
                    block = self._policy.get(key)
                    if block is None or not block.is_ready:
                        break
                    ready_depth += 1
                owned = sum(self.controller.key_owner.get(k) == token and self._policy.get(k) is not None for k in member.path)
                members.append(dict(session=token[0], generation=token[1], allocated_prefix_blocks=allocated_depth,
                                    ready_prefix_blocks=ready_depth, owned_physical_blocks=owned,
                                    deferred_reclaim_blocks=max(0, owned-member.target)))
            free = self._get_num_free_blocks()
            if ready + pending + free != self._num_blocks:
                raise RuntimeError('native physical accounting mismatch')
            return dict(**self.controller.snapshot(), physical_ready_blocks=ready, physical_pending_store_blocks=pending,
                        physical_load_protected_blocks=protected, physical_free_blocks=free,
                        physical_reserved_bytes=self._num_blocks*self.block_bytes,
                        members_physical=members, metrics=dict(self.metrics),
                        pending_store_batches=len(self._writes), pending_load_batches=sum(self._reads.values()),
                        tracked_request_contexts=len(self._contexts))
