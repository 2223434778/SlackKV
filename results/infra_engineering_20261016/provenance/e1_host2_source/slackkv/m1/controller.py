"""Logical membership and ceilings; physical ownership stays in native manager."""
from dataclasses import dataclass, field

Token = tuple[str, int]


def max_min(demands: dict[Token, int], capacity: int, cursor: int = 0):
    if type(capacity) is not int or capacity < 0 or any(type(n) is not int or n < 0 for n in demands.values()):
        raise ValueError('non-negative integer block counts required')
    if not demands:
        return {}, cursor
    low, high = 0, max(demands.values())
    while low < high:
        level = (low + high + 1) // 2
        if sum(min(n, level) for n in demands.values()) <= capacity:
            low = level
        else:
            high = level - 1
    result = {key: min(n, low) for key, n in demands.items()}
    remaining = capacity - sum(result.values())
    keys = sorted(demands)
    offset = cursor % len(keys)
    order = keys[offset:] + keys[:offset]
    for key in order:
        if remaining and result[key] < demands[key]:
            result[key] += 1
            remaining -= 1
    return result, cursor + 1


@dataclass
class Member:
    token: Token
    path: tuple[bytes, ...]
    active: set[str] = field(default_factory=set)
    alive: bool = True
    unsupported: str | None = None
    target: int = 0
    retention_blocks: int = 0


@dataclass(frozen=True)
class Plan:
    handle: str
    token: Token | None
    path: tuple[bytes, ...]
    ceiling: int
    version: int
    reason: str | None = None
    terminal: bool = False


class PrefixController:
    def __init__(self, capacity: int, shared_prefix_blocks: int = 0):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('non-negative capacity required')
        self.capacity = capacity
        if type(shared_prefix_blocks) is not int or shared_prefix_blocks < 0:
            raise ValueError('non-negative shared-root block limit required')
        self.shared_prefix_blocks = shared_prefix_blocks
        self.members: dict[Token, Member] = {}
        self.plans: dict[str, Plan] = {}
        self.key_owner: dict[bytes, Token] = {}
        self.key_refs: dict[bytes, set[Token]] = {}
        self.latest: dict[str, int] = {}
        # One terminal bit for the latest generation, independent of Member GC.
        self.ended: set[str] = set()
        self.version = self.cursor = 0

    def rebalance(self):
        demands = self.demands()
        if self.shared_prefix_blocks and demands:
            # Max-min logical depths, charged by the union of native identities.
            def cost(shares):
                return len({k for t,n in shares.items() for k in self.members[t].path[:n]})
            low, high = 0, max(demands.values())
            while low < high:
                level = (low+high+1)//2
                if cost({t:min(n,level) for t,n in demands.items()}) <= self.capacity:low=level
                else:high=level-1
            shares={t:min(n,low) for t,n in demands.items()}
            tokens=sorted(demands);offset=self.cursor%len(tokens)
            for t in tokens[offset:]+tokens[:offset]:
                if shares[t] < demands[t]:
                    trial={**shares,t:shares[t]+1}
                    if cost(trial) <= self.capacity:shares=trial
            self.cursor += 1
        else:
            shares, self.cursor = max_min(demands, self.capacity, self.cursor)
        for t, m in self.members.items():
            m.target = shares.get(t, 0)
        self.version += 1

    def demands(self):
        return {t:m.retention_blocks for t,m in self.members.items()
                if m.alive and not m.unsupported and m.retention_blocks > 0}

    def observe(self, handle: str, session: str | None, generation: int | None, path: tuple[bytes, ...], terminal: bool = False):
        if handle in self.plans:
            return self.plans[handle]
        reason = None
        token = None
        previous_demands = self.demands()
        if not isinstance(session, str) or not session or type(generation) is not int or generation < 0:
            reason = 'missing_or_invalid_membership'
        elif generation < self.latest.get(session, -1):
            reason = 'stale_generation'
        elif generation == self.latest.get(session, -1) and session in self.ended:
            reason = 'ended_generation_requires_new_generation'
        elif len(set(path)) != len(path):
            reason = 'duplicate_identity'
        else:
            token = (session, generation)
            if generation > self.latest.get(session, -1):
                for old, member in self.members.items():
                    if old[0] == session:
                        member.alive = False
                self.latest[session] = generation
                self.ended.discard(session)
            member = self.members.get(token)
            if member is None:
                member = self.members[token] = Member(token, path)
            elif not member.alive:
                reason = 'ended_generation_requires_new_generation'
            elif member.active:
                member.unsupported = reason = 'parallel_request'
            elif path[:len(member.path)] != member.path:
                member.unsupported = reason = 'rewritten_or_truncated_input'
            if reason is None:
                member.path = path
                if self.shared_prefix_blocks:
                    # Only shared root / independent tails, never merged paths.
                    for other,m in self.members.items():
                        if other == token:continue
                        common=0
                        for a,b in zip(path,m.path):
                            if a != b:break
                            common += 1
                        overlap=set(path)&set(m.path)
                        if other[0] == session and not m.alive and common > self.shared_prefix_blocks:
                            reason='retired_identity_pending'
                            break
                        if overlap and (common > self.shared_prefix_blocks or overlap != set(path[:common])):
                            member.unsupported='unsupported_shared_shape'
                            break
                    if not member.unsupported and reason is None:
                        for key in path:self.key_refs.setdefault(key,set()).add(token)
                else:
                    foreign = {self.key_owner[k] for k in path if k in self.key_owner and self.key_owner[k] != token}
                    if any(t[0] != session or self.members[t].alive for t in foreign):
                        member.unsupported = 'shared_prefix'
                    elif foreign:
                        reason = 'retired_identity_pending'
                    if not member.unsupported and not foreign:
                        for key in path:self.key_owner[key] = token
                reason = reason or member.unsupported
                if reason is None and not terminal:member.retention_blocks = len(path)
            member.active.add(handle)
            # A last request keeps its old prefix for reading, without demand
            # growth or remainder rotation; a new terminal member demands zero.
            if not terminal or self.demands() != previous_demands:self.rebalance()
        ceiling = self.members[token].target if token is not None and reason is None else 0
        plan = Plan(handle, token, path, ceiling, self.version, reason, terminal)
        self.plans[handle] = plan
        return plan

    def allowed(self, plan: Plan) -> int:
        member = self.members.get(plan.token)
        if plan.terminal or plan.reason or member is None or not member.alive or member.unsupported or plan.handle not in member.active:
            return 0
        return min(plan.ceiling, member.target)

    def end(self, session: str, generation: int):
        # Unseen/future events are no-ops; only a known current generation ends.
        # A delayed end for an older generation cannot fence a newer request.
        if self.latest.get(session) != generation or session in self.ended:
            return
        self.ended.add(session)
        member = self.members.get((session, generation))
        if member is not None and member.alive:
            previous_demands=self.demands()
            member.alive = False
            if self.demands()!=previous_demands:self.rebalance()

    def finish(self, handle: str):
        plan = self.plans.pop(handle, None)
        if plan and plan.token in self.members:
            self.members[plan.token].active.discard(handle)

    def collect(self, token: Token):
        """Adapter confirms no private/unsettled keys remain; survivors may share."""
        member = self.members[token]
        if member.alive or member.active:
            return
        for key in member.path:
            if self.key_owner.get(key) == token:
                del self.key_owner[key]
            refs=self.key_refs.get(key)
            if refs is not None:
                refs.discard(token)
                if not refs:del self.key_refs[key]
        del self.members[token]

    def snapshot(self):
        return dict(capacity_blocks=self.capacity, plan_version=self.version,
                    members=[dict(session=t[0], generation=t[1], demand_blocks=m.retention_blocks,
                                  request_path_blocks=len(m.path),
                                  target_blocks=m.target, active_requests=len(m.active), alive=m.alive,
                                  unsupported=m.unsupported) for t, m in sorted(self.members.items())],
                    frozen_plans=len(self.plans), generation_tombstones=len(self.latest),
                    ended_generation_tombstones=len(self.ended),
                    shared_prefix_blocks=self.shared_prefix_blocks,
                    target_distinct_blocks=len({k for m in self.members.values() for k in m.path[:m.target]}),
                    unmaterialized_reservations=0)
