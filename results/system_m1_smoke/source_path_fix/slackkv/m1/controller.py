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


@dataclass(frozen=True)
class Plan:
    handle: str
    token: Token | None
    path: tuple[bytes, ...]
    ceiling: int
    version: int
    reason: str | None = None


class PrefixController:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('non-negative capacity required')
        self.capacity = capacity
        self.members: dict[Token, Member] = {}
        self.plans: dict[str, Plan] = {}
        self.key_owner: dict[bytes, Token] = {}
        self.latest: dict[str, int] = {}
        # One terminal bit for the latest generation, independent of Member GC.
        self.ended: set[str] = set()
        self.version = self.cursor = 0

    def rebalance(self):
        demands = {t: len(m.path) for t, m in self.members.items() if m.alive and not m.unsupported}
        shares, self.cursor = max_min(demands, self.capacity, self.cursor)
        for t, m in self.members.items():
            m.target = shares.get(t, 0)
        self.version += 1

    def observe(self, handle: str, session: str | None, generation: int | None, path: tuple[bytes, ...]):
        if handle in self.plans:
            return self.plans[handle]
        reason = None
        token = None
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
                foreign = {self.key_owner[k] for k in path if k in self.key_owner and self.key_owner[k] != token}
                if any(t[0] != session or self.members[t].alive for t in foreign):
                    member.unsupported = 'shared_prefix'
                elif foreign:
                    reason = 'retired_identity_pending'
                if not member.unsupported and not foreign:
                    for key in path:
                        self.key_owner[key] = token
                reason = reason or member.unsupported
            member.active.add(handle)
            self.rebalance()
        ceiling = self.members[token].target if token is not None and reason is None else 0
        plan = Plan(handle, token, path, ceiling, self.version, reason)
        self.plans[handle] = plan
        return plan

    def allowed(self, plan: Plan) -> int:
        member = self.members.get(plan.token)
        if plan.reason or member is None or not member.alive or member.unsupported or plan.handle not in member.active:
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
            member.alive = False
            self.rebalance()

    def finish(self, handle: str):
        plan = self.plans.pop(handle, None)
        if plan and plan.token in self.members:
            self.members[plan.token].active.discard(handle)

    def collect(self, token: Token):
        """Adapter calls only after confirming no native resident keys remain."""
        member = self.members[token]
        if member.alive or member.active:
            return
        for key in member.path:
            if self.key_owner.get(key) == token:
                del self.key_owner[key]
        del self.members[token]

    def snapshot(self):
        return dict(capacity_blocks=self.capacity, plan_version=self.version,
                    members=[dict(session=t[0], generation=t[1], demand_blocks=len(m.path),
                                  target_blocks=m.target, active_requests=len(m.active), alive=m.alive,
                                  unsupported=m.unsupported) for t, m in sorted(self.members.items())],
                    frozen_plans=len(self.plans), generation_tombstones=len(self.latest),
                    ended_generation_tombstones=len(self.ended),
                    unmaterialized_reservations=0)
