"""One successful demand-continuation allocation per older cold queue head.

Queue ordering only: no capacity prediction, lease, DMA or ownership mutation.
An allocation failure keeps the selected head and follows normal OnDemand.
"""
from __future__ import annotations


class ContinuationCredit:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.used: set[str] = set()
        self.pending: dict[str, str] = {}

    @staticmethod
    def resume(request) -> bool:
        return (request.kv_transfer_params or {}).get("slackkv_trace_resume") is True

    def select(self, queue, *, active_waiting_ids: set[str], protected: bool) -> None:
        if not self.enabled:
            return
        self.used.intersection_update(active_waiting_ids)
        self.pending = {candidate: head for candidate, head in self.pending.items()
                        if candidate in active_waiting_ids and head in active_waiting_ids}
        if protected or len(queue) < 2:
            return
        head = min(queue, key=lambda request: request.arrival_time)
        if (self.resume(head) or head.num_computed_tokens or head.num_preemptions
                or head.status.name != "WAITING"):
            return
        if head.request_id in self.used:
            queue.remove_request(head)
            queue.prepend_request(head)
            return
        candidates = [request for request in queue
                      if self.resume(request) and request.num_output_tokens == 0
                      and request.num_computed_tokens == 0 and not request.num_preemptions
                      and request.status.name == "WAITING"]
        if not candidates:
            return
        candidate = min(candidates, key=lambda request: request.arrival_time)
        self.pending[candidate.request_id] = head.request_id
        queue.remove_request(candidate)
        queue.prepend_request(candidate)

    def admitted(self, request_id: str) -> str | None:
        head = self.pending.pop(request_id, None)
        if head is not None:
            if head in self.used:
                raise AssertionError("continuation credit spent twice")
            self.used.add(head)
            self.pending = {candidate: older for candidate, older in self.pending.items()
                            if older != head}
        return head
