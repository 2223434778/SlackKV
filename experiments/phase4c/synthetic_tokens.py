"""Cross-process deterministic token IDs and exact prefix construction."""
from __future__ import annotations

import hashlib
import random
from typing import Iterable, Sequence


def stable_seed(phase: str, config: str, session_hash: str, turn_index: int, suffix_position: int = 0) -> int:
    digest = hashlib.sha256(f"{phase}|{config}|{session_hash}|{turn_index}|{suffix_position}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_token_ids(length: int, *, phase: str, config: str, session_hash: str, turn_index: int,
                            vocab_size: int, special_ids: Iterable[int] = (), excluded_ids: Iterable[int] = ()) -> list[int]:
    if length < 0 or vocab_size <= 0:
        raise ValueError("invalid token length or vocabulary")
    excluded = set(special_ids) | set(excluded_ids)
    allowed = [token_id for token_id in range(vocab_size) if token_id not in excluded]
    if not allowed and length:
        raise ValueError("no usable token IDs")
    rng = random.Random(stable_seed(phase, config, session_hash, turn_index))
    return [allowed[rng.randrange(len(allowed))] for _ in range(length)]


def make_next_prompt(previous_served: Sequence[int], target_isl: int, cached_tokens: int,
                     *, phase: str, config: str, session_hash: str, turn_index: int,
                     vocab_size: int, special_ids: Iterable[int] = (), excluded_ids: Iterable[int] = ()) -> list[int]:
    if cached_tokens < 0 or cached_tokens > len(previous_served) or target_isl < cached_tokens:
        raise ValueError("requested shared prefix cannot be produced")
    suffix = deterministic_token_ids(target_isl - cached_tokens, phase=phase, config=config,
                                     session_hash=session_hash, turn_index=turn_index,
                                     vocab_size=vocab_size, special_ids=special_ids, excluded_ids=excluded_ids)
    if suffix and cached_tokens < len(previous_served) and suffix[0] == previous_served[cached_tokens]:
        replacement = deterministic_token_ids(1, phase=phase, config=config, session_hash=session_hash,
                                              turn_index=turn_index, suffix_position=1, vocab_size=vocab_size,
                                              special_ids=special_ids, excluded_ids=excluded_ids)[0]
        if replacement == previous_served[cached_tokens]:
            replacement = (replacement + 1) % vocab_size
        suffix[0] = replacement
    prompt = list(previous_served[:cached_tokens]) + suffix
    if len(prompt) != target_isl:
        raise AssertionError("constructed prompt length mismatch")
    if common_prefix(prompt, previous_served) != cached_tokens:
        raise AssertionError("token-level prefix does not end at the requested boundary")
    return prompt


def common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    size = 0
    for a, b in zip(left, right):
        if a != b:
            break
        size += 1
    return size


def longest_full_block_match(prompt: Sequence[int], previous_prompts: Sequence[Sequence[int]], block_size: int = 16) -> int:
    longest = 0
    for previous in previous_prompts:
        shared = common_prefix(prompt, previous)
        longest = max(longest, (shared // block_size) * block_size)
    return longest


def token_hash(token_ids: Sequence[int]) -> str:
    packed = ",".join(map(str, token_ids)).encode("ascii")
    return hashlib.sha256(packed).hexdigest()
