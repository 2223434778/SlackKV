"""Deterministic Phase 1C KV-offload admission rules."""
def slack_ratio(pause_s, down_s, up_s):
    return pause_s / (down_s + up_s)

def admit(policy, pause_s, down_s, up_s):
    if policy == "KeepAll": return False
    if policy == "OffloadAll-QueueJIT": return True
    if policy == "FeasibilityOracle": return slack_ratio(pause_s, down_s, up_s) >= 1.0
    if policy.startswith("FixedTTL-"): return pause_s >= float(policy.split("-",1)[1])
    if policy.startswith("SlackRatio-"): return slack_ratio(pause_s, down_s, up_s) >= float(policy.split("-",1)[1])
    raise ValueError(policy)
