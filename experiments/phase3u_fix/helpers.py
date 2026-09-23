POLICIES=("OnDemand-K2","Blocking70-K2","NaiveAsync40-K2","ReservationLedger2-Async40-K2")
def policy_semantics(policy):
    return {
        "service_ms":70.0 if policy=="Blocking70-K2" else (40.0 if policy in ("NaiveAsync40-K2","ReservationLedger2-Async40-K2") else None),
        "synchronous":policy=="Blocking70-K2",
        "block_on_prefetch":policy=="Blocking70-K2",
        "lease":policy=="ReservationLedger2-Async40-K2",
        "promote":policy!="OnDemand-K2",
    }
def h2d(native_count,external_tokens):
    native=native_count*112.0; external=external_tokens*112.0/4096.0; total=native+external
    return native,external,total,max(0.0,total-1344.0)
def source_conserved(local,external):return local+external==49152
def x_slowdown(duration,baseline):return duration/baseline
def category_percentiles(values,percentile):
    if not values:return "NA"
    xs=sorted(float(x) for x in values); pos=(len(xs)-1)*percentile/100; lo=int(pos); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
