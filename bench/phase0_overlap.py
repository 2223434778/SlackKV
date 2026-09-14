#!/usr/bin/env python3
"""Phase 0 compute/transfer overlap benchmark using explicit non-default CUDA streams."""
import argparse, csv, math, statistics, time
from pathlib import Path
import torch

MiB = 1 << 20

def event_ms(start, end):
    end.synchronize()
    return start.elapsed_time(end)

def percentile(values, p):
    values = sorted(values)
    if not values: return float("nan")
    x = (len(values)-1)*p; lo, hi = int(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] + (values[hi]-values[lo])*(x-lo)

class Workload:
    def __init__(self, name, device):
        self.name, self.device = name, device
        if name == "gemm_fp16":
            self.a = torch.randn((4096,4096), device=device, dtype=torch.float16)
            self.b = torch.randn((4096,4096), device=device, dtype=torch.float16)
            self.out = torch.empty((4096,4096), device=device, dtype=torch.float16)
        else:
            # 512 MiB tensors; this is explicitly a synthetic memory-bandwidth workload.
            n = 128 * MiB
            self.a = torch.ones(n, device=device, dtype=torch.float32)
            self.b = torch.full((n,), 1.001, device=device, dtype=torch.float32)
            self.out = torch.empty(n, device=device, dtype=torch.float32)
        self.repeats = 1

    def launch(self, stream):
        with torch.cuda.stream(stream):
            for _ in range(self.repeats):
                if self.name == "gemm_fp16": torch.mm(self.a, self.b, out=self.out)
                else: torch.add(self.a, self.b, out=self.out)

    def calibrate(self, target_ms):
        stream = torch.cuda.Stream(device=self.device)
        self.repeats = 1
        for _ in range(8):
            torch.cuda.synchronize(self.device)
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            with torch.cuda.stream(stream): start.record(); self.launch(stream); end.record()
            ms = event_ms(start, end)
            if ms >= target_ms * 0.75: break
            self.repeats *= max(2, math.ceil(target_ms / max(ms, 0.01)))
        return ms

def transfer_setup(kind, size):
    n = size
    if kind == "peer_in": return torch.empty(n, dtype=torch.uint8, device="cuda:1"), torch.empty(n, dtype=torch.uint8, device="cuda:0"), 0
    if kind == "peer_out": return torch.empty(n, dtype=torch.uint8, device="cuda:0"), torch.empty(n, dtype=torch.uint8, device="cuda:1"), 1
    if kind == "cpu_in": return torch.empty(n, dtype=torch.uint8, pin_memory=True), torch.empty(n, dtype=torch.uint8, device="cuda:0"), 0
    if kind == "cpu_out": return torch.empty(n, dtype=torch.uint8, device="cuda:0"), torch.empty(n, dtype=torch.uint8, pin_memory=True), 0
    raise ValueError(kind)

def launch_copy(src, dst, stream, repeats):
    with torch.cuda.stream(stream):
        for _ in range(repeats): dst.copy_(src, non_blocking=True)

def transfer_time(src, dst, device, repeats):
    stream = torch.cuda.Stream(device=device); torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.cuda.stream(stream): start.record(); launch_copy(src,dst,stream,repeats); end.record()
    return event_ms(start,end)

def write_row(writer, **row): writer.writerow(row)

FIELDS = ["record_type","workload","transfer_type","size_bytes","size_mib","trial","compute_ms","transfer_ms","total_ms","compute_slowdown","transfer_slowdown","overlap_efficiency","transfer_repeats","success","error"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",required=True); ap.add_argument("--summary-output",required=True)
    ap.add_argument("--warmup",type=int,default=5); ap.add_argument("--iterations",type=int,default=30)
    ap.add_argument("--sizes-mib",default="256,1024,4096"); ap.add_argument("--target-ms",type=float,default=200.0); args=ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()<2: raise SystemExit("requires two CUDA devices")
    peer = torch.cuda.can_device_access_peer(0,1) and torch.cuda.can_device_access_peer(1,0)
    sizes=[int(x)*MiB for x in args.sizes_mib.split(",")]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for work_name in ("gemm_fp16","synthetic_memory_bandwidth"):
      workload=Workload(work_name,0); workload.calibrate(args.target_ms)
      compute_stream=torch.cuda.Stream(device=0)
      compute_times=[]
      for _ in range(args.warmup): workload.launch(compute_stream)
      torch.cuda.synchronize(0)
      for trial in range(args.iterations):
        start,end=torch.cuda.Event(True),torch.cuda.Event(True)
        with torch.cuda.stream(compute_stream): start.record(); workload.launch(compute_stream); end.record()
        ms=event_ms(start,end); compute_times.append(ms)
        rows.append(dict(record_type="compute_only",workload=work_name,transfer_type="none",size_bytes=0,size_mib=0,trial=trial,compute_ms=ms,transfer_ms="",total_ms=ms,compute_slowdown="",transfer_slowdown="",overlap_efficiency="",transfer_repeats=0,success=True,error=""))
      compute_base=statistics.median(compute_times)
      for kind in ("peer_in","peer_out","cpu_in","cpu_out"):
       if kind.startswith("peer") and not peer:
        for size in sizes: rows.append(dict(record_type="unsupported",workload=work_name,transfer_type=kind,size_bytes=size,size_mib=size/MiB,trial=0,compute_ms="",transfer_ms="",total_ms="",compute_slowdown="",transfer_slowdown="",overlap_efficiency="",transfer_repeats=0,success=False,error="direct CUDA peer access unavailable"))
        continue
       for size in sizes:
        try:
          src,dst,transfer_device=transfer_setup(kind,size)
          one_ms=transfer_time(src,dst,transfer_device,1); repeats=max(1,round(args.target_ms/max(one_ms,0.1)))
          transfer_times=[]
          for _ in range(args.warmup): transfer_time(src,dst,transfer_device,repeats)
          for trial in range(args.iterations):
            ms=transfer_time(src,dst,transfer_device,repeats); transfer_times.append(ms)
            rows.append(dict(record_type="transfer_only",workload=work_name,transfer_type=kind,size_bytes=size,size_mib=size/MiB,trial=trial,compute_ms="",transfer_ms=ms,total_ms=ms,compute_slowdown="",transfer_slowdown="",overlap_efficiency="",transfer_repeats=repeats,success=True,error=""))
          transfer_base=statistics.median(transfer_times)
          # Warm-up is per configuration, outside the measured concurrent trials.
          transfer_stream=torch.cuda.Stream(device=transfer_device)
          for _ in range(args.warmup):
            workload.launch(compute_stream); launch_copy(src,dst,transfer_stream,repeats)
          torch.cuda.synchronize(0); torch.cuda.synchronize(1)
          for trial in range(args.iterations):
            transfer_stream=torch.cuda.Stream(device=transfer_device)
            cs,ce=torch.cuda.Event(True),torch.cuda.Event(True); ts,te=torch.cuda.Event(True),torch.cuda.Event(True)
            torch.cuda.synchronize(0); torch.cuda.synchronize(1); wall=time.perf_counter()
            with torch.cuda.stream(compute_stream): cs.record(); workload.launch(compute_stream); ce.record()
            with torch.cuda.stream(transfer_stream): ts.record(); launch_copy(src,dst,transfer_stream,repeats); te.record()
            torch.cuda.synchronize(0); torch.cuda.synchronize(1); total=(time.perf_counter()-wall)*1000
            cm=event_ms(cs,ce); tm=event_ms(ts,te); efficiency=(compute_base+transfer_base-total)/min(compute_base,transfer_base)
            rows.append(dict(record_type="concurrent",workload=work_name,transfer_type=kind,size_bytes=size,size_mib=size/MiB,trial=trial,compute_ms=cm,transfer_ms=tm,total_ms=total,compute_slowdown=cm/compute_base-1,transfer_slowdown=tm/transfer_base-1,overlap_efficiency=efficiency,transfer_repeats=repeats,success=True,error=""))
          del src,dst; torch.cuda.empty_cache()
        except RuntimeError as e:
          rows.append(dict(record_type="failed",workload=work_name,transfer_type=kind,size_bytes=size,size_mib=size/MiB,trial=0,compute_ms="",transfer_ms="",total_ms="",compute_slowdown="",transfer_slowdown="",overlap_efficiency="",transfer_repeats=0,success=False,error=str(e).replace("\n"," ")))
    with open(args.output,"w",newline="") as f: writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    groups={}
    for r in rows:
      if r["record_type"]=="concurrent" and r["success"]:
        groups.setdefault((r["workload"],r["transfer_type"],r["size_mib"]),[]).append(r)
    summary=[]
    for (w,t,s),rs in groups.items():
      def med(k): return statistics.median(float(r[k]) for r in rs)
      summary.append({"workload":w,"transfer_type":t,"size_mib":s,"count":len(rs),"compute_ms_p50":med("compute_ms"),"transfer_ms_p50":med("transfer_ms"),"total_ms_p50":med("total_ms"),"compute_slowdown_p50":med("compute_slowdown"),"transfer_slowdown_p50":med("transfer_slowdown"),"overlap_efficiency_p50":med("overlap_efficiency")})
    with open(args.summary_output,"w",newline="") as f: writer=csv.DictWriter(f,fieldnames=list(summary[0]) if summary else ["workload"],lineterminator="\n"); writer.writeheader(); writer.writerows(summary)
if __name__ == "__main__": main()
