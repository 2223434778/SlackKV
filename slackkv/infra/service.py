"""Owned POSIX service lifetime with one continuous engineering budget ledger."""
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from slackkv.infra.config import layout
from slackkv.infra.http import get
from slackkv.infra.runtime import PINNED,check_runtime,digest,save,versions

HISTORY_GPU_HOURS=4.570877168576147


class ServiceStopped(RuntimeError):
    """Acknowledged user stop, distinct from watchdog/startup/workload failure."""


def init_budget(path):
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
    with (path/'authorization.json').open('x',encoding='utf-8') as f:
        json.dump(dict(grant='infra-e1-20261016',limit_gpu_hours=.5,ledger_path=str(path),
            historical_gpu_hours=HISTORY_GPU_HOURS,scope='E1 functional smoke and scoped E2 validation; all startup/cleanup included'),f,indent=2)


def budget(path):
    path=Path(path).resolve();auth=json.loads((path/'authorization.json').read_text(encoding='utf-8'))
    if auth['ledger_path']!=str(path) or not 0<auth['limit_gpu_hours']<=.5:raise ValueError('invalid/moved engineering grant')
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in path.glob('runs/*/execution.json')]
    if any(r['status']=='RUNNING' or not r['cleanup_verified'] for r in rows):raise RuntimeError('unresolved owned run; reconcile evidence before launching')
    if any(not math.isfinite(r['gpu_hours']) or r['gpu_hours']<0 for r in rows):raise ValueError('invalid cost')
    return auth,math.fsum(r['gpu_hours'] for r in rows)


def gpu_state(index):
    text=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    values=[int(x.strip()) for x in text.split(',')]
    return dict(memory_mib=values[0],utilization_percent=values[1])


def birth(pid):
    try:return Path(f'/proc/{pid}/stat').read_text(encoding='utf-8').rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def live_group(pgid):
    output=subprocess.check_output(['ps','-eo','pid=,pgid=,stat='],text=True,timeout=5)
    return [int(p) for line in output.splitlines() for p,g,s in [line.split()]
            if int(g)==pgid and not s.startswith('Z')]


def command(config,host_bytes):
    argv=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',config.model,'--served-model-name','slackkv',
        '--dtype','float16','--tensor-parallel-size','1','--enable-prefix-caching',
        '--kv-cache-memory-bytes',str(config.gpu_kv_bytes),'--max-model-len',str(config.max_model_len),
        '--max-num-seqs',str(config.session_slots),'--max-num-batched-tokens','8192',
        '--host',config.host,'--port',str(config.port)]
    if config.policy!='nohost':
        extra=dict(block_size=16,cpu_bytes_to_use=host_bytes,spec_name='InfraCPUSpec',spec_module_path='slackkv.infra.native')
        argv+=['--kv-transfer-config',json.dumps(dict(kv_connector='OffloadingConnector',kv_role='kv_both',kv_connector_extra_config=extra))]
    return argv


def doctor(config,config_only=False):
    result=dict(config=config.snapshot(),config_valid=True,gpu_initialized=False)
    if config_only:return dict(**result,status='PASS',scope='schema only; no model/native/content proof')
    result['layout']=layout(config);result['versions']=versions()
    if result['versions']!=PINNED:raise ValueError('validated package versions required: '+str(PINNED)+'; found '+str(result['versions']))
    if sys.version_info[:2]!=(3,11):raise ValueError('validated runtime requires Python3.11; CPU config tests support3.11+')
    runtime=check_runtime(config.runtime)
    result['runtime']=dict(upstream_commit=runtime['upstream_commit'],patch=runtime['patch'],manifest_sha256=digest(Path(config.runtime)/'slackkv-runtime.json'))
    with socket.socket() as probe:
        if probe.connect_ex((config.host,config.port))==0:raise RuntimeError('port already occupied; will not stop existing service')
    result['gpu']=gpu_state(config.gpu)
    if result['gpu']['memory_mib']>512 or result['gpu']['utilization_percent']!=0:raise RuntimeError('selected GPU occupied; leave other service alone')
    if not any(Path(config.model).glob('*.safetensors')):raise ValueError('local model weights missing')
    if not (Path(config.model)/'tokenizer_config.json').exists():raise ValueError('chat tokenizer configuration missing')
    result['host_mem_available_bytes']=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines() if l.startswith('MemAvailable:'))
    if config.policy!='nohost' and result['host_mem_available_bytes']<config.host_kv_bytes+2*2**30:raise RuntimeError('insufficient host headroom')
    return dict(**result,status='PASS',scope='static checks; runtime factory and real content need integration validation')


class ManagedService:
    def __init__(self,config,ledger,name):
        if not name or Path(name).name!=name or name in ('.','..'):raise ValueError('plain unique run name required')
        self.config=config;self.ledger=Path(ledger).resolve();self.out=self.ledger/'runs'/name
        self.proc=None;self.started=None;self.record=None;self.lock=None;self.old_handlers={}

    def __enter__(self):
        try:return self._enter()
        except BaseException:
            if self.lock and self.started is None:self.lock.close()
            raise

    def _enter(self):
        if os.name!='posix':raise RuntimeError('serve requires Linux; doctor --config-only/demo unit tests work without GPUs')
        import fcntl
        self.lock=(self.ledger/'budget.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth,spent=budget(self.ledger)
        limit=min(self.config.max_run_seconds,int((auth['limit_gpu_hours']-spent)*3600)-2)
        if limit<self.config.ready_timeout_seconds+45:raise RuntimeError('insufficient engineering grant for startup plus cleanup reserve')
        self.out.mkdir(parents=True,exist_ok=False)
        try:
            checks=doctor(self.config)
            save(self.out/'doctor.json',checks)
            argv=command(self.config,self.config.host_kv_bytes)
            env=dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=str(self.config.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                VLLM_BATCH_INVARIANT=str(int(self.config.batch_invariant)),VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
            env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
            # Use pinned pristine Python modules, not installed historical research hooks.
            env['PYTHONPATH']=str(Path(self.config.runtime).resolve())+os.pathsep+str(Path(__file__).resolve().parents[2])
            self.started=time.monotonic()
            self.record=dict(status='RUNNING',cleanup_verified=False,gpu_hours=0,gpu_count=1,
                started_unix=time.time(),config=self.config.snapshot(),command=argv,budget_before=spent,
                historical_gpu_hours=auth['historical_gpu_hours'],wall_limit_seconds=limit,
                parent_pid=os.getpid(),parent_birth=birth(os.getpid()),
                project_source_sha256={str(p.relative_to(Path(__file__).parents[2])):digest(p) for p in Path(__file__).parent.glob('*.py')})
            save(self.out/'execution.json',self.record)
            self.log=(self.out/'server.log').open('w')
            self.proc=subprocess.Popen(argv,env=env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            self.record.update(owned_pgid=self.proc.pid,owned_birth=birth(self.proc.pid))
            save(self.out/'execution.json',self.record)
            def interrupted(signum,frame):
                if signum==signal.SIGALRM:raise TimeoutError('owned service watchdog')
                raise ServiceStopped('requested owned service stop')
            for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
                self.old_handlers[sig]=signal.signal(sig,interrupted)
            signal.alarm(limit-35)
            deadline=time.monotonic()+self.config.ready_timeout_seconds
            while time.monotonic()<deadline:
                if self.proc.poll() is not None:raise RuntimeError('vLLM startup failed; inspect server.log')
                try:
                    if get(self.config.url+'/health',2)=='':break
                except OSError:pass
                time.sleep(.5)
            else:raise TimeoutError('vLLM ready deadline exceeded')
            self.record['ready_seconds']=time.monotonic()-self.started
            save(self.out/'execution.json',self.record)
            return self
        except BaseException as error:
            self.stop(error)
            raise

    def stop(self,error=None):
        if self.started is None:
            if self.lock:self.lock.close()
            return
        signal.alarm(0)
        for sig,handler in self.old_handlers.items():signal.signal(sig,handler)
        remaining=[];released=False
        try:
            if self.proc:
                current=birth(self.proc.pid)
                if current is not None and current!=self.record['owned_birth']:raise RuntimeError('PID identity changed; refuse signal')
                for sig,seconds in ((signal.SIGTERM,20),(signal.SIGKILL,5)):
                    if not live_group(self.proc.pid):break
                    try:os.killpg(self.proc.pid,sig)
                    except ProcessLookupError:break
                    until=time.monotonic()+seconds
                    while live_group(self.proc.pid) and time.monotonic()<until:time.sleep(.2)
                self.proc.poll();remaining=live_group(self.proc.pid)
                self.record['server_exit_code']=self.proc.returncode
                self.log.close()
            until=time.monotonic()+5
            while True:
                state=gpu_state(self.config.gpu)
                released=state['memory_mib']<=512
                if released or time.monotonic()>=until:break
                time.sleep(.2)
            self.record['gpu_after']=state
            self.record['owned_live_pids_after']=remaining
            self.record['cleanup_verified']=not remaining and released
            self.record['status']=('FAILED' if error else 'COMPLETE') if self.record['cleanup_verified'] else 'RUNNING'
            if isinstance(error,ServiceStopped) and self.record['cleanup_verified']:
                self.record['status']='STOPPED'
            if error:self.record['error']=str(error)
        finally:
            elapsed=time.monotonic()-self.started
            self.record['wall_seconds_including_startup_cleanup']=elapsed
            self.record['gpu_hours']=(elapsed+1)/3600
            self.record['continuous_gpu_hours']=self.record['historical_gpu_hours']+self.record['budget_before']+self.record['gpu_hours']
            save(self.out/'execution.json',self.record)
            self.started=None
            if self.lock:self.lock.close()

    def __exit__(self,kind,error,tb):
        self.stop(error)
        return isinstance(error,ServiceStopped)


def stop_owned(out):
    record=json.loads((Path(out)/'execution.json').read_text(encoding='utf-8'))
    if record['status']!='RUNNING':return 'already stopped'
    pid=record['parent_pid']
    if type(pid) is not int or pid<=1 or record.get('parent_birth') is None or birth(pid)!=record['parent_birth']:
        raise RuntimeError('owner absent or PID changed; manual reconciliation needed; no signal sent')
    os.kill(pid,signal.SIGTERM)
    return 'termination sent to verified owner; poll execution.json for cleanup acknowledgement'
