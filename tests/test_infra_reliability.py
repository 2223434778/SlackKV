import json
import os
from pathlib import Path
import socket
import sys

import pytest

from slackkv.infra.config import Config
from slackkv.infra.service import ManagedService,ServiceStopped,init_budget,stop_owned


def test_stale_owner_identity_never_signals_unrelated_process(tmp_path,monkeypatch):
    (tmp_path/'execution.json').write_text(json.dumps(dict(status='RUNNING',parent_pid=123,parent_birth='old')))
    monkeypatch.setattr('slackkv.infra.service.birth',lambda pid:'new')
    sent=[];monkeypatch.setattr('os.kill',lambda *args:sent.append(args))
    with pytest.raises(RuntimeError):stop_owned(tmp_path)
    assert sent==[]


def test_stop_rejects_process_group_and_missing_birth(tmp_path,monkeypatch):
    (tmp_path/'execution.json').write_text(json.dumps(dict(status='RUNNING',parent_pid=0,parent_birth=None)))
    monkeypatch.setattr('os.kill',lambda *args:pytest.fail('must not signal PID0'))
    with pytest.raises(RuntimeError):stop_owned(tmp_path)


@pytest.mark.skipif(os.name!='posix',reason='real POSIX owned-group contract; Windows runs other CPU protocol tests')
def test_actual_failed_child_is_reaped_and_charged(tmp_path,monkeypatch):
    init_budget(tmp_path)
    config=Config('model','runtime',ready_timeout_seconds=2,max_run_seconds=60)
    monkeypatch.setattr('slackkv.infra.service.doctor',lambda c:{'status':'CPU_TEST'})
    monkeypatch.setattr('slackkv.infra.service.command',lambda *a:[sys.executable,'-c','raise SystemExit(7)'])
    monkeypatch.setattr('slackkv.infra.service.gpu_state',lambda i:dict(memory_mib=19,utilization_percent=0))
    with pytest.raises(RuntimeError):
        with ManagedService(config,tmp_path,'failed'):pass
    record=json.loads((tmp_path/'runs/failed/execution.json').read_text())
    assert record['status']=='FAILED' and record['cleanup_verified'] and record['gpu_hours']>0
    assert record['server_exit_code']==7 and not record['owned_live_pids_after']


@pytest.mark.skipif(os.name!='posix',reason='real POSIX owned-group contract; no GPU required')
def test_actual_ready_child_user_stop_is_acknowledged(tmp_path,monkeypatch):
    init_budget(tmp_path)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    config=Config('model','runtime',port=port,ready_timeout_seconds=3,max_run_seconds=60)
    script='from http.server import BaseHTTPRequestHandler,HTTPServer\nclass H(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200);self.end_headers()\nHTTPServer(("127.0.0.1",'+str(port)+'),H).serve_forever()'
    monkeypatch.setattr('slackkv.infra.service.doctor',lambda c:{'status':'CPU_TEST'})
    monkeypatch.setattr('slackkv.infra.service.command',lambda *a:[sys.executable,'-c',script])
    monkeypatch.setattr('slackkv.infra.service.gpu_state',lambda i:dict(memory_mib=19,utilization_percent=0))
    with ManagedService(config,tmp_path,'stopped'):
        raise ServiceStopped('user requested')
    record=json.loads((tmp_path/'runs/stopped/execution.json').read_text())
    assert record['status']=='STOPPED' and record['cleanup_verified'] and not record['owned_live_pids_after']
