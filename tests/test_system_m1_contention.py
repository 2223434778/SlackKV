import json
from pathlib import Path

from experiments.system_m1.contention import SEQUENCE, store_cap
from experiments.system_m1.smoke import command


def config():
    return json.loads(Path('experiments/system_m1/contention_config.json').read_text())


def test_common_terminal_gate_uses_identical_resources_and_native_full():
    c=config()
    for policy in ('m1','fair','full'):
        argv=command(c,policy,'mechanism',234881024)
        extra=json.loads(argv[-1])['kv_connector_extra_config']
        assert extra['cpu_bytes_to_use']==234881024
        assert '--enforce-eager' not in argv
        if policy=='m1':
            assert extra['slackkv_terminal_store_gate'] is True
        else:
            assert extra['spec_name']=='TerminalCPUOffloadingSpec'
    original=json.loads(Path('experiments/system_m1/smoke_config.json').read_text())
    assert 'slackkv_terminal_store_gate' not in json.loads(command(original,'m1','mechanism',234881024)[-1])['kv_connector_extra_config']


def test_one_frozen_workload_grows_sessions_and_counts_all_four_fair_members():
    c=config()
    assert len(SEQUENCE)==17
    assert sum(c['session_tokens'].values())>2*c['host_tokens']
    assert store_cap(c,'fair','a',8192)==2048
    assert store_cap(c,'fair','s',1024)==2048
    assert store_cap(c,'full','a',8192)==8192
    assert store_cap(c,'full','pressure0',16384)==0
    for sid in c['session_tokens']:
        turns=[t for _,s,t,_ in SEQUENCE if s==sid]
        assert turns==sorted(set(turns))
        assert sum(end for _,s,_,end in SEQUENCE if s==sid)==1
