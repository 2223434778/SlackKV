import json
from pathlib import Path, PurePosixPath
import tarfile

root = Path.cwd()/'results/host_prefix_followup_20261010'
failures = root/'tool_failures'
failures.mkdir(parents=True, exist_ok=True)
(root/'analysis.json').rename(failures/'invalid_empty_analysis_attempt1.json')
(failures/'extract_attempt1.json').write_text(json.dumps(dict(
    status='LOCAL_TOOL_FAILURE', error='tarfile data-filter os.path.realpath PermissionError WinError5',
    secondary_error='following analysis command wrote empty runs/gpu_hours0 before extraction completed; INVALID',
    correction='reject empty analysis; validate exported regular-file paths lexically and copy bytes without tar metadata or links',
    gpu_hours=0), indent=2)+'\n')
with tarfile.open('.tmp/host-prefix-followup-evidence.tar.gz') as archive:
    for member in archive:
        path = PurePosixPath(member.name)
        assert member.isfile() and not path.is_absolute()
        assert path.parts and all(p not in ('..', '.') and ':' not in p and '\\' not in p for p in path.parts)
        target = root.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
print('validated archive extracted', len(list(root.rglob('*'))))
