"""Run directories: where a run's artifacts go, and what identifies it.

Every artifact used to be keyed on the single string `--save_path` (default
`tem`), so two runs differing only in `--seed` were indistinguishable on disk and
nothing stopped one from overwriting another -- which is how a published 7.57 GB
baseline once became a 67 MB fragment.

A run now owns a directory under `runs/`:

    runs/<run_id>/manifest.json   args, resolved dataset lists, git sha, env
                  final.mod       last checkpoint
                  best.mod        best test NDCG so far
                  history.pkl     the old .his pickle, format unchanged
                  metrics.json    final numbers, keyed by dataset name
                  log.txt         streamed as the run goes

`Models/` and `History/` are left alone so the published baselines keep loading.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

RUNS_DIRNAME = 'runs'
MANIFEST_NAME = 'manifest.json'
HISTORY_NAME = 'history.pkl'
METRICS_NAME = 'metrics.json'
LOG_NAME = 'log.txt'
FINAL_STEM = 'final'
BEST_STEM = 'best'
SLUG_MAX = 40


def repo_root():
    """The repository root, resolved from this file rather than the cwd.

    node_classification/main.py runs from its own directory and used to reach
    the shared directories as '../Models/'; everything here is absolute so both
    entry points agree regardless of where they were started.
    """
    return os.path.dirname(os.path.abspath(__file__))


def runs_root():
    return os.path.join(repo_root(), RUNS_DIRNAME)


def legacy_path(kind, name, suffix):
    """A path in the pre-run-directory layout, e.g. Models/pretrain_link1.mod."""
    return os.path.join(repo_root(), kind, name + suffix)


def slugify(setting, resolved):
    """A filename-safe, human-scannable form of --dataset_setting.

    Commas become hyphens so `cora,citeseer+pubmed` reads as
    `cora-citeseer+pubmed`. Long specs are truncated and given a digest of the
    *resolved* dataset lists, because two different specs can resolve to the
    same sets and the truncated text alone would not distinguish them.
    """
    slug = re.sub(r'[^A-Za-z0-9_+.-]', '', setting.replace(',', '-'))
    if len(slug) > SLUG_MAX:
        digest = hashlib.sha1(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:6]
        slug = slug[:SLUG_MAX].rstrip('-+.') + '-' + digest
    return slug or 'unnamed'


def make_run_id(args, resolved):
    """`<timestamp>_<slug>_s<seed>`, or --run_name verbatim.

    Local time, not UTC: TimeLogger stamps every line in local time, so a UTC
    directory name would contradict the log it contains. The manifest records
    `started_utc` for the unambiguous machine-readable version.
    """
    if getattr(args, 'run_name', None):
        return args.run_name
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return '%s_%s_s%d' % (stamp, slugify(args.dataset_setting, resolved), args.seed)


def create_run_dir(run_id, overwrite=False):
    """Make the run directory, refusing to reuse one that already exists.

    Generated ids carry a second-resolution timestamp, so this only really bites
    on an explicit --run_name -- which is exactly the case worth protecting.
    """
    path = os.path.join(runs_root(), run_id)
    if os.path.exists(path) and not overwrite:
        raise SystemExit(
            "run directory already exists: %s\n"
            "Pass --run_name with a different name, or --overwrite to reuse it." % path
        )
    os.makedirs(path, exist_ok=True)
    return path


def git_info():
    def git(*cmd):
        try:
            out = subprocess.run(('git',) + cmd, cwd=repo_root(), capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None
    return {
        'commit': git('rev-parse', 'HEAD'),
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(git('status', '--porcelain')),
    }


def env_info():
    info = {
        'python': platform.python_version(),
        'host': platform.node(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    try:
        import torch as t
        info['torch'] = t.__version__
        info['gpus'] = [t.cuda.get_device_name(i) for i in range(t.cuda.device_count())]
    except Exception:
        pass
    return info


def _jsonable(obj):
    """Coerce numpy scalars and anything else exotic into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    item = getattr(obj, 'item', None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(obj)


def write_json(path, payload):
    with open(path, 'w') as fs:
        json.dump(_jsonable(payload), fs, indent=2, sort_keys=False)
        fs.write('\n')


def start_manifest(run_path, run_id, args, trn_datasets, tst_datasets_group):
    payload = {
        'run_id': run_id,
        'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'finished_utc': None,
        'argv': sys.argv,
        'datasets': {
            'setting': args.dataset_setting,
            'train': list(trn_datasets),
            'test_groups': [list(group) for group in tst_datasets_group],
        },
        'git': git_info(),
        'env': env_info(),
        'args': vars(args),
    }
    write_json(os.path.join(run_path, MANIFEST_NAME), payload)
    return payload


def update_manifest(run_path, **fields):
    path = os.path.join(run_path, MANIFEST_NAME)
    try:
        with open(path) as fs:
            payload = json.load(fs)
    except Exception:
        payload = {}
    payload.update(fields)
    write_json(path, payload)
    return payload


def resolve_checkpoint(spec):
    """Find the checkpoint and history `--load <spec>` refers to.

    Accepts a run directory (`runs/X`, or bare `X` under runs/), a checkpoint
    inside one (`runs/X/best`), or a legacy bare name (`pretrain_link1`, i.e.
    Models/pretrain_link1.mod plus History/pretrain_link1.his).

    Returns (mod_path, history_path); history_path may not exist, which callers
    must tolerate -- `_best` checkpoints never had one.
    """
    def in_run_dir(directory, stem):
        return (os.path.join(directory, stem + '.mod'),
                os.path.join(directory, HISTORY_NAME))

    for base in (os.path.join(runs_root(), spec),
                 spec if os.path.isabs(spec) else os.path.join(repo_root(), spec)):
        if os.path.isdir(base):
            return in_run_dir(base, FINAL_STEM)
        if os.path.exists(base + '.mod'):
            return in_run_dir(os.path.dirname(base), os.path.basename(base))

    return legacy_path('Models', spec, '.mod'), legacy_path('History', spec, '.his')
