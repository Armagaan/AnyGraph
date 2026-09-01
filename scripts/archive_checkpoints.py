"""Archive the published .mod checkpoints as plain state dicts.

The released checkpoints pickle whole ``nn.Module`` objects, so they only load
while the class names and module paths in ``model.py`` stay exactly as they are.
Any rename, move, or package restructure silently orphans them -- and the
authors' OneDrive mirror has already expired once, so they are not
re-downloadable at will.

This script converts each checkpoint into a state dict plus the configuration
needed to rebuild the model, and records a SHA256 of the source file so the
copy can be proven faithful later.

    python scripts/archive_checkpoints.py            # convert
    python scripts/archive_checkpoints.py --verify   # re-check against sources

Run it BEFORE refactoring model.py.
"""
import argparse
import hashlib
import json
import os
import sys

import torch as t

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

ARCHIVE_VERSION = 1


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fs:
        for block in iter(lambda: fs.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def describe(model):
    """Recover the construction-time configuration from a loaded AnyGraph."""
    experts = list(model.experts)
    first = experts[0].trainable_nn
    kind = type(first).__name__

    if kind == 'MLP':
        layers = list(first.dense_layers)
        latdim = layers[0].linear.weight.shape[0]
        cfg = {'nn': 'mlp', 'fc_layer': len(layers)}
    elif kind == 'GraphTransformer':
        layers = list(first.gt_layers)
        latdim = layers[0].multi_head_attention.embed_dim
        cfg = {'nn': 'gt', 'gt_layer': len(layers),
               'head': layers[0].multi_head_attention.num_heads}
    else:
        raise SystemExit('Unrecognised trainable_nn: ' + kind)

    cfg['latdim'] = int(latdim)
    cfg['expert_num'] = len(experts)
    # trn_count drives the load-balancing factor in assign_experts(); it is a
    # plain attribute, so state_dict() would drop it.
    cfg['trn_count'] = [int(getattr(e, 'trn_count', 1)) for e in experts]
    return cfg


def load_module(path):
    # Unpickling imports model.py, which imports params.py, which calls
    # parse_args() at module scope against sys.argv. Hide our own flags from it.
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        # weights_only=True (the torch>=2.6 default) cannot unpickle an nn.Module.
        ckp = t.load(path, map_location='cpu', weights_only=False)
    finally:
        sys.argv = saved_argv
    return ckp['model']


def convert(mod_path, out_path):
    print('reading  ' + mod_path)
    digest = sha256(mod_path)
    model = load_module(mod_path)
    cfg = describe(model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    t.save({
        'archive_version': ARCHIVE_VERSION,
        'state_dict': state,
        'config': cfg,
        'source_file': os.path.basename(mod_path),
        'source_sha256': digest,
    }, out_path)

    entry = {
        'source_file': os.path.basename(mod_path),
        'source_sha256': digest,
        'source_bytes': os.path.getsize(mod_path),
        'archive_file': os.path.basename(out_path),
        'archive_bytes': os.path.getsize(out_path),
        'tensor_count': len(state),
        'param_count': int(sum(v.numel() for v in state.values())),
        'config': cfg,
    }
    print('  -> {archive_file}  ({param_count:,} params in {tensor_count} tensors, '
          '{archive_bytes:,} bytes vs {source_bytes:,})'.format(**entry))
    return entry


def verify(mod_path, out_path):
    print('verifying ' + os.path.basename(out_path))
    archive = t.load(out_path, map_location='cpu', weights_only=False)
    if archive['source_sha256'] != sha256(mod_path):
        raise SystemExit('  FAIL: source file changed since it was archived')

    original = load_module(mod_path).state_dict()
    stored = archive['state_dict']
    if set(original) != set(stored):
        raise SystemExit('  FAIL: tensor names differ')
    for k in original:
        if not t.equal(original[k].detach().cpu(), stored[k]):
            raise SystemExit('  FAIL: tensor differs: ' + k)
    print('  OK: {} tensors identical'.format(len(stored)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models-dir', default=os.path.join(REPO_ROOT, 'Models'))
    ap.add_argument('--verify', action='store_true',
                    help='re-check existing archives against their sources')
    args = ap.parse_args()

    mods = sorted(f for f in os.listdir(args.models_dir) if f.endswith('.mod'))
    if not mods:
        raise SystemExit('No .mod files in ' + args.models_dir +
                         '. Download them from https://huggingface.co/hkuds/AnyGraph')

    manifest_path = os.path.join(args.models_dir, 'archive_manifest.json')
    entries = []
    for name in mods:
        mod_path = os.path.join(args.models_dir, name)
        out_path = os.path.join(args.models_dir, name[:-len('.mod')] + '.state.pt')
        if args.verify:
            if not os.path.exists(out_path):
                raise SystemExit('Missing archive: ' + out_path + ' (run without --verify first)')
            verify(mod_path, out_path)
        else:
            entries.append(convert(mod_path, out_path))

    if not args.verify:
        with open(manifest_path, 'w') as fs:
            json.dump({'archive_version': ARCHIVE_VERSION, 'checkpoints': entries}, fs, indent=2)
        print('manifest written: ' + manifest_path)


if __name__ == '__main__':
    main()
