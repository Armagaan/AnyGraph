# Feature: ad-hoc dataset sets in `--dataset_setting`

**Status:** implemented on `dev` in `a0f23eb`, merged as `f8e9a10`.
The code lives in [dataset_groups.py](../dataset_groups.py); this document is the
design record.
**Motivation:** run a pretrain-and-zero-shot-test cycle over an *arbitrary* pair of
dataset sets, rather than only over the six group names hard-coded in
`main.py`'s `__main__` block.

## Why this is needed

`--dataset_setting` currently accepts a group name, a single dataset name, `A+B`,
or `A_in_B`. The `A+B` form resolves **both sides through the group dict**:

```python
elif '+' in args.dataset_setting:
    idx = args.dataset_setting.index('+')
    trn_datasets = datasets[args.dataset_setting[:idx]]
    tst_datasets = datasets[args.dataset_setting[idx+1:]]
```

So `--dataset cora+citeseer` raises `KeyError: 'cora'`: `cora` is a *member* of
`datasets['all']`, not a *key*. There is no way to express "train on these two
graphs, zero-shot test on those two" without editing the dict in the source.

That matters in practice because the smallest existing groups are `academic`
(9 datasets, including the 1M-node `citation-classic`) and `others` (11,
including `ppa` at 45M edges). Any `A+B` smoke test therefore costs ~6-8 minutes
of data loading before the first training step — which is how a broken run gets
discovered late rather than early.

## Proposed grammar

One new concept: a **side** may be a comma-separated list, whose elements are
either dataset names or group names. Everything else is unchanged.

| spec | meaning |
| --- | --- |
| `link1` | group (unchanged) |
| `cora` | single dataset (unchanged) |
| `link1+link2` | train link1, test [link1, link2] (unchanged) |
| `ecommerce_in_link1` | intersection (unchanged) |
| `cora,citeseer` | **new** — train and test on exactly these two |
| `cora,citeseer+pubmed,products_home` | **new** — train on the first pair, test on [first pair, second pair] |
| `academic,cora` | **new** — a group and a dataset mixed; flattened, de-duplicated |

Group semantics are unchanged: under `A+B` the model trains on A and is
evaluated against **two** test groups, A (full-shot) and B (zero-shot), reported
as `Test0` and `Test1`.

## Implementation sketch

Replace the four-branch `if/elif` chain with a resolver applied per side. The
resolver takes the group dict as a parameter so the root and node-classification
entry points can share it despite holding different dicts.

```python
def resolve_side(spec, datasets, known):
    """Resolve one side of a --dataset_setting expression to dataset names.

    A side is a group name, or a comma-separated list of dataset and/or group
    names. Groups are flattened; the result is de-duplicated, order-preserving.
    """
    if spec in datasets:            # whole group, fast path
        return list(datasets[spec])
    out, seen = [], set()
    for part in (p.strip() for p in spec.split(',')):
        if not part:
            continue
        members = datasets[part] if part in datasets else [part]
        for name in members:
            if name not in known:
                raise SystemExit(
                    f"unknown dataset or group '{name}' in --dataset_setting "
                    f"'{spec}'. Groups: {', '.join(sorted(datasets))}."
                )
            if name not in seen:
                seen.add(name)
                out.append(name)
    if not out:
        raise SystemExit(f"--dataset_setting '{spec}' resolved to no datasets")
    return out
```

and at the call site:

```python
known = {name for members in datasets.values() for name in members}

if '_in_' in setting:
    left, right = setting.split('_in_', 1)
    left = resolve_side(left, datasets, known)
    right = resolve_side(right, datasets, known)
    trn = tst = [d for d in left if d in right]
    tst_groups = [tst]
elif '+' in setting:
    a, b = setting.split('+', 1)
    trn = resolve_side(a, datasets, known)
    tst = resolve_side(b, datasets, known)
    tst_groups = [trn, tst]
else:
    trn = tst = resolve_side(setting, datasets, known)
    tst_groups = [tst]

handler = MultiDataHandler(trn, tst_groups)   # + handler_cls in node_classification
```

## Details that will bite the implementer

- **Validate against the union of the dict's values, not `datasets['all']`.**
  The current single-dataset branch tests `args.dataset_setting in datasets['all']`.
  In `node_classification/main.py` that is wrong: `datasets['node']` contains
  `home` and `tech`, which are real directories under `node_data/` but are *not*
  listed in that file's `datasets['all']`. Validating against `all` would reject
  `--dataset home`. Hence the `known` set above.

- **Check `_in_` before `+`.** The current chain tests `+` first. That happens to
  work because no existing group name contains both tokens, but a spec like
  `a,b_in_c` is ambiguous under the old order. `_in_` is the longer, more
  specific token, so match it first, and use `split(sep, 1)` rather than
  `index()` so a stray second separator is a clear error rather than a silent
  truncation.

- **Both entry points need the change.** `main.py` and
  `node_classification/main.py` each carry their own copy of the dict and the
  parse chain, and they already diverge: the node file renames `link1`/`link2`
  to `div1`/`div2` and adds `node`. Put `resolve_side` somewhere importable by
  both (a new small module, or `params.py`) and pass the dict in — do not add it
  to `data_handler.py` near the handler classes without reading the class-order
  warning in CLAUDE.md.

- **De-duplication is already handled downstream.** `MultiDataHandler.__init__`
  does `list(set(trn + all_tst))`, so a dataset named on both sides of a `+` is
  constructed once and shared between the train and test handler lists. Ad-hoc
  sets that overlap will not double-load; no change needed there.

- **This also fixes a documented papercut.** An unrecognised `--dataset_setting`
  currently falls through all four branches and dies with a bare
  `NameError: name 'trn_datasets' is not defined` (the default value `training`
  is itself not a group, so a bare `python main.py` hits this). The resolver
  raises a message naming the bad token and listing the valid groups.

- **Shell quoting.** Commas and `+` need no quoting in bash. Keep the spec free
  of spaces, or it must be quoted.

## Suggested acceptance check

The case that motivated this — a genuine `A+B` train/zero-shot-test cycle that
finishes in minutes rather than hours, while still exercising the multi-dataset
MoE routing where the recent OOM actually surfaced:

```bash
uv run python main.py --gpu 2 \
  --dataset "cora,citeseer+pubmed,products_home" --epoch 1 --seed 42
```

Node counts: cora 2,708 / citeseer 3,327 / pubmed 19,717 / products_home 9,790.

Regression cases that must behave exactly as before:

```bash
uv run python main.py --load pretrain_link1 --epoch 0 --dataset cora
uv run python main.py --load pretrain_link1 --epoch 0 --dataset ecommerce_in_link1
uv run python main.py --dataset link1+link2 --epoch 0 --load pretrain_link1
```
