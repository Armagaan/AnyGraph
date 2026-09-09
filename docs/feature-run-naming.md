# Feature: run directories and provenance

**Status:** implemented on `dev` in `9ccb412`, merged as `ecea7f8`.
The code lives in [run_dir.py](../run_dir.py); this document is the design record.
**Motivation:** make a training run identifiable after the fact, and make it
impossible for one run to silently destroy another.

## Why this is needed

Every artifact a run produces is keyed on the single string `--save_path`, which
defaults to `tem`:

| artifact | path | written when |
| --- | --- | --- |
| final model | `Models/<save_path>.mod` | every test epoch, via `save_history` |
| best model | `Models/<save_path>_best.mod` | whenever test NDCG improves |
| metrics | `History/<save_path>.his` | every test epoch |

Four consequences:

- **Test groups are positional and anonymous.** `.his` keys are `Test0Recall`,
  `Test1NDCG`; nothing records which datasets `Test0` held. The published
  `pretrain_link1.his` says there were two test groups and nothing more. Under
  `--dataset "cora,citeseer+pubmed,products_home"` the name cannot carry it either.
- **The checkpoint records architecture, not provenance.** `describe_model()`
  ([model.py:27](../model.py#L27)) captures `nn`, `fc_layer`, `latdim`, `expert_num`
  and `trn_count` — enough to rebuild the network, which was all B6 needed. It does
  not capture `dataset_setting`, `seed`, `lr`, `epoch`, `trn_mode`, `assignment` or
  `proj_method`. Two runs differing only in `--seed` are indistinguishable on disk.
- **No collision check.** On 2026-09-09 a run with `--save pretrain_link1`
  overwrote the 7.57 GB published baseline with a 67 MB fragment from a fraction of
  one epoch. Recovery needed the B0 archive and the original tarball.
- **`_best.mod` cannot be loaded.** `load_model` opens `History/<name>.his`
  unconditionally — for `self.metrics` and for the resume-epoch computation — and
  the `_best` save never gets one, so the better checkpoint is the unusable one.

## Decisions taken

- **A directory per run, with a manifest.** Filenames are a bad database: a spec
  like `cora,citeseer+pubmed,products_home` does not fit in one, and two different
  specs can resolve to the same dataset sets. The directory name only has to be
  unique and scannable; `manifest.json` carries the truth.
- **Run ids are generated, never absent.** `<local timestamp>_<spec slug>_s<seed>`,
  with `--run_name` replacing the whole thing. Local rather than UTC so the
  directory name agrees with `log.txt`, which TimeLogger stamps in local time;
  the manifest carries `started_utc` for the unambiguous machine-readable form.
- **`Models/` and `History/` are left alone**, so the published baselines keep
  loading exactly as they do now.

## Layout

```
runs/20260909-172200_cora-citeseer+pubmed-products_home_s42/
    manifest.json    args, resolved dataset lists, git sha, environment, timings
    final.mod
    best.mod
    history.pkl      the existing .his pickle, unchanged in format
    metrics.json     final numbers per test group, keyed by dataset name
    log.txt          streamed, not dumped at exit
```

### Run id

`<local timestamp>_<slug>_s<seed>`, e.g. `20260909-172200_link1+link2_s42`.

The slug is `--dataset_setting` with `,` replaced by `-` and anything outside
`[A-Za-z0-9_+-]` dropped. Over 40 characters it is truncated and a 6-hex digest of
the *resolved* dataset lists is appended, so
`ecommerce,cora,arxiv,collab,ddi,...` becomes `ecommerce-cora-arxiv-collab-ddi-a3f1c2`.
`--run_name X` replaces the generated id outright.

Note this drops the `cora2+pubmed2` count shorthand floated in the original
discussion — it is shorter but tells you nothing about which datasets were involved.

### manifest.json

```jsonc
{
  "run_id": "20260909-172200_cora-citeseer+pubmed-products_home_s42",
  "status": "completed",              // running | completed | failed
  "started_utc": "...", "finished_utc": "...",
  "argv": ["main.py", "--dataset", "cora,citeseer+pubmed,products_home", ...],
  "args": { ... every parsed flag ... },
  "datasets": {
    "setting": "cora,citeseer+pubmed,products_home",
    "train": ["cora", "citeseer"],
    "test_groups": [["cora", "citeseer"], ["pubmed", "products_home"]]
  },
  "git": {"commit": "a0f23eb...", "branch": "feat/...", "dirty": false},
  "env": {"python": "3.11.x", "torch": "2.6.0+cu124", "gpus": ["NVIDIA L40S"],
          "host": "...", "cuda_visible_devices": "2"},
  "checkpoint": {"version": 1, "config": { ...describe_model output... }}
}
```

`test_groups` is what de-anonymises `Test0`/`Test1`. **`history.pkl` keeps its
current keys and format** — changing them would break the published `.his` files and
the resume-epoch computation that reads `metrics['TrainLoss']`. The mapping from
index to dataset list lives in the manifest instead.

## Implementation sketch

A new shared module, `run_dir.py`, imported by both entry points the way
`dataset_groups.py` already is:

- `make_run_id(args, resolved)` — slug, truncation, digest, `--run_name` override.
- `create_run_dir(run_id, overwrite=False)` — `SystemExit` if the directory exists
  and `--overwrite` was not passed. This is the clobber guard.
- `write_manifest(path, ...)` / `update_manifest(path, **fields)` — written once at
  start with `status: running`, updated at exit.
- `repo_root()` — resolved from `__file__`, not the cwd.

Then in [main.py](../main.py) and [node_classification/main.py](../node_classification/main.py):

- `save_model` / `save_history` write into the run directory instead of
  `./Models/` and `../Models/`.
- `load_model` resolves three forms: a run directory (`runs/X`), a checkpoint
  inside one (`runs/X/best`), or a legacy bare name (`pretrain_link1` →
  `Models/pretrain_link1.mod`).

### Changes that go beyond renaming

- **A missing history becomes non-fatal.** `load_model` currently dies with
  `FileNotFoundError`. It should warn, start `self.metrics` empty and resume from
  epoch 0. This is what makes `best.mod` loadable, in the new layout and the old one.
- **`log.txt` is streamed.** `Utils/TimeLogger` accumulates every line into a
  module-level `logmsg` string that is **never written anywhere** — dead since
  upstream. Rather than dumping it at exit, add `set_logfile(path)` and append per
  call, so a killed run keeps its log. (The `logs/*.log` files in the repo root come
  from shell redirection, not from this.)
- **`--epoch 0` runs get a directory too.** `save_history` early-returns at epoch 0,
  so zero-shot evaluations currently write nothing at all. They should produce a
  manifest, `metrics.json` and `log.txt` — just no checkpoints. Those runs are the
  main experimental output; losing them is the bigger gap.

### Details that will bite the implementer

- **`node_classification/main.py` keeps its own `Exp`, with its own `save_model`,
  `save_history` and `load_model` writing `../Models/` and `../History/`.** B8
  unified `data_handler.py` and `model.py`, not this. Both copies need the change;
  `repo_root()` resolved from `__file__` removes the `../` asymmetry.
- **Paths are cwd-relative today** and CLAUDE.md documents that as a trap. Resolving
  `runs/` against the repo root fixes it for new artifacts; `Models/` and `History/`
  should be resolved the same way rather than left half-migrated.
- **The clobber guard only bites on `--run_name`.** Generated ids carry a
  second-resolution timestamp, so natural collisions are near-impossible; the guard
  exists for deliberate names.
- **`--save_path` still exists.** Decide whether to keep it as an alias for
  `--run_name` or retire it. Retiring changes the CLI for anyone with scripts;
  keeping it means two flags meaning the same thing.

## Verification

- Two runs with identical flags land in different directories; a third with an
  explicit `--run_name` colliding with an existing one exits without writing.
- `manifest.json` round-trips: `test_groups` matches the `Test group N` banner
  `Exp.__init__` prints, and `args.seed`/`dataset_setting` match what was passed.
- `--load runs/<id>` and `--load runs/<id>/best` both restore and resume; the
  legacy `--load pretrain_link1` still produces cora ≈ 0.822 / 0.66.
- A run killed mid-epoch leaves `log.txt` populated and `status: running` in the
  manifest.
- `--epoch 0 --dataset cora` produces a directory with a manifest, metrics and log,
  and no `.mod`.
