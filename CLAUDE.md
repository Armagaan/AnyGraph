# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of **AnyGraph** (arXiv 2408.10700), a graph foundation model doing zero-shot link prediction / node classification across 30+ heterogeneous graph datasets via a Mixture-of-Experts of graph experts. Research code: no test suite, no linter, everything runs through `main.py`.

The fork diverges from upstream `c2c5bfe` by a pre-experimentation cleanup (branches `fix/*`, `chore/*`, `refactor/*`, all merged into `dev`). `main` is a pristine upstream mirror — never commit to it; cut upstream-candidate branches from it and merge them into `dev` to run.

All nine cleanup branches are pushed to `origin` and merged into `dev`. **No PRs have been opened** — the five `fix/*` and `chore/*` branches are shaped as upstream candidates and are waiting on manual review. `main` is 0/0 against `upstream/main`.

## Environment and setup

```bash
uv sync                                              # provisions python + deps
export ANYGRAPH_DATA="/path/to/zero-shot datasets"   # or pass --data_dir
```

torch 2.6.0+cu124, numpy 2.x, scipy 1.18, scikit-learn, setproctitle. **No `torch_geometric`** — it was imported but never used. Do not re-pin to the README's original torch 1.13: the hardware here is Ada (sm_89), which torch 1.13 predates.

Datasets and pretrained `.mod` checkpoints come from HuggingFace (`hkuds/AnyGraph_datasets`, `hkuds/AnyGraph`). The OneDrive link in the original README is dead.

## Local state on this machine

None of this is in git, and none of it is reconstructible from the repo:

- **Data root**: `/mnt/data1/burouj/anygraph-data/zero-shot datasets` — 34 dataset directories plus `node_data/` (5 more), 9.3 GB. The path contains a space, so always quote it:
  `export ANYGRAPH_DATA="/mnt/data1/burouj/anygraph-data/zero-shot datasets"`
- **`Models/` mixes tracked and untracked files.** Tracked: `README.md`, `archive_manifest.json`. Untracked and gitignored: `pretrain_link{1,2}.mod` (7.57 GB and 3.00 GB — the published legacy pickles) and `pretrain_link{1,2}.state.pt` (67.6 MB each — state-dict archives from `scripts/archive_checkpoints.py`). The manifest records each pickle's SHA256, so `--verify` can still prove the archives match their sources.
- The original downloads sit in the repo root, gitignored: `AnyGraph pretrained models .tar.gz` (9.5 GB) and `zero-shot datasets.zip` (5.0 GB). Both have been extracted and verified, so they are safe to delete to reclaim ~14.5 GB.
- **`.git/info/exclude`** (local, never committed) carries `Models/*.mod`, `Models/*.pt`, `*.tar.gz`, `*.zip`, `.venv/`, `__pycache__/`, `anygraph.pdf`. It has to live there rather than in `.gitignore` because branches cut from `main` predate the `.gitignore` update — on such a branch **`git add -A` will try to hash ~25 GB and hang**. Stage explicit paths when working on a branch cut from `main`.

## Reference numbers

Zero-shot, `--seed 42`, for spotting regressions after a change. Approximate by nature — see the reproducibility note below; treat a deviation as real only if it exceeds the spread the run itself reports.

| run | expect |
| --- | --- |
| `--load pretrain_link1 --epoch 0 --dataset cora` | Recall@20 ≈ 0.822, NDCG@20 ≈ 0.66 |
| `--load pretrain_link1 --epoch 0 --dataset ml1m` | Recall@20 ≈ 0.099, NDCG@20 ≈ 0.157 |
| node: `--load pretrain_link2 --epoch 0 --dataset node` | Acc ≈ 0.65, macro-F1 ≈ 0.44 |

A full-group run (`--dataset link2`) takes hours; use a single small dataset for smoke tests.

The final evaluation block re-runs each test dataset **5 times**, calling `make_projectors()` afresh each time, and logs `Recall_mean`/`Recall_std` (`Acc_mean`/`F1_std` for node classification). Those are the numbers to compare — see the reproducibility note below.

## Commands

All commands take `--data_dir` or read `ANYGRAPH_DATA`.

```bash
# Zero-shot link prediction: train on one group, test on the other
uv run python main.py --load pretrain_link1 --epoch 0 --dataset link2
uv run python main.py --load pretrain_link2 --epoch 0 --dataset ecommerce_in_link1

# Re-train from scratch
uv run python main.py --dataset link2+link1 --save pretrain_link2 --seed 42

# Node classification (same modules, separate entry point)
cd node_classification && uv run python main.py --load pretrain_link2 --epoch 0 --dataset node

# Convert the published pickled checkpoints to state dicts
uv run python scripts/archive_checkpoints.py [--verify]
```

`--dataset`, `--load`, `--save` are argparse *prefix abbreviations* of `--dataset_setting`, `--load_model`, `--save_path`. `--epoch 0` means test-only: the training loop is skipped, `save_history()` early-returns, and only the final repeated-evaluation block runs. `--gpu 0,1` splits the topology encoder (device 0) from the trainable net (device 1); more than 2 raises.

## `--dataset_setting` grammar

Group names are defined inline in the `__main__` block of [main.py](main.py), and **again, differently**, in [node_classification/main.py](node_classification/main.py): that copy adds `node` and renames `link1`/`link2` to `div1`/`div2` (same member lists). So `--dataset link2` works at the root and dies in `node_classification/`. Parsed four ways:

- a group name (`link1`, `ecommerce`, `all`, …) → train and test on the same set
- a single dataset name from `all` → that dataset only
- `A+B` → train on group A, test on **two** groups: A (full-shot) and B (zero-shot)
- `A_in_B` → the intersection of groups A and B, used for both train and test

## Architecture

The central idea: **every dataset is projected into one shared 512-d space before any learned parameter touches it**, so a single expert can serve graphs with completely different feature spaces and node counts.

**Projection pipeline** (per dataset, recomputed periodically), in `DataHandler.make_projectors()`:
1. `Adj_Projector` — low-rank SVD of the adjacency (`t.svd_lowrank`, `q=latdim`). Asymmetric (bipartite) graphs concatenate `u` and `v`; symmetric ones add them.
2. `Feat_Projector` — the same SVD on raw node features, when a `feats.pkl` exists.
3. The two are summed, then passed through `TopoEncoder`: a parameter-free `gnn_layer`-step sparse propagation summing all hops. Output is `handler.projectors`, a CPU tensor.

These projectors are **not learned**. They are recomputed every `handler.reproj_steps` steps and at each epoch end (`remake_initial_projections`), which is why SVD dominates runtime — and which is deliberate: §3.3.1 of the paper treats re-projection as feature/structure augmentation. Both projector classes fall back to a chunked path on OOM.

**Expert / MoE.** `Expert` = frozen `TopoEncoder` + a trainable `MLP` (or `GraphTransformer` under `--nn`). `AnyGraph` holds `expert_num` experts, each with its **own Adam optimizer** (`self.opts`); only the selected expert's optimizer steps on a given batch. Routing (`assign_experts`) is training-free: each expert scores each dataset via `Expert.attempt()` — mean sigmoid margin between positive and negative edges — and the argmax wins. With `reca=True` the score is scaled by a load-balancing factor from each expert's `trn_count`. `--assignment` selects `top1` (default), `top2` (coin-flip between near-ties), or `one-graph-one-expert`.

**Joint training loop.** `JointTrnData` pre-slices every dataset into `args.batch`-sized index ranges and shuffles across datasets, so the outer DataLoader uses `batch_size=1` and each "batch" is really one dataset's chunk plus its `dataset_id` — hence the `ancs[0]`, `poss[0]` unwrapping in `train_epoch`. Large datasets are downsampled per step by `ratio_500_all` so no single dataset dominates an epoch.

**Prediction.** Everything is a dot product in the shared space. Link prediction scores anchors against the last `cand_size` embeddings with training edges masked (`pred_for_test`); node classification treats class embeddings as the trailing rows and takes an argmax (`pred_for_node_test`). At test time `rerun_embed=True` only on the first batch; the embedding table is cached on the expert as `self.final_embeds`.

**Task split.** `NodeDataHandler` overrides exactly two methods of `DataHandler` — `fewshot_filename()` and `load_data()` — plus the `data_subdir` (`node_data`) and `step_sample_target` (250 vs 500) class attributes. `get_data_files()`, `set_shapes()` and `finalize_loading()` are *shared*: they are the seams the two `load_data()` bodies were factored around, not overrides. `MultiDataHandler` takes the handler class to build. `node_classification/` holds only `main.py` (its `Utils/` is a stale `__pycache__`), which puts the repo root on `sys.path`.

## Things that will bite you

- **Runs are not bitwise reproducible, and cannot be.** `--seed` fixes negative sampling, per-step downsampling, dropout, anchor selection and loader shuffling, but `t.svd_lowrank` on a sparse adjacency is nondeterministic on CUDA even with an identical seed and input. CPU is exact but 8–48× slower, scaling with graph size, so it is not used. This is largely intended (see §3.3.1). **Read every result as mean ± std over the repeat evaluations the code already performs** — spread is ~0.002 NDCG on cora, but F1 on node classification swings by ~0.03.
- **Two checkpoint formats.** New saves are `{version, config, state_dict}`. The published `.mod` files pickle the whole `nn.Module` and still load through a legacy path in `load_model`, which logs that re-saving migrates them. Those pickles record `model.Expert`, so **moving model classes to another module orphans them** — the archives under `Models/*.state.pt` plus `archive_manifest.json` are the insurance. Checkpoint config (`latdim`, `expert_num`, …) overrides the CLI on load and says so, because the projectors read `args.latdim` as the SVD rank.
- `Expert.trn_count` is a plain attribute, not a buffer, so `state_dict()` drops it. It drives routing load-balancing and is carried in the checkpoint config.
- **`args` still holds run configuration globally**, but per-dataset shapes (`node_num`, `user_num`, `item_num`) now live on the handler. Don't put dataset-specific state back on `args`.
- `--tst_mode`/`--trn_mode` must pair as `tst`+`train-all` or `val`+`fewshot`; validated at argparse time.
- `node_classification/main.py` reads and writes `../Models/` and `../History/` because it runs from its own directory.
- **All checkpoint/history paths are relative to the cwd** (`./Models/`, `./History/`), so root runs must start from the repo root — there is no path resolution against `__file__`. `--data_dir`/`ANYGRAPH_DATA` is the only path that is cwd-independent.
- **`--load X` also needs `History/X.his`.** `load_model` opens it unconditionally to restore `self.metrics` and to compute the resume epoch. A checkpoint without its history file raises `FileNotFoundError` — which is exactly the case for the `_best` saves below.
- **Training silently writes a second checkpoint.** Whenever the test NDCG (Acc for node classification) improves, `run()` saves `<save_path>_best.mod` — selected on the **last** test group, i.e. the zero-shot group under `A+B`. It gets no `.his`, so it cannot be `--load`ed without one being supplied. `--save X` therefore produces both `X.mod` and `X_best.mod`.
- **`--dataset_setting` defaults to `training`, which is not a group in either `datasets` dict.** All four parse branches fall through and `main.py` dies with a bare `NameError: name 'trn_datasets' is not defined`. Always pass `--dataset`.
- **`--gpu N` works, but torch's error messages will not agree.** `main.py` sets `CUDA_VISIBLE_DEVICES` from `--gpu` before any CUDA context exists, so `cuda:0` is *always* correct — it is the remapped device, not physical GPU 0. An OOM under `--gpu 2` still says "GPU 0 has a total capacity of...", because torch reports the visible index. Check `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory` against `--query-gpu=index,uuid` to see where a process really is. (GPUs 0 and 1 on this box are usually held by another user's ollama, ~29 GB each; GPU 2 is the free one.)
- **`--attempt_cache` is a memory cliff.** `Expert.attempt()` gathers three `attempt_cache x latdim` float tensors at once to score a dataset during routing, so peak cost is ~3x that product. The upstream default of 1e7 asks for ~61 GB at `latdim=512` and cannot fit on any current GPU; it is now 1e6. The OOM retry loop halves the sample and frees the failed gathers first — if you see `attempt: out of memory, retrying with N samples`, lower the default rather than relying on the climb-down.
- **Class order in [data_handler.py](data_handler.py) is load-bearing.** `NodeDataHandler` must sit after *every* `DataHandler` method, including `make_projectors`. Inserting it mid-class silently rehomes the methods below it onto the subclass, and `DataHandler` loses them — this bit once during the unification, surfacing only as `AttributeError` at runtime.
