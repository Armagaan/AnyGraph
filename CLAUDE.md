# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of **AnyGraph** (arXiv 2408.10700), a graph foundation model doing zero-shot link prediction / node classification across 30+ heterogeneous graph datasets via a Mixture-of-Experts of graph experts. Research code: no test suite, no linter, everything runs through `main.py`.

The fork diverges from upstream `c2c5bfe` by a pre-experimentation cleanup (branches `fix/*`, `chore/*`, `refactor/*`, all merged into `dev`). `main` is a pristine upstream mirror — never commit to it; cut upstream-candidate branches from it and merge them into `dev` to run.

## Environment and setup

```bash
uv sync                                              # provisions python + deps
export ANYGRAPH_DATA="/path/to/zero-shot datasets"   # or pass --data_dir
```

torch 2.6.0+cu124, numpy 2.x, scipy 1.18, scikit-learn, setproctitle. **No `torch_geometric`** — it was imported but never used. Do not re-pin to the README's original torch 1.13: the hardware here is Ada (sm_89), which torch 1.13 predates.

Datasets and pretrained `.mod` checkpoints come from HuggingFace (`hkuds/AnyGraph_datasets`, `hkuds/AnyGraph`). The OneDrive link in the original README is dead.

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

Group names are defined inline in the `__main__` block of [main.py](main.py) (and again in [node_classification/main.py](node_classification/main.py), which adds `node`). Parsed four ways:

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

**Task split.** `DataHandler` (link prediction) and `NodeDataHandler` (node classification) differ only in `get_data_files()` and `load_data()`; the seams are the `data_subdir` / `step_sample_target` class attributes and the `fewshot_filename()`, `set_shapes()`, `finalize_loading()` methods. `MultiDataHandler` takes the handler class to build. `node_classification/` holds only `main.py`, which puts the repo root on `sys.path`.

## Things that will bite you

- **Runs are not bitwise reproducible, and cannot be.** `--seed` fixes negative sampling, per-step downsampling, dropout, anchor selection and loader shuffling, but `t.svd_lowrank` on a sparse adjacency is nondeterministic on CUDA even with an identical seed and input. CPU is exact but 8–48× slower, scaling with graph size, so it is not used. This is largely intended (see §3.3.1). **Read every result as mean ± std over the repeat evaluations the code already performs** — spread is ~0.002 NDCG on cora, but F1 on node classification swings by ~0.03.
- **Two checkpoint formats.** New saves are `{version, config, state_dict}`. The published `.mod` files pickle the whole `nn.Module` and still load through a legacy path in `load_model`, which logs that re-saving migrates them. Those pickles record `model.Expert`, so **moving model classes to another module orphans them** — the archives under `Models/*.state.pt` plus `archive_manifest.json` are the insurance. Checkpoint config (`latdim`, `expert_num`, …) overrides the CLI on load and says so, because the projectors read `args.latdim` as the SVD rank.
- `Expert.trn_count` is a plain attribute, not a buffer, so `state_dict()` drops it. It drives routing load-balancing and is carried in the checkpoint config.
- **`args` still holds run configuration globally**, but per-dataset shapes (`node_num`, `user_num`, `item_num`) now live on the handler. Don't put dataset-specific state back on `args`.
- `--tst_mode`/`--trn_mode` must pair as `tst`+`train-all` or `val`+`fewshot`; validated at argparse time.
- `node_classification/main.py` reads and writes `../Models/` and `../History/` because it runs from its own directory.
