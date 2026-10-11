# QUICK START — From zip to trained LWLM in 6 steps

Read this first. It is the short, ordered "do-this-then-that" companion to the
full `README.md`. Total reading time: 3 minutes.

---

## Step 0 — Unzip and place the folder

Unzip anywhere you like. The project is self-contained and every path inside it
is anchored to the project root (the folder that contains `lwlm/`), so it does
NOT need to sit next to anything special:

```
 anywhere/                      ← your choice
└── LWLM/                       ← the unzipped project root
    ├── LWLM_pipeline.ipynb
    ├── README.md
    ├── QUICK_START.md          ← you are here
    ├── requirements.txt
    ├── lwlm/                   ← the Python package (do not modify)
    ├── scripts/                ← CLI entry points
    ├── Dataset/                ← ★ the ONLY folder you fill in ★
    ├── checkpoints/  results/  logs/   ← outputs (auto-created)
    └── processed/              ← created at runtime by Step 3
```

The only rule: **keep the folder layout as-is** (never move `lwlm/` out of
`LWLM/`, never run scripts by copying them elsewhere).

> Advanced: if you must keep the code somewhere else from your data, set the
> environment variable `LWLM_ROOT` to the project root path before launching.

---

## Step 1 — Put YOUR DeepMIMO data in (2 minutes)

1. Download the **O1_3p5** scenario (`.mat` files) from
   <https://deepmimo.net/scenarios/o1-scenario/> (you already have it).
2. Drop the files into `LWLM/Dataset/`. Both layouts are auto-detected:

```
LWLM/Dataset/O1_3p5/*.mat      ← extracted scenario folder (preferred)
LWLM/Dataset/*.mat             ← loose .mat files directly in Dataset/
```

Nothing else to configure — no `.npy` index files to make, no paths to edit.
**No dataset ships with this zip; the code only reads yours.**

Sanity check: `Dataset/` must contain files like `O1_3p5.mat`-style DeepMIMO
parameter/channel files (typically a few hundred MB to a few GB).

---

## Step 2 — Install the environment (5 minutes)

Python ≥ 3.10 recommended.

```bash
cd LWLM
pip install -r requirements.txt
```

Then verify torch sees your hardware:

```bash
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())"
```

- `cuda: True` → you will train on GPU (recommended).
- `cuda: False` → everything still runs on CPU, but use `SMALL_DEBUG = True`
  (Step 4) — paper-scale training on CPU is not practical.

---

## Step 3 — ⚠️ BEFORE-RUN ADJUSTMENT CHECKLIST ⚠️

Open the notebook `LWLM_pipeline.ipynb` (or `scripts/*.py`) and go through this
list **once** before pressing Run All:

| # | What to adjust | Where | Default (paper scale) | When to change |
|---|----------------|-------|----------------------|----------------|
| 1 | **`SMALL_DEBUG`** | notebook config cell (cell 4) | `False` | **First run ever: set `True`** — 1 pretrain epoch, 2 fine-tune epochs, 200 samples, ~minutes instead of days. Flip back to `False` for the real run. |
| 2 | **`--user_rows`** (data conversion size) | notebook Step 1 cell / `scripts/generate_data.py` | `100` rows ≈ 20 000 UE | Start with `10–20` for a smoke test. `2700` = the full O1_3p5 grid. Bigger = more data + hours of conversion. |
| 3 | **Epochs** | notebook config cell | pretrain `140`, fine-tune `1000` | These are the original paper-scale values. Reduce for your own experiments; keep for reproducing the paper. |
| 4 | **`ft_dataset`** | notebook config cell | `10000` | Number of labelled UE used per fine-tuning task. Must be ≤ your converted pool size. |
| 5 | **`test_samples`** | notebook config cell | `5400` | Auto-capped if your test pool is smaller. |
| 6 | **`num_workers`** | notebook config cell | `0` on Windows, `4` elsewhere | Keep `0` on Windows (Jupyter spawn issues) or if you hit DataLoader crashes. |
| 7 | **GPU / accelerator** | nothing to edit | `accelerator="auto"` | Fixed in this refactor — auto-detects CPU/GPU. No code change needed (the original repo crashed here). |
| 8 | **Batch size** | notebook config cell | `32` | Lower it (8/16) if you hit CUDA out-of-memory. |

**Recommended first pass:** `SMALL_DEBUG = True` + `user_rows = 10`
→ end-to-end run in ~10–30 min → confirms your environment and data are OK →
then set `SMALL_DEBUG = False` and convert the full dataset for the real run.

---

## Step 4 — Run it

### Option A — Notebook (recommended)

```bash
cd LWLM
jupyter lab LWLM_pipeline.ipynb      # or: jupyter notebook LWLM_pipeline.ipynb
```

Run cells **top to bottom** (Kernel → Restart & Run All). The pipeline:

```
Step 1  convert Dataset/*.mat → processed/O1_3p5/*.h5py + index .npy   (skips if done)
Step 2  visualize one CSI sample (amplitude / delay–angle maps)
Step 3  PRETRAIN  — pretrain_mix (SF-MCM + DTI + PICL)   → checkpoints/pretrain_mix/
Step 4  FINE-TUNE — SingleBSLoc                          → checkpoints/SingleBSLoc/
Step 5  FINE-TUNE — MultiBSLoc (4-BS attention fusion)   → checkpoints/MultiBSLoc/
Step 6  FINE-TUNE — toa and aoa                          → checkpoints/toa/, checkpoints/aoa/
Step 7  TEST      — mean/median/P90 errors + CDF plots   → results/
```

### Option B — Command line (identical CLI args to the original repo)

```bash
cd LWLM

# 1) convert dataset (one-time; resumable / skips finished parts)
python scripts/generate_data.py --user_rows 100

# 2) pretrain
python scripts/train.py --task pretrain_mix \
    --model_path checkpoints/pretrain_mix --is_frozen 0 --is_load 0 --depth 4

# 3) fine-tune each task (example: SingleBSLoc; load the pretrain ckpt)
python scripts/train.py --task SingleBSLoc \
    --model_path checkpoints/SingleBSLoc --is_frozen 0 --is_load 1 --depth 4 \
    --load_pretrained_mdl_path checkpoints/pretrain_mix/testbest.ckpt \
    --FT_dataset 10000 --BW 10 --pilot_subcarrier_interval 1 \
    --pilot_antenna_interval 1 --BS_Num 4

# 4) test (repeat with the matching script per task)
python scripts/test_singleBS.py --ckpt checkpoints/SingleBSLoc/testbest.ckpt
python scripts/test_multiBS.py  --ckpt checkpoints/MultiBSLoc/testbest.ckpt
python scripts/test_toa.py      --ckpt checkpoints/toa/testbest.ckpt
python scripts/test_aoa.py      --ckpt checkpoints/aoa/testbest.ckpt
```

Checkpoint names: best epoch is saved as `testbest-epoch=XXX.ckpt`
(monitors `val/ave_loss`); the latest epoch as `test_latest.ckpt`.

---

## Step 5 — What you get

| Output | Location |
|--------|----------|
| Converted per-sample dataset + index splits | `processed/` (~4–6 GB at paper scale) |
| Pretrained + 4 fine-tuned checkpoints | `checkpoints/<task>/testbest-epoch=XXX.ckpt` |
| Localization error metrics (mean / median / P90) | printed in notebook Step 7 + `results/*.txt` |
| Paper-style CDF error curves | `results/*.png` |
| TensorBoard training curves | `logs/` → `tensorboard --logdir logs` |

---

## Step 6 — Troubleshooting (fast answers)

| Symptom | Cause → Fix |
|---------|-------------|
| `Dataset folder not found` / `No .mat files found` | Step 1 not done → put `.mat` files in `LWLM/Dataset/` |
| `ModuleNotFoundError: lwlm` | You launched the notebook/script from outside the project → `cd LWLM` first (paths are root-anchored, but `jupyter` must see the folder) |
| Very slow first cell (Step 1) | Normal — DeepMIMO ray-tracing conversion of ~20k UE takes hours. Use `--user_rows 10` to verify first. |
| `CUDA out of memory` | Lower `batch_size` (32 → 16 → 8) in the config cell |
| DataLoader worker crashes (esp. Windows) | Set `num_workers = 0` in the config cell |
| `torch.cuda.is_available() == False` | CPU-only torch installed → either accept CPU + `SMALL_DEBUG=True`, or reinstall the CUDA build of torch per pytorch.org |
| Re-running Step 1 re-converts everything? | No — conversion is skipped per (BS, BW) folder that already exists; delete `processed/` only if you changed `user_rows`/data |

For the full background — what was broken in the original repo, the exact
data layout produced, and the citation — see `README.md`.
