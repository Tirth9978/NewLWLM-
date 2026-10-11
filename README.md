# LWLM: Large Wireless Localization Model — Restructured & Fixed

Clean, runnable reorganization of the official **LWLM** repository
([guangjinpan/LWLM](https://github.com/guangjinpan/LWLM)) for the paper
*"Large Wireless Localization Model (LWLM): A Foundation Model for Positioning in 6G Networks"* (arXiv:2505.10134).

> The scientific code (model, data reading, transformations, losses) is **unchanged** —
> this refactor only fixes the broken plumbing so that everything actually runs:
> proper package structure, a single end-to-end notebook, a one-command dataset
> converter that reads **your** DeepMIMO `.mat` files from `Dataset/`, and all
> missing files (index `.npy` splits, `requirements.txt`, folders) generated automatically.

**➡ In a hurry? Read [`QUICK_START.md`](QUICK_START.md) — the ordered 6-step
checklist: place the folder, drop in your `.mat` data, the before-run
adjustment table (`SMALL_DEBUG`, `user_rows`, batch size, …), run commands,
outputs, and a troubleshooting table.**

---

## 1. What was broken in the original repo (and is fixed here)

| # | Original problem | Fix here |
|---|------------------|----------|
| 1 | No notebook, 10 loose `.py` files | `lwlm/` package + `LWLM_pipeline.ipynb` + `scripts/` |
| 2 | `data_gen_test.py` imports a non-existent `music` module → crash | dead import removed (unused) |
| 3 | `train.npy` / `val_1000.npy` / `pretrain_data.npy` **never generated** by any script → `generate_Dataset` crashes | created automatically by `generate_data.py` / notebook Step 1 |
| 4 | `pretrain/` folder referenced by the pretrain loader but never created | loader falls back to the `train/` pool (same data, no 2× disk) |
| 5 | Output paths like `../../LWLM/dataset/...` and `../scenarios` (CWD-dependent) | all paths anchored to the project root, works from any CWD |
| 6 | `data_gen_test.py` never creates its output folder → `FileNotFoundError` | folders auto-created |
| 7 | `accelerator='gpu'` hardcoded → crash without GPU; `log_every_n_steps=0` invalid | `accelerator="auto"`, valid logging defaults |
| 8 | `sdp_kernel(...)` deprecated/removed in newer PyTorch → crash | guarded compatibility shim |
| 9 | No `requirements.txt` | `requirements.txt` provided |
| 10 | `results/` dir missing when saving test outputs | auto-created |

The **h5py file format, folder naming (`BS{bs}_128sc_32at_{bw}M/{bs}_{idx}.h5py`),
channel normalization, delay–angle transform, all label scalings (`/10`, `/100`) and the
model architecture are kept byte-for-byte compatible** with the original code.

---

## 2. Project structure

```
LWLM/
├── LWLM_pipeline.ipynb        ← ONE end-to-end notebook: data prep → pretrain →
│                                 fine-tune (5 tasks) → test → CDF plots
├── README.md
├── requirements.txt
│
├── Dataset/                   ← ★ PUT YOUR DeepMIMO O1_3p5 .mat FILES HERE ★
│   └── (O1_3p5/ or loose .mat files — both are auto-detected)
│
├── lwlm/                      ← importable Python package
│   ├── paths.py               ← project-root-anchored paths (fixes all relative-path bugs)
│   ├── config.py              ← EnvPara builder (paper defaults)
│   ├── transforms.py          ← channel_normalization / freq_to_delay_angle_domain (verbatim)
│   ├── datasets.py            ← generate_Dataset, generate_Dataset_test (verbatim logic)
│   ├── data_generation.py     ← DeepMIMO .mat → per-sample .h5py + index .npy files
│   ├── model.py               ← wireless_loc_fm (wlfm.py, verbatim)
│   ├── wrapper.py             ← Lightning Wrapper (train_model.py, verbatim)
│   ├── training.py            ← shared train() routine (original train.py __main__, fixed)
│   └── evaluation.py          ← shared inference / metrics / CDF helpers
│
├── scripts/                   ← command-line entry points (same CLI args as original)
│   ├── generate_data.py       ← python scripts/generate_data.py [--user_rows 100 ...]
│   ├── train.py               ← python scripts/train.py --task pretrain_mix ...
│   ├── test_singleBS.py
│   ├── test_multiBS.py
│   ├── test_toa.py
│   └── test_aoa.py
│
├── checkpoints/               ← trained .ckpt files land here
├── processed/                 ← (created at runtime) O1_3p5 h5py pool + index .npy files
├── results/                   ← test outputs (.txt, plots)
└── logs/                      ← TensorBoard logs
```

---

## 3. Setup

```bash
pip install -r requirements.txt
```

> DeepMIMOv3 is the official channel generator of the DeepMIMO project
> (https://deepmimo.net). If the pip package name changes, install per the
> DeepMIMOv3 documentation.

**Dataset — one-time download (your side):**
1. Download the **O1_3p5** scenario from https://deepmimo.net/scenarios/o1-scenario/
2. Extract it so that `.mat` files end up in the `Dataset/` folder of this project, e.g. either layout works:
   - `Dataset/O1_3p5/*.mat`  ← extracted scenario folder inside `Dataset/`
   - `Dataset/*.mat`         ← loose `.mat` files directly inside `Dataset/`

The converter auto-detects both layouts. **No dataset is generated or shipped by this
repo — it only reads yours.**

---

## 4. How to run

### Option A — Notebook (recommended)

Open `LWLM_pipeline.ipynb` and run top-to-bottom:

| Step | What it does |
|------|--------------|
| 1 | Convert `Dataset/*.mat` → `processed/O1_3p5/{train,test}/...` + index splits (skips automatically if already done) |
| 2 | Inspect one CSI sample (amplitude map + delay–angle domain) |
| 3 | **Pretraining** — `pretrain_mix` (SF-MCM + DTI + PICL) |
| 4 | **Fine-tune SingleBSLoc** (loads pretrain checkpoint) |
| 5 | **Fine-tune MultiBSLoc** (4-BS attention fusion) |
| 6 | **Fine-tune ToA** and **AoA** |
| 7 | **Test + evaluation** — mean / median / P90 error + paper-style CDF plot into `results/` |

Defaults = **paper scale** (`FT_dataset=10000`, `BS_Num=4`, `epochs=1000`, …).
Flip `SMALL_DEBUG = True` in the config cell for a quick smoke run.

### Option B — Command line (same arguments as the original README)

```bash
# 1) dataset conversion (reads Dataset/*.mat, writes processed/ + index files)
python scripts/generate_data.py --user_rows 100        # 100 rows ≈ 20k UE positions

# 2) pretraining
python scripts/train.py --task pretrain_mix \
    --model_path checkpoints/pretrain_mix --is_frozen 0 --is_load 0 --depth 4

# 3) fine-tuning (example: single-BS localization)
python scripts/train.py --task SingleBSLoc \
    --model_path checkpoints/SingleBSLoc \
    --is_frozen 0 --is_load 1 --depth 4 \
    --load_pretrained_mdl_path checkpoints/pretrain_mix/testbest.ckpt \
    --FT_dataset 10000 --BW 10 \
    --pilot_subcarrier_interval 1 --pilot_antenna_interval 1 --BS_Num 4

# 4) testing
python scripts/test_singleBS.py --ckpt checkpoints/SingleBSLoc/testbest.ckpt
python scripts/test_multiBS.py  --ckpt checkpoints/MultiBSLoc/testbest.ckpt
python scripts/test_toa.py      --ckpt checkpoints/toa/testbest.ckpt
python scripts/test_aoa.py      --ckpt checkpoints/aoa/testbest.ckpt
```

Checkpoint filenames: `ModelCheckpoint` saves the best epoch as
`testbest-epoch=XXX.ckpt` (monitors `val/ave_loss`) and the latest as
`test_latest.ckpt`.

---

## 5. Data layout produced by Step 1 (identical to the original repo)

```
processed/
├── train.npy              ← UE indices used for fine-tuning   (first FT_dataset are taken)
├── val_1000.npy           ← 1000 validation UE indices
├── pretrain_data.npy      ← UE indices available for SSL pretraining
└── O1_3p5/
    ├── train/BS3_128sc_32at_10M/3_0.h5py ...   (BS 3,4,9,10 × BW 10/20/50 MHz)
    └── test/BS3_128sc_32at_10M/3_0.h5py ...    (5400 UE positions, BW 10 MHz, BS 3,4,9,10)
```
*(the original test generator produced only BS 3, which made `test_multiBS.py`
crash on the missing BS 4/9/10 files — here all four BSs are generated)*

Each `.h5py` file contains exactly the original keys:
`channel_real, channel_imag, UElocation, LoS, distance, BSlocation, DoD_phi, DoD_theta, bandwidth`.

**Disk / time estimate (paper scale):** ~20,000 UE positions × 12 (BS, BW) combos ≈
240k small files ≈ **4–6 GB** and a few hours of DeepMIMO ray-tracing. Scale
`--user_rows` up (e.g. 2700 ≈ full grid) or down to fit your machine.

---

## 6. Citation

If you use this code, please cite the original paper (and star the original repo):

```bibtex
@article{pan2025large,
  title={Large Wireless Localization Model (LWLM): A Foundation Model for Positioning in 6G Networks},
  author={Pan, Guangjin and Huang, Kaixuan and Chen, Hui and Zhang, Shunqing and H{\"a}ger, Christian and Wymeersch, Henk},
  journal={arXiv preprint arXiv:2505.10134},
  year={2025}
}
```

Original author contact: guangjin.pan@chalmers.se
