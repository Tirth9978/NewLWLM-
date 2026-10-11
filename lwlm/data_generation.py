"""DeepMIMO ``.mat`` -> per-sample ``.h5py`` converter.

This merges the original ``data_gen_train.py`` + ``data_gen_test.py`` into one
configurable, restart-safe module and additionally generates the **index
files** (``train.npy``, ``val_1000.npy``, ``pretrain_data.npy``) that the
original repository referenced but never produced.

Faithful to the original generation parameters:

* pretrain + fine-tune pool: BS 3/4/9/10, BW 10/20/50 MHz, 32-ant ULA,
  128 subcarriers, num_paths = 5, single-antenna UE, ``enable_BS2B = 0``;
* test pool: BS 3, BW 10 MHz, ``user_rows = arange(2700)`` with
  ``user_subsampling = 0.01`` (-> 5400 UE positions, as in the original).

Differences (bug fixes only):
  * scenario folder auto-located inside ``Dataset/`` (both common layouts);
  * output written to ``<project>/processed/O1_3p5/{train,test}/`` (no more
    ``../../LWLM/dataset/...``);
  * output folders created before writing (original test generator crashed);
  * dead ``import music`` removed;
  * generation is skipped when the index files already exist (restart-safe);
  * ``del`` + ``gc`` between bandwidth passes to bound peak RAM.

The h5py file layout and dataset keys are byte-compatible with the original:
``BS{bs}_128sc_32at_{bw}M/{bs}_{ue}.h5py`` with keys ``channel_real,
channel_imag, UElocation, LoS, distance, BSlocation, DoD_phi, DoD_theta,
bandwidth``.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np

from .paths import bs_folder, index_file, locate_scenario, processed_dir

SCENARIO = "O1_3p5"
BS_LIST = [3, 4, 9, 10]
BW_LIST = [10, 20, 50]
N_SC = 128
N_ANT = 32
NUM_PATHS = 5


# ---------------------------------------------------------------------------
# low-level: save one generated (BS, BW) block  [verbatim h5py writer]
# ---------------------------------------------------------------------------
def _save_block(dataset, bs_i: int, bs_id: int, bw: int, split: str, log) -> int:
    """Save ``dataset[bs_i]`` (one BS) as per-UE h5py files. Returns #users."""
    import h5py

    folder = bs_folder(bs_id, bw, split=split, n_sc=N_SC, n_ant=N_ANT, scenario=SCENARIO)
    n_users = len(dataset[bs_i]['user']['channel'])
    for ue_i in range(n_users):
        with h5py.File(folder / f"{bs_id}_{ue_i}.h5py", 'w') as f:
            f.create_dataset('channel_real', data=np.real(dataset[bs_i]['user']['channel'][ue_i]))
            f.create_dataset('channel_imag', data=np.imag(dataset[bs_i]['user']['channel'][ue_i]))
            f.create_dataset('UElocation', data=dataset[bs_i]['user']['location'][ue_i])
            f.create_dataset('LoS', data=dataset[bs_i]['user']['LoS'][ue_i])
            f.create_dataset('distance', data=dataset[bs_i]['user']['distance'][ue_i])
            f.create_dataset('BSlocation', data=dataset[bs_i]['location'])
            f.create_dataset('DoD_phi', data=dataset[bs_i]['user']["paths"][ue_i]["DoD_phi"])
            f.create_dataset('DoD_theta', data=dataset[bs_i]['user']["paths"][ue_i]["DoD_theta"])
            f.create_dataset('bandwidth', data=bw)
    log(f"    saved {n_users} UE files -> {folder.name}")
    return n_users


def _deepmimo_params(dataset_folder: Path, user_rows, user_subsampling, active_bs, bw):
    """DeepMIMOv3 parameter dict - identical values to the original scripts."""
    import DeepMIMOv3 as DeepMIMO

    parameters = DeepMIMO.default_params()
    parameters['scenario'] = SCENARIO
    parameters['dataset_folder'] = str(dataset_folder)
    parameters['num_paths'] = NUM_PATHS
    parameters['user_rows'] = user_rows
    parameters['user_subsampling'] = user_subsampling
    parameters['active_BS'] = np.array(active_bs)
    parameters['OFDM']['bandwidth'] = bw / 1000            # GHz (BW in MHz)
    parameters['OFDM']['subcarriers'] = N_SC
    parameters['OFDM']['selected_subcarriers'] = np.arange(N_SC)
    parameters['ue_antenna']['shape'] = np.array([1, 1])   # single antenna
    parameters['bs_antenna']['shape'] = np.array([N_ANT, 1])  # ULA of 32 elements
    # NOTE: the original repo wrote  parameters['enable_BS2B'] = 0  which is a
    # no-op typo (unknown key).  The real DeepMIMOv3 key is 'enable_BS2BS' and
    # it defaults to 1; setting it to 0 matches the author's intent and skips
    # the BS-to-BS channel computation (user channels are unaffected).
    parameters['enable_BS2BS'] = 0
    return parameters


# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------
def generate_train_pool(user_rows: int = 100,
                        n_finetune: int = 10000,
                        n_val: int = 1000,
                        log=print,
                        skip_existing: bool = True) -> dict:
    """Generate the pretraining + fine-tuning pool.

    Generates ``user_rows`` grid rows (200 UE positions per row in O1_3p5) for
    every (BS, BW) combination in ``BS_LIST x BW_LIST`` and writes the index
    files used by ``generate_Dataset``.

    Returns a metadata dict (also stored in ``processed/meta.json``).
    """
    import DeepMIMOv3 as DeepMIMO  # noqa: F401  (fail early if not installed)

    meta_file = processed_dir() / "meta.json"
    train_idx_file = index_file("train.npy")
    val_idx_file = index_file("val_1000.npy")
    pre_idx_file = index_file("pretrain_data.npy")

    if skip_existing and train_idx_file.exists() and val_idx_file.exists() and pre_idx_file.exists():
        log(f"[skip] index files already exist in {processed_dir()} "
            f"(delete 'processed/' to regenerate)")
        if meta_file.exists():
            return json.loads(meta_file.read_text())
        return {"skipped": True}

    dataset_folder = locate_scenario(SCENARIO)
    log(f"[data] scenario folder : {dataset_folder}")
    log(f"[data] output folder   : {processed_dir() / SCENARIO}")
    n_users_expected = user_rows * 200  # O1_3p5 grid has 200 UE positions per row
    est_gb = n_users_expected * len(BS_LIST) * len(BW_LIST) * 17e-3 / 1024
    log(f"[data] generating {user_rows} rows x 200 UE = {n_users_expected} positions "
        f"per (BS, BW)  ->  ~{est_gb:.1f} GB of h5py files")
    t0 = time.time()

    user_rows_arr = np.arange(user_rows)
    n_users = None
    for bw in BW_LIST:
        log(f"[data] BW = {bw} MHz - generating channels for BS {BS_LIST} "
            f"(this is the slow DeepMIMO ray-tracing step)")
        parameters = _deepmimo_params(dataset_folder, user_rows_arr, 1, BS_LIST, bw)
        dataset = DeepMIMO.generate_data(parameters)
        log(f"    generated coordinates: {len(dataset[0]['user']['channel'])}")
        for bs_i, bs_id in enumerate(BS_LIST):
            n = _save_block(dataset, bs_i, bs_id, bw, split="train", log=log)
            n_users = n if n_users is None else min(n_users, n)
        del dataset
        gc.collect()

    # ------------------------------------------------------------------
    # index files (the original repo never generated these!)
    # ------------------------------------------------------------------
    n_users = int(n_users or 0)
    if n_users == 0:
        raise RuntimeError("DeepMIMO generation produced 0 users - check the scenario files.")

    all_idx = np.arange(n_users)
    if n_users > n_val + 1:
        # normal case: paper-scale pool -> FT first, val takes the next n_val (disjoint)
        n_ft = min(n_finetune, n_users - n_val)
        n_va = n_val
    else:
        # small pool: keep 90% for fine-tuning, 10% (>=1) for validation
        n_ft = max(int(n_users * 0.9), 0)
        n_va = n_users - n_ft
    np.save(train_idx_file, all_idx[:n_ft])           # fine-tuning: first n_ft
    np.save(val_idx_file, all_idx[n_ft:n_ft + n_va])  # validation: next n_va (disjoint)
    np.save(pre_idx_file, all_idx)                    # SSL pretraining pool: all

    meta = {
        "scenario": SCENARIO,
        "user_rows": user_rows,
        "user_subsampling": 1,
        "n_users": n_users,
        "n_finetune": int(n_ft),
        "n_val": int(n_va),
        "BS_list": BS_LIST,
        "BW_list": BW_LIST,
        "num_paths": NUM_PATHS,
        "n_sc": N_SC,
        "n_ant": N_ANT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    log(f"[data] DONE in {meta['elapsed_min']} min - {n_users} UE positions x "
        f"{len(BS_LIST)} BS x {len(BW_LIST)} BW")
    log(f"[data] index files: train.npy ({n_ft}), val_1000.npy ({n_va}), "
        f"pretrain_data.npy ({n_users})")
    return meta


def generate_test_pool(user_rows: int = 2700,
                       user_subsampling: float = 0.01,
                       bs_list=None,
                       log=print,
                       skip_existing: bool = True) -> dict:
    """Generate the test pool (original ``data_gen_test.py``).

    Default parameters reproduce the original 5400 test UE positions.
    FIX: the original generator wrote only BS 3, which made the multi-BS test
    (``test_multiBS.py`` needs test files for BS 3/4/9/10) crash; here all
    ``bs_list`` base stations are generated (default BS 3,4,9,10), which keeps
    the single-BS tests unchanged and makes the multi-BS test workable.
    """
    import h5py  # noqa: F401
    import DeepMIMOv3 as DeepMIMO  # noqa: F401

    if bs_list is None:
        bs_list = BS_LIST

    meta_file = processed_dir() / "meta_test.json"
    if skip_existing and meta_file.exists():
        log(f"[skip] test pool already generated (delete 'processed/' to regenerate)")
        return json.loads(meta_file.read_text())

    dataset_folder = locate_scenario(SCENARIO)
    bw = 10
    log(f"[data] TEST pool: BS {bs_list}, BW {bw} MHz, rows arange({user_rows}) "
        f"x subsampling {user_subsampling}")
    t0 = time.time()

    parameters = _deepmimo_params(dataset_folder, np.arange(user_rows), user_subsampling, bs_list, bw)
    dataset = DeepMIMO.generate_data(parameters)
    n_test = None
    for bs_i, bs_id in enumerate(bs_list):
        n = _save_block(dataset, bs_i, bs_id, bw, split="test", log=log)
        n_test = n if n_test is None else min(n_test, n)
    del dataset
    gc.collect()

    meta = {
        "scenario": SCENARIO,
        "split": "test",
        "BS_list": list(bs_list),
        "BW": bw,
        "user_rows": user_rows,
        "user_subsampling": user_subsampling,
        "n_test": int(n_test or 0),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    log(f"[data] test pool DONE: {n_test} UE positions x {len(bs_list)} BS")
    return meta


def generate_all(user_rows: int = 100, n_finetune: int = 10000, n_val: int = 1000,
                 log=print, skip_existing: bool = True) -> dict:
    """Full conversion: train pool + test pool (used by notebook Step 1)."""
    meta = generate_train_pool(user_rows=user_rows, n_finetune=n_finetune,
                               n_val=n_val, log=log, skip_existing=skip_existing)
    meta_test = generate_test_pool(log=log, skip_existing=skip_existing)
    meta["test"] = meta_test
    return meta
