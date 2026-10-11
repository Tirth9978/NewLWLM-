#!/usr/bin/env python3
"""DeepMIMO .mat -> h5py conversion (fixed version of data_gen_train.py + data_gen_test.py).

Reads the O1_3p5 scenario from ``Dataset/`` and writes the per-sample h5py pool
plus the index files (train.npy / val_1000.npy / pretrain_data.npy) that the
original repository was missing.

Examples:
    python scripts/generate_data.py                     # paper-ish scale (100 rows ~ 20k UE)
    python scripts/generate_data.py --user_rows 3       # tiny (original repo default, 600 UE)
    python scripts/generate_data.py --user_rows 2700    # full grid (slow, huge)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from lwlm.data_generation import generate_all  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="DeepMIMO O1_3p5 .mat -> per-sample h5py conversion")
    p.add_argument("--user_rows", type=int, default=100,
                   help="number of O1_3p5 grid rows for the train pool "
                        "(200 UE positions per row; original repo default: 3, "
                        "paper pretraining uses the full grid)")
    p.add_argument("--n_finetune", type=int, default=10000,
                   help="number of UE indices stored in train.npy (fine-tuning pool)")
    p.add_argument("--n_val", type=int, default=1000,
                   help="number of UE indices stored in val_1000.npy (validation pool)")
    p.add_argument("--test_rows", type=int, default=2700,
                   help="number of grid rows for the test pool (original: 2700 -> 5400 UE)")
    p.add_argument("--test_subsampling", type=float, default=0.01,
                   help="user subsampling factor for the test pool (original: 0.01)")
    p.add_argument("--force", action="store_true",
                   help="regenerate even if index files already exist")
    args = p.parse_args()

    from lwlm.data_generation import generate_test_pool, generate_train_pool

    meta = generate_train_pool(user_rows=args.user_rows, n_finetune=args.n_finetune,
                               n_val=args.n_val, skip_existing=not args.force)
    meta_test = generate_test_pool(user_rows=args.test_rows,
                                   user_subsampling=args.test_subsampling,
                                   skip_existing=not args.force)
    print("\n=== conversion summary ===")
    print("train pool:", meta)
    print("test pool :", meta_test)


if __name__ == "__main__":
    main()
