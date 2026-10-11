#!/usr/bin/env python3
"""Training CLI - same arguments as the original repository's ``train.py``,
with the environment bugs fixed (CPU fallback, valid logging, path anchoring).
The training routine lives in ``lwlm.training`` so the notebook can reuse it.

Examples (mirroring the original README):
    python scripts/train.py --task pretrain_mix \
        --model_path checkpoints/pretrain_mix --is_frozen 0 --is_load 0 --depth 4

    python scripts/train.py --task SingleBSLoc \
        --model_path checkpoints/SingleBSLoc \
        --is_frozen 0 --is_load 1 --depth 4 \
        --load_pretrained_mdl_path checkpoints/pretrain_mix/testepoch=139.ckpt \
        --FT_dataset 10000 --BW 10 \
        --pilot_subcarrier_interval 1 --pilot_antenna_interval 1 --BS_Num 4
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from lwlm.training import set_seed, train  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="LWLM environment parameters")
    # Training Parameters (identical to the original train.py)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--input_tdim", type=int, default=32)    # antenna number
    parser.add_argument("--input_fdim", type=int, default=128)   # CSI feature map
    parser.add_argument("--input_fmap", type=int, default=2)     # feature maps
    parser.add_argument("--fshape", type=int, default=4)         # patch parameters
    parser.add_argument("--tshape", type=int, default=4)
    parser.add_argument("--fstride", type=int, default=4)
    parser.add_argument("--tstride", type=int, default=4)
    parser.add_argument("--task", type=str, default='SingleBSLoc')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--load_pretrained_mdl_path", type=str, default='')
    parser.add_argument("--pretrain_stage", type=bool, default=True)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--DTI_embed_dim", type=int, default=96)
    parser.add_argument("--AFMCM_embed_dim", type=int, default=96)
    parser.add_argument("--PICL_embed_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)          # transformer parameters
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_antenna_number", type=int, default=16)
    parser.add_argument("--mask_subcarrier_number", type=int, default=64)
    parser.add_argument("--is_frozen", type=int, default=1)
    parser.add_argument("--BW", type=int, default=10)
    parser.add_argument("--DTI_flag", type=int, default=1)
    parser.add_argument("--AFMCM_flag", type=int, default=1)
    parser.add_argument("--PICL_flag", type=int, default=1)
    parser.add_argument("--FT_dataset", type=int, default=10000)
    parser.add_argument("--pilot_subcarrier_interval", type=int, default=1)
    parser.add_argument("--pilot_antenna_interval", type=int, default=1)
    parser.add_argument("--BS_Num", type=int, default=4)
    parser.add_argument("--is_load", type=int, default=0)
    # --- additional (new, safe defaults) ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_samples", type=int, default=5400)
    parser.add_argument("--detect_anomaly", type=int, default=0,
                        help="1 = enable torch autograd anomaly detection (debug, slow)")
    return parser


def make_envpara(args) -> dict:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EnvPara = vars(args).copy()
    EnvPara["pretrain_stage"] = True
    EnvPara["device"] = device
    EnvPara["input_feature_dim"] = 2 * EnvPara["BS_Num"]
    EnvPara["detect_anomaly"] = bool(EnvPara["detect_anomaly"])
    return EnvPara


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    EnvPara = make_envpara(args)
    print(EnvPara)
    train(EnvPara)


if __name__ == "__main__":
    main()
