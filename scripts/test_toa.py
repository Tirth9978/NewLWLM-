#!/usr/bin/env python3
"""ToA estimation test (fixed version of the original ``test_toa.py``).

Example:
    python scripts/test_toa.py --ckpt checkpoints/toa/testepoch=602.ckpt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from lwlm.config import paper_envpara, apply_overrides  # noqa: E402
from lwlm.evaluation import run_inference, save_results_txt, summarize_errors  # noqa: E402
from lwlm.wrapper import Wrapper  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="LWLM ToA estimation test")
    p.add_argument("--ckpt", type=str, required=True, help="fine-tuned .ckpt file")
    p.add_argument("--BW", type=int, default=10)
    p.add_argument("--test_samples", type=int, default=5400)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_result_file", type=str, default="toa_est")
    args = p.parse_args()

    EnvPara = paper_envpara(task="toa", bs_num=1)
    apply_overrides(EnvPara, BW=args.BW, test_samples=args.test_samples,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    save_result_file=args.save_result_file)

    result, label, losses = run_inference(Wrapper, EnvPara, args.ckpt)
    summarize_errors(result, label, "toa")
    save_results_txt(result, label, EnvPara["save_result_file"])


if __name__ == "__main__":
    main()
