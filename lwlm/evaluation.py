"""Shared inference / evaluation helpers for the four test scripts and the notebook.

The per-task inference logic is identical to the original ``test_*.py`` files
(same task strings, same ``x10`` / ``x100`` de-normalisation scalings, same
valid-sample filtering, same mean-error metric); CDF plotting was added since
the paper reports error CDFs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import generate_Dataset_test
from .paths import results_dir


def task_to_inference(task: str) -> str:
    """Map a training task name to its inference counterpart (original scheme)."""
    m = {
        "SingleBSLoc": "inference_SingleBSLoc",
        "MultiBSLoc": "inference_multiBSLoc",
        "toa": "inference_toa",
        "aoa": "inference_aoa",
    }
    return m.get(task, task)


def run_inference(Wrapper, EnvPara: dict, ckpt_path: str, log=print):
    """Run the original test loop and return (pred, label, loss) arrays in metres/deg.

    * localisation tasks: pred/label are (N, 2) positions in metres
    * toa / aoa tasks:    pred/label are (N, 1) distances (m) / angles (deg)
    """
    device = EnvPara["device"]
    task = task_to_inference(EnvPara["task"])

    FMmodel = Wrapper.load_from_checkpoint(ckpt_path, EnvPara=EnvPara, strict=False)
    FMmodel.to(device)
    FMmodel.eval()

    test_dataset = generate_Dataset_test(EnvPara)
    test_dataloader = DataLoader(test_dataset, batch_size=EnvPara.get("batch_size", 32),
                                 num_workers=EnvPara.get("num_workers", 4),
                                 shuffle=False, drop_last=False, pin_memory=True)

    n = len(test_dataset)
    if EnvPara["task"] in ("SingleBSLoc", "MultiBSLoc"):
        result = np.zeros((n, 2))
        label = np.zeros((n, 2))
    else:  # toa / aoa
        result = np.zeros((n, 1))
        label = np.zeros((n, 1))
    losses = np.zeros((n, 1))
    cnt = 0

    with torch.no_grad():
        for channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, \
                channel_delay_angle_ri, UElocation_all, BSconf_all in test_dataloader:
            FMmodel.eval()
            channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.to(device).float()
            channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.to(device).float()
            channel_delay_angle_ri = channel_delay_angle_ri.to(device).float()
            UElocation_all = UElocation_all.to(device).float()
            BSconf_all = BSconf_all.to(device).float()

            pred, loss = FMmodel.channel_fdmdl(
                channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
                channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
                channel_delay_angle_ri=channel_delay_angle_ri,
                UElocation_all=UElocation_all, BSconf_all=BSconf_all, task=task)

            b = len(channel_Antenna_subcarrier_aa)
            if EnvPara["task"] in ("SingleBSLoc", "MultiBSLoc"):
                # original de-normalisation: positions were divided by 10
                result[cnt:cnt + b] = pred.cpu().detach().numpy() * 10
                label[cnt:cnt + b] = UElocation_all[:, 0, :2].cpu().detach().numpy() * 10
            elif EnvPara["task"] == "toa":
                # original de-normalisation: distance was divided by 100
                result[cnt:cnt + b] = pred.cpu().detach().numpy() * 100
                label[cnt:cnt + b, 0] = UElocation_all[:, 0, 2].cpu().detach().numpy() * 100
            elif EnvPara["task"] == "aoa":
                # original de-normalisation: angle was divided by 100
                result[cnt:cnt + b] = pred.cpu().detach().numpy() * 100
                label[cnt:cnt + b, 0] = UElocation_all[:, 0, 3].cpu().detach().numpy() * 100
            losses[cnt:cnt + b, 0] = loss.cpu().detach().numpy()
            cnt += b

    return result, label, losses


def summarize_errors(result, label, task: str, log=print) -> np.ndarray:
    """Original metric: filter non-covered UEs, compute the error per sample."""
    valid_indices = np.where(np.sum(np.abs(label), 1) != 0)[0]
    valid_result = result[valid_indices, :]
    valid_label = label[valid_indices, :]

    if task in ("SingleBSLoc", "MultiBSLoc"):
        errors = np.linalg.norm(valid_result - valid_label, axis=1)  # Euclidean (m)
        log(f"mean localization error : {np.mean(errors):.4f} m")
        log(f"median localization err : {np.median(errors):.4f} m")
        log(f"P90 localization error  : {np.percentile(errors, 90):.4f} m")
    else:
        errors = np.abs(valid_result[:, 0] - valid_label[:, 0])  # MAE (m / deg)
        unit = "m" if task == "toa" else "deg"
        log(f"mean {task} error : {np.mean(errors):.4f} {unit}")

    return errors


def save_results_txt(result, label, save_result_file: str, log=print) -> Path:
    """Original output: `label_x label_y pred_x pred_y` txt file in results/."""
    out_dir = results_dir()
    valid_indices = np.where(np.sum(np.abs(label), 1) != 0)[0]
    merged = np.hstack((label[valid_indices, :], result[valid_indices, :]))
    out = out_dir / f"{save_result_file}.txt"
    np.savetxt(out, merged, fmt="%.6f", delimiter=' ')
    log(f"[eval] results saved -> {out}")
    return out


def plot_error_cdf(errors: np.ndarray, title: str, save_result_file: str,
                   xlabel: str = "Error", log=print) -> Path:
    """Paper-style CDF of the (localization / toa / aoa) error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errs = np.sort(np.asarray(errors).ravel())
    cdf = np.arange(1, len(errs) + 1) / len(errs)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    ax.plot(errs, cdf * 100, lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF [%]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = results_dir() / f"{save_result_file}_cdf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"[eval] CDF plot saved -> {out}")
    return out
