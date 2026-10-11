"""Central environment-parameters builder (``EnvPara``).

Collects every knob that was previously scattered across ``train.py`` and the
four ``test_*.py`` files into one place.  ``paper_envpara()`` reproduces the
paper-scale settings used by the original README commands.
"""

from __future__ import annotations

import torch


def paper_envpara(task: str = "SingleBSLoc",
                  epochs: int = 1000,
                  ft_dataset: int = 10000,
                  bs_num: int = 4,
                  bw: int = 10,
                  is_frozen: int = 0,
                  lr: float = 1e-4,
                  batch_size: int = 32,
                  num_workers: int = 4,
                  device: str | torch.device | None = None,
                  detect_anomaly: bool = False,
                  seed: int = 42) -> dict:
    """Paper-scale configuration (values follow the original repo's defaults).

    Parameters mirror the original CLI arguments one-to-one.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    EnvPara = {
        # ---- model input shape -------------------------------------------------
        "input_tdim": 32,        # antenna number
        "input_fdim": 128,       # subcarrier number
        "input_fmap": 2,         # (amplitude, phase) / (real, imag) maps
        "fshape": 4, "tshape": 4, "fstride": 4, "tstride": 4,   # patch parameters
        # ---- task --------------------------------------------------------------
        "task": task,
        "pretrain_stage": True,
        # ---- transformer / decoders -------------------------------------------
        "embed_dim": 256,
        "depth": 4,
        "latent_dim": 128,
        "num_heads": 4,
        "DTI_embed_dim": 96,
        "AFMCM_embed_dim": 96,
        "PICL_embed_dim": 64,
        "DTI_flag": 1,
        "AFMCM_flag": 1,
        "PICL_flag": 1,
        # ---- pretraining masks -------------------------------------------------
        "mask_antenna_number": 16,
        "mask_subcarrier_number": 64,
        # ---- optimisation ------------------------------------------------------
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "num_workers": num_workers,
        # ---- fine-tuning -------------------------------------------------------
        "is_frozen": is_frozen,
        "is_load": 0,
        "FT_dataset": ft_dataset,
        "BW": bw,
        "BS_Num": bs_num,
        "pilot_subcarrier_interval": 1,   # 1 = all channel entries used as pilots
        "pilot_antenna_interval": 1,
        # ---- misc --------------------------------------------------------------
        "model_path": "",
        "load_pretrained_mdl_path": "",
        "save_result_file": "result",
        "test_samples": 5400,
        "detect_anomaly": detect_anomaly,
        "seed": seed,
    }
    EnvPara["input_feature_dim"] = 2 * EnvPara["BS_Num"]
    EnvPara["device"] = device
    return EnvPara


def apply_overrides(EnvPara: dict, **overrides) -> dict:
    """Update ``EnvPara`` with non-None ``overrides`` and refresh derived keys."""
    for k, v in overrides.items():
        if v is not None:
            EnvPara[k] = v
    EnvPara["input_feature_dim"] = 2 * EnvPara.get("BS_Num", 1)
    return EnvPara
