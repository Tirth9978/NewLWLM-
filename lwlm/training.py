"""Shared training routine (extracted from the original ``train.py`` __main__).

The original ``train.py`` mixed CLI parsing with the training loop, which made
it impossible to reuse from a notebook.  The routine itself is unchanged apart
from the environment fixes (accelerator auto-detection, valid logging defaults,
path anchoring, optional seeding).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_model_path(task: str) -> Path:
    from .paths import checkpoints_dir
    return checkpoints_dir() / task


def train(EnvPara: dict, log=print):
    """Train (pretrain or fine-tune) the LWLM Wrapper. Returns best ckpt path."""
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader

    from .datasets import generate_Dataset
    from .paths import logs_dir
    from .wrapper import Wrapper

    seed_everything(EnvPara.get("seed", 42), workers=True)
    log(f"device: {EnvPara['device']}")

    if not EnvPara.get("model_path"):
        EnvPara["model_path"] = str(default_model_path(EnvPara["task"]))
    Path(EnvPara["model_path"]).mkdir(parents=True, exist_ok=True)

    FMmodel = Wrapper(EnvPara=EnvPara)

    if (EnvPara["task"] in ("SingleBSLoc", "MultiBSLoc", "aoa", "toa")) or (EnvPara["is_load"] == 1):
        assert EnvPara.get("load_pretrained_mdl_path"), \
            f"task '{EnvPara['task']}' requires load_pretrained_mdl_path (pretrain checkpoint)"
        FMmodel = Wrapper.load_from_checkpoint(EnvPara["load_pretrained_mdl_path"],
                                               EnvPara=EnvPara, strict=False)
        log(f"loaded pretrained weights: {EnvPara['load_pretrained_mdl_path']}")

    FMmodel.to(EnvPara["device"])

    train_dataset = generate_Dataset(EnvPara)
    train_dataloader = DataLoader(train_dataset, batch_size=EnvPara["batch_size"],
                                  num_workers=EnvPara["num_workers"], shuffle=True,
                                  drop_last=False, pin_memory=True)
    val_dataset = generate_Dataset(EnvPara, is_val=1)
    val_dataloader = DataLoader(val_dataset, batch_size=EnvPara["batch_size"],
                                num_workers=EnvPara["num_workers"], shuffle=True,
                                drop_last=False, pin_memory=True)

    model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename='test{epoch:02d}',     # -> 'testepoch=XXX.ckpt' (same naming as original)
        save_top_k=1,                   # only save the best model
        monitor="val/ave_loss",         # monitor validation loss to determine the best model
        mode="min",
        save_weights_only=True,
    )
    latest_model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename='test_latest',
        save_top_k=1,
        save_last=True,
        every_n_epochs=5,
        save_weights_only=True,
    )

    logger = TensorBoardLogger(str(logs_dir()), name=EnvPara["task"])
    trainer = Trainer(
        enable_progress_bar=False,
        devices=1,
        accelerator="auto",             # FIX: was hard-coded 'gpu' (crashed on CPU)
        logger=logger,
        max_epochs=EnvPara["epochs"],
        callbacks=[model_checkpoint, latest_model_checkpoint],
    )
    trainer.fit(FMmodel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    log(f"best checkpoint: {model_checkpoint.best_model_path}")
    return model_checkpoint.best_model_path
