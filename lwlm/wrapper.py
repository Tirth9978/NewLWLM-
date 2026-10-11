"""Lightning ``Wrapper`` - logic identical to the original ``train_model.py``.

Two environment-compatibility fixes only (no training-logic change):
  * ``torch.backends.cuda.sdp_kernel`` is deprecated and removed in recent
    PyTorch versions -> wrapped in a guarded shim (it only selected the math
    attention backend; the default backend works identically for this model);
  * ``torch.autograd.set_detect_anomaly(True)`` is a *debug* tool that slows
    GPU training by an order of magnitude -> now opt-in via
    ``EnvPara["detect_anomaly"]`` (default False, as in normal training runs).
"""

from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .model import *  # noqa: F401,F403  (wireless_loc_fm and helpers, verbatim)
from .model import wireless_loc_fm


def _enable_math_sdp():
    """Original code: ``sdp_kernel(enable_math=True, enable_flash=False,
    enable_mem_efficient=False)``.  Deprecated in torch>=2.4 - try, and
    silently continue with the default SDPA backend if unavailable."""
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    except Exception:
        pass


class Wrapper(pl.LightningModule):
    def __init__(self, EnvPara):
        super().__init__()

        self.channel_fdmdl = wireless_loc_fm(
            fshape=EnvPara["fshape"], tshape=EnvPara["tshape"], fstride=EnvPara["fstride"],
            tstride=EnvPara["tstride"],
            input_fdim=EnvPara["input_fdim"], input_tdim=EnvPara["input_tdim"],
            input_fmap=EnvPara["input_fmap"], embed_dim=EnvPara["embed_dim"], depth=EnvPara["depth"],
            num_heads=EnvPara["num_heads"], device=EnvPara["device"], EnvPara=EnvPara)
        self.task = EnvPara["task"]
        self.train_epoch_loss = []
        self.valepoch_loss = []
        self.EnvPara = EnvPara
        torch.autograd.set_detect_anomaly(bool(EnvPara.get("detect_anomaly", False)))
        _enable_math_sdp()

    def forward(self, channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri,
                channel_delay_angle_ri, UElocation_all, BSconf_all):

        if (self.task == "pretrain_mix"):
            loss = self.channel_fdmdl(
                channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
                channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
                channel_delay_angle_ri=channel_delay_angle_ri, UElocation_all=UElocation_all,
                BSconf_all=BSconf_all, task=self.task,
                mask_antenna_number=self.EnvPara["mask_antenna_number"],
                mask_subcarrier_number=self.EnvPara["mask_subcarrier_number"])
        elif self.task == "SingleBSLoc":
            loss = self.channel_fdmdl(
                channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
                channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
                channel_delay_angle_ri=channel_delay_angle_ri, UElocation_all=UElocation_all,
                BSconf_all=BSconf_all, task=self.task)
        elif self.task == "MultiBSLoc":
            loss = self.channel_fdmdl(
                channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
                channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
                channel_delay_angle_ri=channel_delay_angle_ri, UElocation_all=UElocation_all,
                BSconf_all=BSconf_all, task=self.task)
        elif self.task == "toa":
            loss = self.channel_fdmdl(
                channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
                channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
                channel_delay_angle_ri=channel_delay_angle_ri, UElocation_all=UElocation_all,
                BSconf_all=BSconf_all, task=self.task)
        elif self.task == "aoa":
            loss = self.channel_fdmdl(
                channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
                channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
                channel_delay_angle_ri=channel_delay_angle_ri, UElocation_all=UElocation_all,
                BSconf_all=BSconf_all, task=self.task)

        return loss

    def training_step(self, batch, batch_idx):
        channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, \
            UElocation_all, BSconf_all = batch
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.float()
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.float()
        channel_delay_angle_ri = channel_delay_angle_ri.float()
        UElocation_all = UElocation_all.float()
        BSconf_all = BSconf_all.float()

        loss = self.forward(
            channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
            channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
            channel_delay_angle_ri=channel_delay_angle_ri, UElocation_all=UElocation_all,
            BSconf_all=BSconf_all)

        self.train_epoch_loss.append(loss.detach())
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # calculate average loss
        avg_loss = torch.stack(self.train_epoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Training Loss: {avg_loss.item()}",
              len(self.train_epoch_loss))
        self.log('train/ave_loss', avg_loss.item(), prog_bar=True, sync_dist=True,
                 on_step=False, on_epoch=True)
        # Clear the loss list for the next epoch of training
        self.train_epoch_loss.clear()

    def validation_step(self, batch, batch_idx):
        channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, \
            UElocation_all, BSconf_all = batch
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.float()
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.float()
        channel_delay_angle_ri = channel_delay_angle_ri.float()
        UElocation_all = UElocation_all.float()
        BSconf_all = BSconf_all.float()

        loss = self.forward(
            channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa,
            channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri,
            channel_delay_angle_ri=channel_delay_angle_ri, UElocation_all=UElocation_all,
            BSconf_all=BSconf_all)

        self.valepoch_loss.append(loss.detach())
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # calculate average loss
        if len(self.valepoch_loss) == 0:
            return
        avg_loss = torch.stack(self.valepoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Validation Loss: {avg_loss.item()}",
              len(self.valepoch_loss))
        self.log('val/ave_loss', avg_loss.item(), prog_bar=True, sync_dist=True,
                 on_step=False, on_epoch=True)

        # Clear the loss list for the next epoch of training
        self.valepoch_loss.clear()

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=self.EnvPara["lr"], weight_decay=1e-4)
        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, max_lr=1e-3, total_steps=15625000, pct_start=0.1, final_div_factor=1e2)

        return {
            'optimizer': self.optim,
            # 'lr_scheduler': {'scheduler': self.schedule, 'interval': 'step'}
        }
