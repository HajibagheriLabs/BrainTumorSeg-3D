"""3D segmentation network built from the model config."""

from monai.networks.nets import AttentionUnet, UNet
from torch import nn

from braintumorseg.config import ModelConfig


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.architecture == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channels=cfg.channels,
            strides=cfg.strides,
            num_res_units=cfg.num_res_units,
            dropout=cfg.dropout,
        )
    if cfg.architecture == "attention_unet":
        return AttentionUnet(
            spatial_dims=3,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channels=cfg.channels,
            strides=cfg.strides,
            dropout=cfg.dropout,
        )
    raise ValueError(f"unknown architecture {cfg.architecture!r}")
