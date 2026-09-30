"""Config-driven training loop with mixed precision and MLflow logging."""

import dataclasses
import math
import os
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, set_track_meta
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from torch import nn

from braintumorseg.config import Config, save_config
from braintumorseg.data import case_records, evaluation_cases, read_splits
from braintumorseg.inference import predict_labels
from braintumorseg.metrics import dice, region_mask
from braintumorseg.model import build_model
from braintumorseg.transforms import train_transforms, val_transforms


def run_dir(cfg: Config) -> Path:
    return cfg.runtime.output_dir / cfg.name


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loss(name: str) -> nn.Module:
    options = {
        "to_onehot_y": True,
        "softmax": True,
        # background fills most of every patch, so the dice term scores tumour classes only
        "include_background": False,
        # a class absent from one patch would give a 0/0 term, so dice spans the batch
        "batch": True,
    }
    losses = {"dice": DiceLoss, "dice_ce": DiceCELoss, "dice_focal": DiceFocalLoss}
    return losses[name](**options)


def flatten_config(tree: dict, prefix: str = "") -> dict[str, str]:
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= flatten_config(value, f"{name}.")
        else:
            flat[name] = str(value)
    return flat


def _git_state() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unavailable"}
    # the paths, not just a flag, show whether an uncommitted change could touch the result
    dirty = ", ".join(line[3:] for line in status if line.strip())
    return {
        "git_commit": commit,
        "git_dirty": str(bool(dirty)),
        "git_dirty_paths": dirty,
    }


@contextmanager
def tracked_run(cfg: Config) -> Iterator[mlflow.ActiveRun]:
    """The MLflow run tied to this config's run directory, created once and resumed after."""
    tracking = cfg.runtime.tracking_dir.resolve()
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name(cfg.runtime.experiment)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            cfg.runtime.experiment, artifact_location=(tracking / "artifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    id_file = run_dir(cfg) / "mlflow_run_id"
    run_id = id_file.read_text(encoding="utf-8").strip() if id_file.exists() else None
    with mlflow.start_run(
        run_id=run_id, experiment_id=experiment_id, run_name=cfg.name
    ) as run:
        id_file.write_text(run.info.run_id, encoding="utf-8")
        yield run


def _save_atomic(state: dict[str, Any], path: Path) -> None:
    # an interrupted write must never leave a truncated checkpoint that a resume would load
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    os.replace(partial, path)


def validation_dice(
    model: nn.Module, dataset: CacheDataset, cfg: Config
) -> dict[str, float]:
    """Mean per-case Dice per region; cases with an empty region are left out of its mean."""
    model.eval()
    scores: dict[str, list[float]] = {region: [] for region in cfg.data.regions}
    for sample in dataset:
        pred, truth = predict_labels(model, sample["image"], cfg), sample["label"][0]
        for region, labels in cfg.data.regions.items():
            scores[region].append(
                dice(region_mask(pred, labels), region_mask(truth, labels))
            )
    return {region: float(np.nanmean(values)) for region, values in scores.items()}


def record_identity(cfg: Config, device: torch.device) -> None:
    """Log what defines a run to the active MLflow run and its directory, once per run."""
    out = run_dir(cfg)
    save_config(cfg, out / "config.yaml")
    mlflow.log_params(flatten_config(dataclasses.asdict(cfg)))
    on_gpu = device.type == "cuda"
    hardware = torch.cuda.get_device_name(device) if on_gpu else device.type
    tags = {"run_dir": str(out.resolve()), "device": hardware}
    tags["checkpoint_run"] = cfg.inference.checkpoint_run or cfg.name
    mlflow.set_tags(tags | _git_state())
    mlflow.log_artifact(str(out / "config.yaml"))


def load_best_model(cfg: Config, device: torch.device) -> nn.Module:
    path = (
        cfg.runtime.output_dir / (cfg.inference.checkpoint_run or cfg.name) / "best.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: train this config first")
    model = build_model(cfg.model).to(device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)["model"]
    )
    return model.eval()


def train(cfg: Config, device: torch.device) -> Path:
    """Train with the config, resuming its run directory if one exists; returns that directory."""
    out = run_dir(cfg)
    if cfg.inference.checkpoint_run is not None:
        print(
            f"{cfg.name} reuses the model of {cfg.inference.checkpoint_run}; nothing to train"
        )
        return out
    last = out / "last.pt"
    resume = (
        torch.load(last, map_location=device, weights_only=True)
        if last.exists()
        else None
    )
    if resume is not None and resume["epoch"] >= cfg.train.epochs:
        print(f"{out} already holds a finished run of {resume['epoch']} epochs")
        return out
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    set_track_meta(False)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    amp = cfg.train.amp and device.type == "cuda"

    splits = read_splits(cfg.data.splits_file)
    transform = train_transforms(cfg, device)
    transform.set_random_state(seed=cfg.seed)
    # crops are new tensors, so the cached volumes are shared without a copy per sample
    cache = {"cache_rate": float(cfg.runtime.cache_on_device), "num_workers": 1}
    cache |= {"copy_cache": False, "progress": False}
    train_set = CacheDataset(case_records(cfg, splits["train"]), transform, **cache)
    val_records = case_records(cfg, evaluation_cases(splits, "val"))
    val_set = CacheDataset(val_records, val_transforms(device), **cache)
    # samples already live on the device, so worker processes would only add copies
    loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = build_loss(cfg.train.loss)

    start, best, history = 0, -math.inf, []
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start, best = resume["epoch"], resume["best"]
        logged = pd.read_csv(out / "history.csv").to_dict("records")
        history = [record for record in logged if record["epoch"] <= start]
        print(f"resuming {out} after epoch {start}")

    with tracked_run(cfg):
        if resume is None:
            record_identity(cfg, device)

        for epoch in range(start + 1, cfg.train.epochs + 1):
            began = time.perf_counter()
            model.train()
            losses = []
            lr = optimizer.param_groups[0]["lr"]
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    logits = model(batch["image"])
                loss = loss_fn(logits.float(), batch["label"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # kept on the device so the loop never waits on a host sync per step
                losses.append(loss.detach())
            scheduler.step()
            record = {
                "epoch": epoch,
                "steps": len(losses),
                "lr": lr,
                "train_loss": torch.stack(losses).mean().item(),
            }

            if epoch % cfg.train.val_every == 0 or epoch == cfg.train.epochs:
                scores = validation_dice(model, val_set, cfg)
                record |= {
                    f"val_dice_{region}": value for region, value in scores.items()
                }
                record["val_dice_mean"] = float(np.mean(list(scores.values())))
                if record["val_dice_mean"] > best:
                    best = record["val_dice_mean"]
                    best_state = {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": best,
                    }
                    _save_atomic(best_state, out / "best.pt")
                last_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                }
                _save_atomic(last_state, last)

            record["epoch_seconds"] = time.perf_counter() - began
            history.append(record)
            pd.DataFrame(history).to_csv(
                out / "history.csv", index=False, lineterminator="\n"
            )
            mlflow.log_metrics(
                {k: v for k, v in record.items() if k != "epoch"}, step=epoch
            )
            shown = " ".join(f"{k} {v:.4g}" for k, v in record.items() if k != "epoch")
            print(f"epoch {epoch}/{cfg.train.epochs} {shown}", flush=True)

        mlflow.log_metric("best_val_dice_mean", best)
        mlflow.log_artifact(str(out / "history.csv"))
    return out
