from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from oc_pregrasp.data.rollout_dataset import YawPregraspRolloutDataset
from oc_pregrasp.models.scorer_mlp import PointNetScorer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-mode", default="raw", choices=["raw", "pca_short"])
    return parser.parse_args()


def make_splits(size: int, val_fraction: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(size))
    rng.shuffle(indices)
    val_size = max(1, int(round(size * val_fraction)))
    return indices[val_size:], indices[:val_size]


def move_batch(batch, device: torch.device):
    return {
        "point_cloud": batch["point_cloud"].to(device=device, dtype=torch.float32),
        "angle": batch["angle"].to(device=device, dtype=torch.float32),
        "object_pose": batch["object_pose"].to(device=device, dtype=torch.float32),
        "label": batch["label"].to(device=device, dtype=torch.float32),
    }


def _progress(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def run_epoch(model, loader, criterion, device, optimizer=None, desc: str = ""):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    total_positive = 0
    progress = _progress(
        loader,
        enabled=True,
        desc=desc,
        dynamic_ncols=True,
        leave=False,
    )
    with torch.set_grad_enabled(training):
        for raw_batch in progress:
            batch = move_batch(raw_batch, device)
            logits = model(
                batch["point_cloud"], batch["angle"], batch["object_pose"]
            )
            loss = criterion(logits, batch["label"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            labels = batch["label"]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            count = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
            total_correct += int((preds == labels).sum().detach().cpu())
            total_positive += int(labels.sum().detach().cpu())
            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{total_loss / max(total_count, 1):.4f}",
                    acc=f"{total_correct / max(total_count, 1):.4f}",
                    pos=f"{total_positive / max(total_count, 1):.4f}",
                )
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "positive_rate": total_positive / max(total_count, 1),
    }


def load_checkpoint(path: Path, model, optimizer, device: torch.device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    if isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    start_epoch = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
    best_val = -1.0
    if isinstance(payload, dict) and isinstance(payload.get("val"), dict):
        best_val = float(payload["val"].get("accuracy", -1.0))
    return start_epoch, best_val


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = YawPregraspRolloutDataset(
        args.rollout_root, feature_mode=args.feature_mode
    )
    train_indices, val_indices = make_splits(
        len(dataset), args.val_fraction, args.seed
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PointNetScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    start_epoch = 0
    if args.resume_checkpoint is not None:
        start_epoch, best_val = load_checkpoint(
            args.resume_checkpoint, model, optimizer, device
        )
        print(
            f"resumed scorer from {args.resume_checkpoint} "
            f"(start_epoch={start_epoch}, best_val_acc={best_val:.4f})",
            flush=True,
        )

    history = []
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"epoch {epoch:03d} train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            desc=f"epoch {epoch:03d} val",
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "feature_mode": args.feature_mode,
                    "epoch": epoch,
                    "args": vars(args),
                    "val": val_metrics,
                },
                args.output_dir / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "feature_mode": args.feature_mode,
                "epoch": epoch,
                "args": vars(args),
                "val": val_metrics,
            },
            args.output_dir / "last.pt",
        )

    metrics_path = args.output_dir / "metrics.json"
    if args.resume_checkpoint is not None and metrics_path.is_file():
        try:
            old_history = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(old_history, list):
                history = old_history + history
        except Exception:
            pass
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved best checkpoint to {args.output_dir / 'best.pt'}", flush=True)
    print(f"saved last checkpoint to {args.output_dir / 'last.pt'}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
