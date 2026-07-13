from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from oc_pregrasp.data.rollout_dataset import YawPregraspRolloutDataset
from oc_pregrasp.models.scorer_mlp import PointNetScorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    model = PointNetScorer().to(device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    feature_mode = "raw"
    if isinstance(payload, dict):
        feature_mode = str(
            payload.get(
                "feature_mode",
                payload.get("args", {}).get("feature_mode", "raw"),
            )
        )
    return model, payload, feature_mode


def move_batch(batch, device: torch.device):
    return {
        "object_id": batch["object_id"],
        "point_cloud": batch["point_cloud"].to(device=device, dtype=torch.float32),
        "angle": batch["angle"].to(device=device, dtype=torch.float32),
        "angle_deg": batch["angle_deg"].cpu().numpy().astype(np.float32),
        "object_pose": batch["object_pose"].to(device=device, dtype=torch.float32),
        "label": batch["label"].to(device=device, dtype=torch.float32),
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(scores.shape[0], dtype=np.float64)
    pos_rank_sum = float(ranks[labels].sum())
    auc = (pos_rank_sum - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def rounded_angle_key(angle_deg: np.ndarray):
    return tuple(round(float(v), 6) for v in angle_deg.tolist())


def summarize_top1(candidate_stats):
    per_object = []
    for object_id, candidates in sorted(candidate_stats.items()):
        rows = []
        for angle_key, stats in candidates.items():
            count = stats["count"]
            if count <= 0:
                continue
            rows.append(
                {
                    "angle": angle_key,
                    "count": count,
                    "true_rate": stats["success_sum"] / count,
                    "pred_score": stats["score_sum"] / count,
                }
            )
        if not rows:
            continue
        mlp_row = max(rows, key=lambda row: row["pred_score"])
        oracle_row = max(rows, key=lambda row: row["true_rate"])
        uniform_micro = sum(row["true_rate"] * row["count"] for row in rows) / sum(
            row["count"] for row in rows
        )
        uniform_macro = sum(row["true_rate"] for row in rows) / len(rows)
        base_row = next(
            (
                row
                for row in rows
                if all(abs(row["angle"][i]) < 1e-6 for i in range(3))
            ),
            None,
        )
        per_object.append(
            {
                "object_id": object_id,
                "candidate_count": len(rows),
                "mlp_top1_angle": mlp_row["angle"],
                "mlp_top1_score": mlp_row["pred_score"],
                "mlp_top1_true_rate": mlp_row["true_rate"],
                "oracle_top1_angle": oracle_row["angle"],
                "oracle_top1_true_rate": oracle_row["true_rate"],
                "uniform_micro_true_rate": uniform_micro,
                "uniform_macro_true_rate": uniform_macro,
                "base_000_true_rate": None if base_row is None else base_row["true_rate"],
            }
        )
    return per_object


def mean_available(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return None
    return float(np.mean(values))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, payload, feature_mode = load_model(args.checkpoint, device)
    dataset = YawPregraspRolloutDataset(
        args.rollout_root, feature_mode=feature_mode
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    all_scores = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    candidate_stats = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "success_sum": 0.0, "score_sum": 0.0})
    )

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            logits = model(batch["point_cloud"], batch["angle"], batch["object_pose"])
            labels = batch["label"]
            scores = torch.sigmoid(logits)
            total_loss += float(criterion(logits, labels).detach().cpu())
            total_count += int(labels.numel())
            total_correct += int(((scores >= 0.5).float() == labels).sum().cpu())

            scores_np = scores.reshape(-1).detach().cpu().numpy()
            labels_np = labels.reshape(-1).detach().cpu().numpy()
            all_scores.append(scores_np)
            all_labels.append(labels_np)

            angle_deg = batch["angle_deg"].reshape(-1, 3)
            for object_id, angle, label, score in zip(
                batch["object_id"], angle_deg, labels_np, scores_np
            ):
                key = rounded_angle_key(angle)
                stats = candidate_stats[str(object_id)][key]
                stats["count"] += 1
                stats["success_sum"] += float(label)
                stats["score_sum"] += float(score)

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    per_object = summarize_top1(candidate_stats)
    summary = {
        "checkpoint": str(args.checkpoint),
        "rollout_root": str(args.rollout_root),
        "feature_mode": feature_mode,
        "samples": int(total_count),
        "objects": len(per_object),
        "bce": total_loss / max(total_count, 1),
        "accuracy_at_0_5": total_correct / max(total_count, 1),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "auc": binary_auc(labels, scores),
        "mlp_top1_true_rate_macro": mean_available(per_object, "mlp_top1_true_rate"),
        "oracle_top1_true_rate_macro": mean_available(
            per_object, "oracle_top1_true_rate"
        ),
        "uniform_micro_true_rate_macro": mean_available(
            per_object, "uniform_micro_true_rate"
        ),
        "uniform_macro_true_rate_macro": mean_available(
            per_object, "uniform_macro_true_rate"
        ),
        "base_000_true_rate_macro": mean_available(per_object, "base_000_true_rate"),
        "checkpoint_epoch": (
            int(payload.get("epoch", -1)) if isinstance(payload, dict) else None
        ),
        "per_object": per_object,
    }

    print(
        "Offline scorer eval: "
        f"samples={summary['samples']}, objects={summary['objects']}, "
        f"feature_mode={feature_mode}, "
        f"bce={summary['bce']:.4f}, acc={summary['accuracy_at_0_5']:.4f}, "
        f"auc={summary['auc'] if summary['auc'] is not None else 'NA'}, "
        f"positive={summary['positive_rate']:.4f}",
        flush=True,
    )
    print(
        "Top1 by learned scorer: "
        f"mlp={summary['mlp_top1_true_rate_macro']:.4f}, "
        f"oracle={summary['oracle_top1_true_rate_macro']:.4f}, "
        f"uniform={summary['uniform_micro_true_rate_macro']:.4f}, "
        f"base000={summary['base_000_true_rate_macro']}",
        flush=True,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved offline eval to {args.output_json}", flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
