#!/usr/bin/env python3
"""Export top-K recommendations using the exact RLMRec test protocol.

Run from Colab (after mounting Drive and cloning the original RLMRec repo):
  python src/inference/export_topk.py \
    --repo /content/RLMRec \
    --root /content/drive/MyDrive/Data_SNA_Group14 \
    --model lightgcn --seed 2023 --k 20

The script works with the original HKUDS/RLMRec commit
22413752246de3dee8ab0d509f7f7a8889080f95.

Important: Top-K only considers users who have at least one test interaction,
excludes training interactions, and does NOT exclude validation interactions,
matching the original test pipeline. Do not use test labels to generate ranks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PINNED_COMMIT = "22413752246de3dee8ab0d509f7f7a8889080f95"
ALLOWED_MODELS = ("lightgcn", "lightgcn_plus")


def parse_args():
    parser = argparse.ArgumentParser(description="Export E1 test Top-K from RLMRec checkpoint")
    parser.add_argument("--repo", required=True, type=Path, help="Path to cloned original HKUDS/RLMRec")
    parser.add_argument("--root", required=True, type=Path, help="Data_SNA_Group14 root in Google Drive")
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-commit-check", action="store_true", help="Only for debugging non-pinned original source")
    return parser.parse_args()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def get_test_metrics(metric, ranked_item_batches, ground_truth_batches, user_count):
    """Aggregate through RLMRec's original Metric.eval_batch implementation."""
    final = {name: np.zeros(len(metric.k), dtype=np.float64) for name in metric.metrics}
    for items, ground_truth in zip(ranked_item_batches, ground_truth_batches):
        result = metric.eval_batch((torch.from_numpy(items), ground_truth), metric.k)
        for name in metric.metrics:
            final[name] += result[name] / user_count
    return {
        f"{name}@{k}": float(final[name][index])
        for name in metric.metrics
        for index, k in enumerate(metric.k)
    }


def check_against_e1(actual_metrics, e1_metrics_path: Path, tolerance: float = 1e-5):
    """Fail on a disagreement with the original E1 test results."""
    if not e1_metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing {e1_metrics_path}; cannot verify exported rankings against E1"
        )
    with e1_metrics_path.open(encoding="utf-8") as handle:
        original = json.load(handle)
    expected = original["test_metrics"]
    compared = {}
    for key, reference in expected.items():
        if key not in actual_metrics:
            raise AssertionError(f"Metric {key} from E1 is unavailable in export")
        observed = float(actual_metrics[key])
        reference = float(reference)
        diff = abs(observed - reference)
        compared[key] = {
            "e1": reference,
            "export": observed,
            "abs_difference": diff,
            "passed": math.isclose(observed, reference, rel_tol=1e-5, abs_tol=tolerance),
        }
    if not all(v["passed"] for v in compared.values()):
        details = {k: v for k, v in compared.items() if not v["passed"]}
        raise AssertionError(
            "Export metrics differ from E1; inspect checkpoint/protocol: "
            + json.dumps(details, ensure_ascii=False)
        )
    return compared


def main():
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Enable Colab GPU first")

    repo = args.repo.expanduser().resolve()
    root = args.root.expanduser().resolve()
    dataset_dir = root / "data" / "raw" / "amazon"
    run_dir = root / "experiments" / "E1" / args.model / f"seed_{args.seed}"
    ckpt = run_dir / "checkpoints" / f"{args.model}-amazon-{args.seed}.pth"
    e1_metrics = run_dir / "metrics.json"
    export_dir = root / "experiments" / "predictions" / args.model / f"seed_{args.seed}"
    topk_file = export_dir / "topk_test.npz"
    meta_file = export_dir / "export_meta.json"
    verification_file = export_dir / "export_verification.json"

    if not repo.is_dir():
        raise FileNotFoundError(f"Missing RLMRec repository: {repo}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {ckpt}")
    if not e1_metrics.is_file():
        raise FileNotFoundError(f"Missing E1 metrics.json: {e1_metrics}")
    for item in ("trn_mat.pkl", "val_mat.pkl", "tst_mat.pkl", "usr_emb_np.pkl", "itm_emb_np.pkl"):
        if not (dataset_dir / item).is_file():
            raise FileNotFoundError(f"Missing dataset file: {dataset_dir / item}")
    commit = git_head(repo)
    if commit != PINNED_COMMIT and not args.skip_commit_check:
        raise RuntimeError(f"RLMRec HEAD {commit} != pinned {PINNED_COMMIT}")

    # The RLMRec configurator loads its YAML and embedding data using CWD,
    # and parses --model/--dataset/--device/--seed from sys.argv at import time.
    os.chdir(repo)
    sys.path.insert(0, str(repo / "encoder"))
    sys.argv = [
        str(repo / "encoder" / "train_encoder.py"),
        "--model", args.model,
        "--dataset", "amazon",
        "--seed", str(args.seed),
        "--device", args.device,
        "--cuda", "0",
    ]

    from config.configurator import configs
    from data_utils.build_data_handler import build_data_handler
    from models.bulid_model import build_model
    from trainer.metrics import Metric

    handler = build_data_handler()
    handler.load_data()
    test_loader = handler.test_dataloader
    model = build_model(handler).to(configs["device"])
    # Prefer weights_only=True; original Trainer saves model.state_dict().
    weights = torch.load(ckpt, map_location=configs["device"], weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    if hasattr(model, "final_embeds"):
        model.final_embeds = None  # Fresh inference from restored parameters.
    if hasattr(model, "is_training"):
        model.is_training = False

    metric = Metric()
    eval_ks = sorted({int(k) for k in metric.k})
    if args.k < max(eval_ks):
        raise ValueError(f"--k must be >= max original evaluation K={max(eval_ks)}")
    item_num = int(configs["data"]["item_num"])
    if args.k > item_num:
        raise ValueError(f"--k={args.k} exceeds item count {item_num}")
    user_num = int(configs["data"]["user_num"])

    ids_batches = []
    items_batches = []
    scores_batches = []
    ground_truth_batches = []
    seen_user_ids = set()
    n_users = len(test_loader.dataset.test_users)

    # Mirror RLMRec Metric.eval(): iterate test_loader, use model.full_predict
    # to mask train, and call torch.topk over all item scores.
    with torch.no_grad():
        for batch in test_loader:
            if not isinstance(batch, (list, tuple)):
                batch = [batch]
            users = batch[0].cpu().numpy().astype(np.int64)
            if len(set(users.tolist())) != len(users):
                raise AssertionError("Duplicate user IDs inside a test batch")
            if seen_user_ids.intersection(users.tolist()):
                raise AssertionError("Duplicate test user IDs across batches")
            seen_user_ids.update(users.tolist())

            batch_data = [x.long().to(configs["device"]) for x in batch]
            predictions = model.full_predict(batch_data)
            # Repo Metric._mask_history_pos is a no-op for AllRankTstData,
            # whose only mask is the train mask passed to full_predict.
            best_scores, best_items = torch.topk(predictions, k=args.k, dim=1)
            best_items_cpu = best_items.cpu().numpy().astype(np.int32)
            best_scores_cpu = best_scores.cpu().numpy().astype(np.float32)

            # Strong guard against inadvertently recommending train items.
            train_mask = batch[1].cpu().numpy()
            if np.any(np.take_along_axis(train_mask, best_items_cpu, axis=1) != 0):
                raise AssertionError("Train-masked item appeared in Top-K")
            ground_truth = [
                list(test_loader.dataset.user_pos_lists[int(uid)]) for uid in users
            ]
            ids_batches.append(users)
            items_batches.append(best_items_cpu)
            scores_batches.append(best_scores_cpu)
            ground_truth_batches.append(ground_truth)

    if len(seen_user_ids) != n_users:
        raise AssertionError(f"Exported {len(seen_user_ids)} != {n_users} test users")

    users_np = np.concatenate(ids_batches, axis=0)
    items_np = np.concatenate(items_batches, axis=0)
    scores_np = np.concatenate(scores_batches, axis=0)
    assert items_np.shape == (n_users, args.k)
    assert scores_np.shape == items_np.shape
    assert np.isfinite(scores_np).all()
    assert users_np.min() >= 0 and users_np.max() < user_num
    assert items_np.min() >= 0 and items_np.max() < item_num

    # Reproduce official metrics from exactly the exported rankings.
    actual_metrics = get_test_metrics(metric, items_batches, ground_truth_batches, n_users)
    comparison = check_against_e1(actual_metrics, e1_metrics)

    # Only write files after verification passes to avoid false-looking artifacts.
    export_dir.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (topk_file, meta_file, verification_file)):
        raise FileExistsError(
            f"Export output already exists in {export_dir}; move/archive it before rerunning"
        )
    np.savez_compressed(
        topk_file,
        user_ids=users_np,
        item_ids=items_np,
        scores=scores_np,
    )
    meta = {
        "model": args.model,
        "dataset": "amazon",
        "seed": args.seed,
        "source_repo": "https://github.com/HKUDS/RLMRec",
        "rlmrec_commit": commit,
        "source_checkpoint": str(ckpt),
        "split": "test",
        "top_k": args.k,
        "test_user_count": n_users,
        "total_users": user_num,
        "total_items": item_num,
        "ranking_candidate_pool": "all items except items observed in train for each user",
        "masked_splits": ["train"],
        "validation_masked": False,
        "test_labels_used_to_rank": False,
        "files": {"topk": topk_file.name},
        "npz_fields": {
            "user_ids": "int64, shape (n_test_users,), one row per test user",
            "item_ids": f"int32, shape (n_test_users, {args.k}), ranked item IDs",
            "scores": f"float32, shape (n_test_users, {args.k}), model scores after masking",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    verification = {
        "status": "passed",
        "validation_against": str(e1_metrics),
        "absolute_tolerance": 1e-5,
        "test_metrics_from_export": actual_metrics,
        "compared_metrics": comparison,
        "excluded_train_items": True,
        "test_users_unique": True,
    }
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    verification_file.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== TOP-K EXPORT VERIFIED ===")
    print(f"Model: {args.model}; seed: {args.seed}; K={args.k}")
    print(f"Test users: {n_users}; items: {item_num}")
    for name, value in actual_metrics.items():
        print(f"{name}: {value:.6f}")
    print("All E1 metric comparisons: PASS")
    print(f"Output: {export_dir}")
    print("Files: topk_test.npz, export_meta.json, export_verification.json")


if __name__ == "__main__":
    main()
