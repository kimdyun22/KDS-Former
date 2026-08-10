#!/usr/bin/env python3
"""Train KDS-Former on a KSL manifest split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_MODULE_DIR = REPO_ROOT / "datasets"
for item in [str(DATASET_MODULE_DIR), str(REPO_ROOT)]:
    if item not in sys.path:
        sys.path.insert(0, item)

from dataset_ksl_manifest import KSLManifestDataset, cslr_collate_fn  # noqa: E402
from training.losses import CombinedLoss  # noqa: E402
from models.kds_former import KDSFormer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    skip_nonfinite_gradients: bool = False,
    max_skipped_batches: int = 8,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    ctc = 0.0
    desc = f"{'Train' if training else 'Val'} epoch {epoch}"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True)
    skipped = 0

    for batch_idx, batch in enumerate(pbar):
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lens = batch["target_lens"].to(device, non_blocking=True)
        actual_frames = batch["actual_frames"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_amp):
                gloss_logits = model(skeleton, actual_frames=actual_frames)
                loss = criterion(
                    student_gloss_logits=gloss_logits,
                    targets=targets,
                    target_lengths=target_lens,
                    actual_frames=actual_frames,
                )

        if training:
            assert scaler is not None
            scaler.scale(loss["total"]).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                print(
                    f"[skip] non-finite KDS gradient epoch={epoch} batch={batch_idx} "
                    f"grad_norm={float(grad_norm)} skipped={skipped}/{max_skipped_batches}",
                    flush=True,
                )
                if not skip_nonfinite_gradients or skipped > max_skipped_batches:
                    raise RuntimeError(f"Non-finite KDS gradient at epoch={epoch}")
                continue
            scaler.step(optimizer)
            scaler.update()

        total += float(loss["total"].item())
        ctc += float(loss["ctc"])
        pbar.set_postfix(loss=f"{loss['total'].item():.4f}", ctc=f"{loss['ctc']:.4f}")

    denom = max(len(loader), 1)
    return total / denom, ctc / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True, choices=["k1", "k2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--add-motion", action="store_true")
    parser.add_argument("--temporal-drop-prob", type=float, default=0.0)
    parser.add_argument("--jitter-sigma", type=float, default=0.0)
    parser.add_argument("--part-hand-norm", action="store_true")
    parser.add_argument("--face-mode", choices=["full", "full_face", "mouth_only", "no_face"], default="full")
    parser.add_argument("--classifier", choices=["linear", "normboth"], default="linear")
    parser.add_argument("--norm-scale", type=float, default=32.0)
    parser.add_argument("--d-hand", type=int, default=128)
    parser.add_argument("--d-body", type=int, default=128)
    parser.add_argument("--d-fusion", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--branch-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-nonfinite-gradients", action="store_true")
    parser.add_argument("--max-skipped-batches-per-epoch", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--result-dir", default="outputs/train_summaries")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.protocol}_seed{args.seed}{args.run_suffix}"
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    result_dir = Path(args.result_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = KSLManifestDataset(
        args.manifest,
        "train",
        max_frames=args.max_frames,
        vocab_scope="manifest_train",
        downsample=args.downsample,
        add_motion=args.add_motion,
        temporal_drop_prob=args.temporal_drop_prob,
        jitter_sigma=args.jitter_sigma,
        augment=True,
        part_hand_norm=args.part_hand_norm,
        face_mode=args.face_mode,
        limit=args.limit_train,
    )
    val_dataset = KSLManifestDataset(
        args.manifest,
        "val",
        max_frames=args.max_frames,
        vocab_scope="manifest_train",
        downsample=args.downsample,
        add_motion=args.add_motion,
        part_hand_norm=args.part_hand_norm,
        face_mode=args.face_mode,
        limit=args.limit_val,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cslr_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cslr_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    input_channels = 6 if args.add_motion else 2
    model = KDSFormer(
        num_classes=train_dataset.vocab_size,
        d_hand=args.d_hand,
        d_body=args.d_body,
        d_fusion=args.d_fusion,
        nhead=args.nhead,
        branch_layers=args.branch_layers,
        fusion_layers=args.fusion_layers,
        dim_feedforward=args.dim_feedforward,
        input_channels=input_channels,
        classifier=args.classifier,
        norm_scale=args.norm_scale,
    ).to(device)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    criterion = CombinedLoss(lambda_ctc=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.warmup_epochs, 1),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)

    print(json.dumps({
        "run": run_name,
        "manifest": args.manifest,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "vocab_size": train_dataset.vocab_size,
        "val_oov_rate": val_dataset.oov_rate,
        "trainable_params": trainable_params,
        "amp": not args.no_amp,
        "warmup_epochs": args.warmup_epochs,
        "skip_nonfinite_gradients": args.skip_nonfinite_gradients,
        "downsample": args.downsample,
        "add_motion": args.add_motion,
        "temporal_drop_prob": args.temporal_drop_prob,
        "jitter_sigma": args.jitter_sigma,
        "part_hand_norm": args.part_hand_norm,
        "face_mode": args.face_mode,
        "classifier": args.classifier,
        "norm_scale": args.norm_scale,
        "model_dims": {
            "d_hand": args.d_hand,
            "d_body": args.d_body,
            "d_fusion": args.d_fusion,
            "nhead": args.nhead,
            "branch_layers": args.branch_layers,
            "fusion_layers": args.fusion_layers,
            "dim_feedforward": args.dim_feedforward,
            "input_channels": input_channels,
        },
    }, ensure_ascii=False, indent=2))

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ctc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            epoch,
            optimizer,
            scaler,
            use_amp=not args.no_amp,
            skip_nonfinite_gradients=args.skip_nonfinite_gradients,
            max_skipped_batches=args.max_skipped_batches_per_epoch,
        )
        scheduler.step()
        with torch.no_grad():
            val_loss, val_ctc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                use_amp=not args.no_amp,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ctc": train_ctc,
            "val_loss": val_loss,
            "val_ctc": val_ctc,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{run_name}_epoch{epoch}.pt")
        should_stop = False
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), ckpt_dir / f"{run_name}_best.pt")
            print(f"[best] epoch={epoch} val_loss={val_loss:.6f}")
        else:
            patience_count += 1
            print(f"[patience] {patience_count}/{args.patience}")
            if patience_count >= args.patience:
                print("[early_stop]")
                should_stop = True

        summary = {
            "run": run_name,
            "protocol": args.protocol,
            "seed": args.seed,
            "manifest": args.manifest,
            "checkpoint": str(ckpt_dir / f"{run_name}_best.pt"),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "history": history,
            "vocab_size": train_dataset.vocab_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "val_oov_rate": val_dataset.oov_rate,
            "trainable_params": trainable_params,
            "amp": not args.no_amp,
            "warmup_epochs": args.warmup_epochs,
            "skip_nonfinite_gradients": args.skip_nonfinite_gradients,
            "downsample": args.downsample,
            "add_motion": args.add_motion,
            "temporal_drop_prob": args.temporal_drop_prob,
            "jitter_sigma": args.jitter_sigma,
            "part_hand_norm": args.part_hand_norm,
            "face_mode": args.face_mode,
            "classifier": args.classifier,
            "norm_scale": args.norm_scale,
            "model_dims": {
                "d_hand": args.d_hand,
                "d_body": args.d_body,
                "d_fusion": args.d_fusion,
                "branch_layers": args.branch_layers,
                "fusion_layers": args.fusion_layers,
                "dim_feedforward": args.dim_feedforward,
                "input_channels": input_channels,
            },
        }
        (result_dir / f"{run_name}_train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if should_stop:
            break

    print(f"Training complete: {run_name} best_epoch={best_epoch} best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
