#!/usr/bin/env python3
"""Select a KDS-Former checkpoint using validation-set WER."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_sort_key(path: str) -> tuple[int, int, str]:
    name = Path(path).name
    match = re.search(r"epoch(\d+)", name)
    if match:
        return 0, int(match.group(1)), name
    if "best" in name:
        return 1, 10**9, name
    return 2, 10**9, name


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def eval_command(args: argparse.Namespace, checkpoint: str, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "evaluation/eval_ksl_manifest.py"),
        "--manifest",
        args.manifest,
        "--split",
        "val",
        "--checkpoint",
        checkpoint,
        "--output",
        str(output),
        "--vocab-scope",
        "manifest_train",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--downsample",
        str(args.downsample),
        "--face-mode",
        args.face_mode,
        "--classifier",
        args.classifier,
        "--norm-scale",
        str(args.norm_scale),
        "--d-hand",
        str(args.d_hand),
        "--d-body",
        str(args.d_body),
        "--d-fusion",
        str(args.d_fusion),
        "--nhead",
        str(args.nhead),
        "--branch-layers",
        str(args.branch_layers),
        "--fusion-layers",
        str(args.fusion_layers),
        "--dim-feedforward",
        str(args.dim_feedforward),
    ]
    if args.add_motion:
        cmd.append("--add-motion")
    if args.part_hand_norm:
        cmd.append("--part-hand-norm")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def write_summary(args: argparse.Namespace, rows: list[dict]) -> dict:
    if not rows:
        raise SystemExit("No validation rows were produced.")
    best = min(rows, key=lambda row: row["WER"])
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": args.run_name,
        "family": "kds",
        "selection_metric": "validation WER",
        "selection_split": "val",
        "split": "val",
        "manifest": args.manifest,
        "model": args.model,
        "protocol": args.protocol,
        "seed": args.seed,
        "checkpoint_glob": args.checkpoint_glob,
        "best_checkpoint": best["checkpoint"],
        "best_WER": best["WER"],
        "best_row": best,
        "rows": rows,
        "kds_config": {
            "downsample": args.downsample,
            "add_motion": args.add_motion,
            "part_hand_norm": args.part_hand_norm,
            "face_mode": args.face_mode,
            "classifier": args.classifier,
            "norm_scale": args.norm_scale,
            "d_hand": args.d_hand,
            "d_body": args.d_body,
            "d_fusion": args.d_fusion,
            "nhead": args.nhead,
            "branch_layers": args.branch_layers,
            "fusion_layers": args.fusion_layers,
            "dim_feedforward": args.dim_feedforward,
        },
    }
    summary_path = out_dir / f"{args.run_name}_val_wer_selection.json"
    csv_path = out_dir / f"{args.run_name}_val_wer_selection.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--model", required=True, choices=["kds_s", "kds_m"], help="Model label recorded in selection metadata.")
    parser.add_argument("--protocol", required=True, choices=["k1", "k2"], help="Protocol label recorded in selection metadata.")
    parser.add_argument("--seed", required=True, type=int, choices=[42, 43, 2024], help="Training seed recorded in selection metadata.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/val_wer_selection"))
    parser.add_argument("--split", default="val", choices=["val"], help="Checkpoint selection is validation-only.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--add-motion", action="store_true")
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("Checkpoint selection must use the validation split only.")

    checkpoints = sorted(glob.glob(args.checkpoint_glob), key=checkpoint_sort_key)
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched: {args.checkpoint_glob}")

    run_dir = Path(args.output_dir) / args.run_name / "val"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for checkpoint in checkpoints:
        stem = Path(checkpoint).stem
        output = run_dir / f"{stem}_val.json"
        if not output.exists() or args.force:
            run_command(eval_command(args, checkpoint, output), args.dry_run)
        if args.dry_run:
            continue
        result = json.loads(output.read_text(encoding="utf-8"))
        rows.append(
            {
                "checkpoint": checkpoint,
                "split": "val",
                "WER": result["WER"],
                "output": str(output),
            }
        )

    if args.dry_run:
        print(json.dumps({"run_name": args.run_name, "checkpoints": checkpoints}, indent=2))
        return
    write_summary(args, rows)


if __name__ == "__main__":
    main()
