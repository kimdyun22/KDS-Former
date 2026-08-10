#!/usr/bin/env python3
"""Evaluate the test split from a validation-WER checkpoint selection file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], dry_run: bool) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def repo_relative(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except Exception:
        return str(candidate)


def validate_selection(selection: dict) -> str:
    selection_split = selection.get("selection_split", selection.get("split"))
    selection_metric = selection.get("selection_metric")
    best_checkpoint = selection.get("best_checkpoint")

    if selection_split != "val":
        raise ValueError("The selection file must record selection_split='val'.")
    if selection_metric != "validation WER":
        raise ValueError("The selection file must use validation WER.")
    if not best_checkpoint:
        raise ValueError("The selection file does not contain best_checkpoint.")
    return str(best_checkpoint)


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def command_for(selection: dict, manifest: str, output: Path, batch_size: int, num_workers: int) -> list[str]:
    config = selection.get("kds_config", {})
    best_checkpoint = validate_selection(selection)
    cmd = [
        sys.executable,
        str(ROOT / "evaluation/eval_ksl_manifest.py"),
        "--manifest",
        manifest,
        "--split",
        "test",
        "--checkpoint",
        best_checkpoint,
        "--output",
        str(output),
        "--vocab-scope",
        "manifest_train",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--downsample",
        str(config.get("downsample", 1)),
        "--face-mode",
        str(config.get("face_mode", "full")),
        "--classifier",
        str(config.get("classifier", "linear")),
        "--norm-scale",
        str(config.get("norm_scale", 32.0)),
        "--d-hand",
        str(config.get("d_hand", 128)),
        "--d-body",
        str(config.get("d_body", 128)),
        "--d-fusion",
        str(config.get("d_fusion", 256)),
        "--nhead",
        str(config.get("nhead", 8)),
        "--branch-layers",
        str(config.get("branch_layers", 2)),
        "--fusion-layers",
        str(config.get("fusion_layers", 2)),
        "--dim-feedforward",
        str(config.get("dim_feedforward", 512)),
    ]
    if config.get("add_motion", False):
        cmd.append("--add-motion")
    if config.get("part_hand_norm", False):
        cmd.append("--part-hand-norm")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, help="Validation-WER selection JSON.")
    parser.add_argument("--manifest", help="Manifest to evaluate. Defaults to the manifest recorded in selection.")
    parser.add_argument("--output", help="Output JSON path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-manifest-override",
        action="store_true",
        help="Allow evaluating a manifest different from the one recorded in the selection JSON.",
    )
    args = parser.parse_args()

    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    recorded_manifest = selection.get("manifest")
    manifest = args.manifest or recorded_manifest
    if not manifest:
        raise SystemExit("--manifest is required when the selection JSON does not record a manifest.")
    if args.manifest and recorded_manifest and not args.allow_manifest_override:
        if not same_path(args.manifest, recorded_manifest):
            raise ValueError("Manifest mismatch between the selection file and evaluation request.")

    if args.output:
        output = Path(args.output)
    else:
        run_name = selection.get("run_name", selection_path.stem)
        output = ROOT / "outputs/test_valwer_v2" / run_name / f"{run_name}_test_valwer_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        print(f"[SKIP] exists {output}")
        return
    run(command_for(selection, manifest, output, args.batch_size, args.num_workers), args.dry_run)
    if args.dry_run:
        return

    result = json.loads(output.read_text(encoding="utf-8"))
    result.update(
        {
            "model": selection.get("model"),
            "protocol": selection.get("protocol"),
            "train_seed": selection.get("seed"),
            "checkpoint_source": "validation_selection",
            "selection_split": selection.get(
                "selection_split",
                selection.get("split"),
            ),
            "selection_metric": selection.get("selection_metric"),
            "selection_file": repo_relative(selection_path),
        }
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
