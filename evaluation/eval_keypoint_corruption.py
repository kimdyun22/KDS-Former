#!/usr/bin/env python3
"""Evaluate test-time keypoint corruption for KDS-Former checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENT_ROOT = REPO_ROOT

from datasets.dataset_ksl_manifest import KSLManifestDataset, cslr_collate_fn, parse_label_file  # noqa: E402
from evaluation.eval_ksl_manifest import metrics  # noqa: E402
from models.kds_former import KDSFormer  # noqa: E402

SEEDS = [42, 43, 2024]


def full_reference_from_label(label_path: str, gloss2idx: dict[str, int]) -> str:
    _, glosses = parse_label_file(label_path)
    return " ".join(gloss if gloss in gloss2idx else "<OOV>" for gloss in glosses)


def decode_ctc(logits: torch.Tensor, actual_frames: torch.Tensor, idx2gloss: dict[int, str]) -> list[str]:
    pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    decoded: list[str] = []
    for batch_idx in range(pred.shape[0]):
        seq = pred[batch_idx, : int(actual_frames[batch_idx].item())]
        out: list[str] = []
        last = -1
        for token in seq:
            idx = int(token.item())
            if idx != 0 and idx != last:
                out.append(idx2gloss.get(idx, "<UNK>"))
            last = idx
        decoded.append(" ".join(out))
    return decoded


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(ref) + 1)] for _ in range(len(hyp) + 1)
    ]
    for i in range(1, len(hyp) + 1):
        prev = dp[i - 1][0]
        dp[i][0] = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)
    for j in range(1, len(ref) + 1):
        prev = dp[0][j - 1]
        dp[0][j] = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])
    for i in range(1, len(hyp) + 1):
        for j in range(1, len(ref) + 1):
            if hyp[i - 1] == ref[j - 1]:
                dist, sub, dele, ins = dp[i - 1][j - 1]
                candidates = [(dist, sub, dele, ins)]
            else:
                dist, sub, dele, ins = dp[i - 1][j - 1]
                candidates = [(dist + 1, sub + 1, dele, ins)]
            dist, sub, dele, ins = dp[i][j - 1]
            candidates.append((dist + 1, sub, dele + 1, ins))
            dist, sub, dele, ins = dp[i - 1][j]
            candidates.append((dist + 1, sub, dele, ins + 1))
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    _, substitutions, deletions, insertions = dp[len(hyp)][len(ref)]
    return substitutions, deletions, insertions


def corpus_edit_counts(refs: list[str], hyps: list[str]) -> dict[str, int]:
    substitutions = 0
    deletions = 0
    insertions = 0
    reference_tokens = 0
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        sub, dele, ins = edit_counts(ref_tokens, hyp_tokens)
        substitutions += sub
        deletions += dele
        insertions += ins
        reference_tokens += len(ref_tokens)
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_tokens": reference_tokens,
    }


def git_commit() -> str:
    for repo in [REPO_ROOT, EXPERIMENT_ROOT]:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            continue
    return "unknown"


def repo_relative(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except Exception:
        try:
            return str(candidate.resolve().relative_to(EXPERIMENT_ROOT))
        except Exception:
            return str(candidate)


def mouth_only_mask(skeleton: torch.Tensor) -> torch.Tensor:
    keep = torch.zeros(skeleton.shape[2], dtype=torch.bool, device=skeleton.device)
    keep[:25] = True
    keep[25 + 48 : 25 + 67] = True
    keep[95:137] = True
    out = skeleton.clone()
    out[:, :, ~keep, :] = 0.0
    return out


def apply_corruption(
    skeleton: torch.Tensor,
    actual_frames: torch.Tensor,
    observed_mask: torch.Tensor,
    corruption: str,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if severity <= 0:
        return skeleton
    out = skeleton.clone()
    frame_mask = torch.arange(out.shape[1], device=out.device).unsqueeze(0) < actual_frames.unsqueeze(1)
    eligible = frame_mask[:, :, None] & observed_mask.to(device=out.device, dtype=torch.bool)
    if corruption == "dropout":
        drop = torch.rand(eligible.shape, generator=generator, device=out.device) < severity
        out = torch.where((eligible & drop)[:, :, :, None], torch.zeros_like(out), out)
    elif corruption == "gaussian":
        noise = torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * severity
        out = torch.where(eligible[:, :, :, None], out + noise, out)
    else:
        raise ValueError(f"unknown corruption: {corruption}")
    return out


def load_state_dict(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if state and next(iter(state)).startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    return state


def default_selection_path(run_name: str) -> Path:
    return EXPERIMENT_ROOT / "outputs" / "val_wer_selection" / run_name / f"{run_name}_val_wer_selection.json"


def load_selection(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_selection(selection: dict) -> str:
    selection_split = selection.get("selection_split", selection.get("split"))
    selection_metric = selection.get("selection_metric")
    best_checkpoint = selection.get("best_checkpoint")
    if selection_split != "val":
        raise ValueError("Corruption evaluation requires a validation-selected checkpoint.")
    if selection_metric != "validation WER":
        raise ValueError("Corruption evaluation requires validation-WER selection.")
    if not best_checkpoint:
        raise ValueError("The selection file does not contain best_checkpoint.")
    return str(best_checkpoint)


def validate_severity(corruption: str, severity: float) -> None:
    if corruption == "dropout" and not 0.0 <= severity <= 1.0:
        raise ValueError("Dropout severity must be between 0 and 1.")
    if corruption == "gaussian" and severity < 0.0:
        raise ValueError("Gaussian severity must be non-negative.")


def default_manifest_path(protocol: str, seed: int) -> Path:
    names = {
        "k1": f"seen_signer_unseen_sentence_seed{seed}.json",
        "k2": f"unseen_signer_unseen_sentence_seed{seed}.json",
    }
    return EXPERIMENT_ROOT / "manifests" / names[protocol]


def run_name_for(model_name: str, protocol: str, seed: int) -> str:
    if model_name == "kds_s":
        return f"{protocol}_seed{seed}_kds_former_s"
    if model_name == "kds_m":
        return f"{protocol}_seed{seed}_kds_former_m_mouth_only"
    raise ValueError(f"unknown model: {model_name}")


def build_kds_model(model_name: str, vocab_size: int) -> KDSFormer:
    if model_name == "kds_s":
        return KDSFormer(num_classes=vocab_size)
    return KDSFormer(
        num_classes=vocab_size,
        d_hand=256,
        d_body=256,
        d_fusion=512,
        nhead=8,
        branch_layers=3,
        fusion_layers=3,
        dim_feedforward=1024,
        input_channels=2,
        classifier="linear",
        norm_scale=32.0,
    )


def main() -> None:
    global EXPERIMENT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["kds_s", "kds_m"])
    parser.add_argument("--protocol", required=True, choices=["k1", "k2"])
    parser.add_argument("--train-seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--corruption", required=True, choices=["dropout", "gaussian"])
    parser.add_argument("--severity", type=float, required=True)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None, help="Explicit KSL manifest path. Overrides protocol-based lookup.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Overrides validation-selection lookup.")
    parser.add_argument("--selection-file", default=None, help="Explicit validation-WER selection JSON.")
    parser.add_argument("--run-name", default=None, help="Explicit run name for provenance and default selection lookup.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--experiment-root",
        default=str(REPO_ROOT),
        help="Root containing manifests, checkpoints, and validation-WER selections.",
    )
    args = parser.parse_args()
    EXPERIMENT_ROOT = Path(args.experiment_root).resolve()
    try:
        validate_severity(args.corruption, args.severity)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or run_name_for(args.model, args.protocol, args.train_seed)

    selection_split = None
    selection_metric = None
    selection_file: Path | None = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        checkpoint_source = "explicit_checkpoint"
    else:
        if args.selection_file is not None:
            selection_file = Path(args.selection_file)
        else:
            selection_file = default_selection_path(run_name)
        selection = load_selection(selection_file)
        checkpoint = validate_selection(selection)
        if selection.get("protocol") is not None and selection.get("protocol") != args.protocol:
            raise ValueError("Protocol mismatch between the selection file and corruption evaluation request.")
        if selection.get("seed") is not None and int(selection["seed"]) != args.train_seed:
            raise ValueError("Seed mismatch between the selection file and corruption evaluation request.")
        if selection.get("model") is not None and selection.get("model") != args.model:
            raise ValueError("Model mismatch between the selection file and corruption evaluation request.")
        checkpoint_source = "validation_selection"
        selection_split = selection.get("selection_split", selection.get("split"))
        selection_metric = selection.get("selection_metric")

    manifest = (
        Path(args.manifest).resolve()
        if args.manifest
        else default_manifest_path(args.protocol, args.train_seed).resolve()
    )
    if checkpoint_source == "validation_selection":
        recorded_manifest = selection.get("manifest")
        if recorded_manifest is not None and Path(recorded_manifest).expanduser().resolve() != manifest:
            raise ValueError(
                "Manifest mismatch between the selection file and the corruption evaluation request."
            )

    dataset = KSLManifestDataset(
        manifest_path=str(manifest),
        split="test",
        vocab_scope="manifest_train",
        face_mode="full",
        return_observed_mask=True,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cslr_collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_kds_model(args.model, dataset.vocab_size).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device))
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.corruption_seed)
    refs: list[str] = []
    hyps: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{args.model}/{args.protocol}/{args.corruption}/{args.severity:g}"):
            skeleton = batch["skeleton"].to(device)
            actual_frames = batch["actual_frames"].to(device)
            observed_mask = batch["observed_mask"].to(device)
            skeleton = apply_corruption(
                skeleton,
                actual_frames,
                observed_mask,
                args.corruption,
                args.severity,
                generator,
            )
            if args.model == "kds_m":
                skeleton = mouth_only_mask(skeleton)
            logits = model(skeleton, actual_frames=actual_frames)
            logit_lens = actual_frames
            hyps.extend(decode_ctc(logits, logit_lens, dataset.idx2gloss))
            refs.extend(full_reference_from_label(path, dataset.gloss2idx) for path in batch["label_paths"])

    full_metrics = metrics(refs, hyps)
    edit_summary = corpus_edit_counts(refs, hyps)
    result = {
        "model": args.model,
        "protocol": args.protocol,
        "train_seed": args.train_seed,
        "corruption": args.corruption,
        "severity": args.severity,
        "corruption_seed": args.corruption_seed,
        "run_name": run_name,
        "manifest": repo_relative(manifest),
        "checkpoint": "not_redistributed",
        "checkpoint_id": f"{args.model}/{args.protocol}/seed{args.train_seed}/{Path(checkpoint).stem}",
        "checkpoint_source": checkpoint_source,
        "selection_split": selection_split,
        "selection_metric": selection_metric,
        "selection_file": repo_relative(selection_file),
        "samples": len(dataset),
        "num_samples": len(dataset),
        "vocab_size": dataset.vocab_size,
        "oov_tokens": dataset.oov_tokens,
        "total_tokens": dataset.total_tokens,
        "oov_rate": dataset.oov_rate,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        **edit_summary,
        **full_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
