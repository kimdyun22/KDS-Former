#!/usr/bin/env python3
"""Evaluate KSL checkpoints on an arbitrary manifest split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_MODULE_DIR = REPO_ROOT / "datasets"
for item in [str(DATASET_MODULE_DIR), str(REPO_ROOT)]:
    if item not in sys.path:
        sys.path.insert(0, item)

from dataset_ksl_manifest import KSLManifestDataset, cslr_collate_fn, parse_label_file  # noqa: E402
from models.kds_former import KDSFormer  # noqa: E402


def repo_relative(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(candidate)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    dist = np.zeros((len(hyp) + 1, len(ref) + 1), dtype=np.int32)
    for i in range(len(hyp) + 1):
        dist[i, 0] = i
    for j in range(len(ref) + 1):
        dist[0, j] = j
    for i in range(1, len(hyp) + 1):
        for j in range(1, len(ref) + 1):
            if hyp[i - 1] == ref[j - 1]:
                dist[i, j] = dist[i - 1, j - 1]
            else:
                dist[i, j] = 1 + min(dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1])
    return int(dist[len(hyp), len(ref)])


def metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Compute corpus-level WER over whitespace-tokenised gloss sequences."""
    total_dist = 0
    total_ref = 0
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        total_dist += edit_distance(ref_tokens, hyp_tokens)
        total_ref += len(ref_tokens)
    wer = total_dist / total_ref if total_ref else 0.0

    return {"WER": wer}


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


def load_state_dict(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if state and next(iter(state)).startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-scope", choices=["manifest_train"], default="manifest_train",
                        help="Vocabulary is built from the manifest train split only (the primary-protocol policy).")
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KSLManifestDataset(
        manifest_path=args.manifest,
        split=args.split,
        vocab_scope=args.vocab_scope,
        vocab_path=args.vocab_path,
        downsample=args.downsample,
        add_motion=args.add_motion,
        part_hand_norm=args.part_hand_norm,
        face_mode=args.face_mode,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cslr_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    input_channels = 6 if args.add_motion else 2
    model = KDSFormer(
        num_classes=dataset.vocab_size,
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
    model.load_state_dict(load_state_dict(args.checkpoint, device))
    model.eval()

    refs: list[str] = []
    hyps: list[str] = []
    inv_vocab = dataset.idx2gloss

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {args.split}"):
            skeleton = batch["skeleton"].to(device)
            actual_frames = batch["actual_frames"].to(device)
            logits = model(skeleton, actual_frames=actual_frames)
            hyps.extend(decode_ctc(logits, actual_frames, inv_vocab))
            refs.extend(full_reference_from_label(path, dataset.gloss2idx) for path in batch["label_paths"])

    full_metrics = metrics(refs, hyps)

    result = {
        "manifest": repo_relative(args.manifest),
        "split": args.split,
        "checkpoint": "not_redistributed",
        "checkpoint_id": Path(args.checkpoint).stem,
        "samples": len(dataset),
        "vocab_size": dataset.vocab_size,
        "oov_tokens": dataset.oov_tokens,
        "total_tokens": dataset.total_tokens,
        "oov_rate": dataset.oov_rate,
        "downsample": args.downsample,
        "add_motion": args.add_motion,
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
        **full_metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
