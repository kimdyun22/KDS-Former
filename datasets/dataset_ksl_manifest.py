#!/usr/bin/env python3
"""Manifest-driven KSL dataset for paper experiments."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.keypoint_parser import load_keypoints_2d_for_video  # noqa: E402


AIHUB_KSL_ROOT = Path(os.environ.get("AIHUB_KSL_ROOT", "data/aihub_sign_language"))

VIDEO_ID_RE = re.compile(r"(NIA_SL_SEN\d+_REAL\d{2}_[LRUDF])")


def host_path(path: str) -> str:
    raw = Path(path)
    if raw.is_absolute():
        return str(raw).replace("/data/수어 영상", str(AIHUB_KSL_ROOT / "수어 영상"))
    return str(AIHUB_KSL_ROOT / raw)


def parse_label_file(path: str) -> tuple[str, list[str]]:
    real_path = host_path(path)
    with open(real_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    name = data.get("metaData", {}).get("name", "")
    raw_id = os.path.splitext(name)[0]
    match = VIDEO_ID_RE.search(raw_id)
    video_id = match.group(1) if match else raw_id

    glosses: list[str] = []
    for seg in data.get("data", []):
        for attr in seg.get("attributes", []):
            gloss = attr.get("name")
            if gloss:
                glosses.append(gloss.strip())
    return video_id, glosses


def build_vocab(label_paths: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    glosses: set[str] = set()
    for path in label_paths:
        _, seq = parse_label_file(path)
        glosses.update(g for g in seq if g)
    gloss2idx = {gloss: idx + 1 for idx, gloss in enumerate(sorted(glosses))}
    idx2gloss = {idx: gloss for gloss, idx in gloss2idx.items()}
    return gloss2idx, idx2gloss


def load_manifest_split(manifest_path: str, split: str) -> list[str]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if split not in manifest:
        raise KeyError(f"split {split!r} not found in {manifest_path}")
    paths = manifest[split]
    if not isinstance(paths, list):
        raise TypeError(f"split {split!r} must be a list of paths")
    return sorted(paths)


def load_manifest(manifest_path: str) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"manifest must be a JSON object: {manifest_path}")
    return manifest


def write_vocab(path: str | Path, gloss2idx: dict[str, int], idx2gloss: dict[int, str]) -> None:
    payload = {
        "blank_id": 0,
        "vocab_size_with_blank": len(gloss2idx) + 1,
        "gloss2idx": gloss2idx,
        "idx2gloss": {str(idx): gloss for idx, gloss in idx2gloss.items()},
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    gloss2idx = {str(gloss): int(idx) for gloss, idx in payload["gloss2idx"].items()}
    idx2gloss = {int(idx): str(gloss) for idx, gloss in payload["idx2gloss"].items()}
    return gloss2idx, idx2gloss


def signer_normalize(skeleton: np.ndarray) -> np.ndarray:
    neck_idx, left_shoulder_idx, right_shoulder_idx = 1, 2, 5
    skeleton = skeleton.copy()
    for t in range(skeleton.shape[0]):
        neck = skeleton[t, neck_idx, :].copy()
        if neck[0] == 0 and neck[1] == 0:
            continue
        left = skeleton[t, left_shoulder_idx, :]
        right = skeleton[t, right_shoulder_idx, :]
        skeleton[t, :, 0] -= neck[0]
        skeleton[t, :, 1] -= neck[1]
        shoulder_width = np.linalg.norm(left - right)
        if shoulder_width > 1e-6:
            skeleton[t] /= shoulder_width
    return skeleton


def hand_part_anchor_normalize(skeleton: np.ndarray) -> np.ndarray:
    skeleton = skeleton.copy()
    left_wrist = skeleton[:, 95:96, :].copy()
    right_wrist = skeleton[:, 116:117, :].copy()
    left_valid = np.linalg.norm(left_wrist, axis=-1, keepdims=True) > 1e-6
    right_valid = np.linalg.norm(right_wrist, axis=-1, keepdims=True) > 1e-6
    skeleton[:, 95:116, :] = np.where(left_valid, skeleton[:, 95:116, :] - left_wrist, skeleton[:, 95:116, :])
    skeleton[:, 116:137, :] = np.where(right_valid, skeleton[:, 116:137, :] - right_wrist, skeleton[:, 116:137, :])
    return skeleton


def mask_face_to_mouth(skeleton: np.ndarray) -> np.ndarray:
    skeleton = skeleton.copy()
    keep = np.zeros((137,), dtype=bool)
    keep[:25] = True
    keep[25 + 48 : 25 + 67] = True
    keep[95:137] = True
    skeleton[:, ~keep, :] = 0.0
    return skeleton


def mask_face(skeleton: np.ndarray) -> np.ndarray:
    skeleton = skeleton.copy()
    skeleton[:, 25:95, :] = 0.0
    return skeleton


class KSLManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        max_frames: int = 300,
        vocab_scope: str = "manifest_train",
        vocab_path: str | None = None,
        use_signer_norm: bool = True,
        downsample: int = 1,
        add_motion: bool = False,
        temporal_drop_prob: float = 0.0,
        jitter_sigma: float = 0.0,
        augment: bool = False,
        part_hand_norm: bool = False,
        face_mode: str = "full",
        return_observed_mask: bool = False,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.split = split
        self.max_frames = max_frames
        self.use_signer_norm = use_signer_norm
        if downsample < 1:
            raise ValueError("downsample must be >= 1")
        self.downsample = downsample
        self.add_motion = add_motion
        self.temporal_drop_prob = temporal_drop_prob
        self.jitter_sigma = jitter_sigma
        self.augment = augment
        self.part_hand_norm = part_hand_norm
        self.return_observed_mask = return_observed_mask
        if face_mode == "full_face":
            face_mode = "full"
        if face_mode not in {"full", "full_face", "mouth_only", "no_face"}:
            raise ValueError("face_mode must be full, full_face, mouth_only, or no_face")
        self.face_mode = face_mode
        self.label_paths = load_manifest_split(manifest_path, split)
        if limit is not None:
            self.label_paths = self.label_paths[:limit]

        if vocab_path is not None:
            self.gloss2idx, self.idx2gloss = load_vocab(vocab_path)
        elif vocab_scope == "manifest_train":
            vocab_paths = load_manifest_split(manifest_path, "train")
            self.gloss2idx, self.idx2gloss = build_vocab(vocab_paths)
        else:
            raise ValueError("vocab_scope must be manifest_train")
        self.blank_id = 0
        self.vocab_size = len(self.gloss2idx) + 1
        self.oov_tokens = 0
        self.total_tokens = 0
        for path in self.label_paths:
            _, glosses = parse_label_file(path)
            self.total_tokens += len(glosses)
            self.oov_tokens += sum(1 for gloss in glosses if gloss not in self.gloss2idx)
        self.oov_rate = self.oov_tokens / self.total_tokens if self.total_tokens else 0.0

    def __len__(self) -> int:
        return len(self.label_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        label_path = self.label_paths[index]
        video_id, glosses = parse_label_file(label_path)
        skeleton = load_keypoints_2d_for_video(video_id)
        observed_mask = np.any(skeleton != 0.0, axis=-1)
        if self.use_signer_norm:
            skeleton = signer_normalize(skeleton)
        if self.part_hand_norm:
            skeleton = hand_part_anchor_normalize(skeleton)
        if self.face_mode == "mouth_only":
            skeleton = mask_face_to_mouth(skeleton)
        elif self.face_mode == "no_face":
            skeleton = mask_face(skeleton)

        if self.downsample > 1:
            skeleton = skeleton[:: self.downsample]
            observed_mask = observed_mask[:: self.downsample]

        frame_len = min(int(skeleton.shape[0]), self.max_frames)
        if skeleton.shape[0] >= self.max_frames:
            skeleton = skeleton[: self.max_frames]
            observed_mask = observed_mask[: self.max_frames]
        else:
            pad = np.zeros((self.max_frames - skeleton.shape[0], skeleton.shape[1], 2), dtype=np.float32)
            skeleton = np.concatenate([skeleton, pad], axis=0)
            mask_pad = np.zeros((self.max_frames - observed_mask.shape[0], observed_mask.shape[1]), dtype=bool)
            observed_mask = np.concatenate([observed_mask, mask_pad], axis=0)

        if self.augment and frame_len > 0:
            valid = slice(0, frame_len)
            if self.jitter_sigma > 0:
                noise = np.random.normal(0.0, self.jitter_sigma, size=skeleton[valid].shape).astype(np.float32)
                skeleton[valid] = skeleton[valid] + noise
            if self.temporal_drop_prob > 0:
                drop = np.random.random(size=(frame_len, 1, 1)) < self.temporal_drop_prob
                skeleton[valid] = np.where(drop, 0.0, skeleton[valid])

        if self.add_motion:
            forward = np.zeros_like(skeleton)
            backward = np.zeros_like(skeleton)
            if frame_len > 1:
                forward[: frame_len - 1] = skeleton[1:frame_len] - skeleton[: frame_len - 1]
                backward[1:frame_len] = skeleton[1:frame_len] - skeleton[: frame_len - 1]
            skeleton = np.concatenate([skeleton, forward, backward], axis=-1)

        gloss_ids = [self.gloss2idx[g] for g in glosses if g in self.gloss2idx]
        if not gloss_ids:
            # Preserve the original blank-target handling for all-OOV
            # validation/test samples. Full-reference WER is computed from
            # the original labels and scores unavailable glosses as <OOV>.
            gloss_ids = [self.blank_id]

        item = {
            "video_id": video_id,
            "label_path": label_path,
            "skeleton": torch.from_numpy(skeleton.astype(np.float32)),
            "actual_frames": torch.tensor(frame_len, dtype=torch.long),
            "gloss_ids": torch.tensor(gloss_ids, dtype=torch.long),
            "gloss_len": len(gloss_ids),
        }
        if self.return_observed_mask:
            item["observed_mask"] = torch.from_numpy(observed_mask.astype(bool))
        return item


def cslr_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "video_ids": [item["video_id"] for item in batch],
        "label_paths": [item["label_path"] for item in batch],
        "skeleton": torch.stack([item["skeleton"] for item in batch], dim=0),
        "actual_frames": torch.stack([item["actual_frames"] for item in batch], dim=0),
        "targets": torch.cat([item["gloss_ids"] for item in batch], dim=0),
        "target_lens": torch.tensor([item["gloss_len"] for item in batch], dtype=torch.long),
    }
    if "observed_mask" in batch[0]:
        collated["observed_mask"] = torch.stack([item["observed_mask"] for item in batch], dim=0)
    return collated
