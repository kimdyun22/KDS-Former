import os
import re
import json
from pathlib import Path

import numpy as np


AIHUB_KSL_ROOT = Path(os.environ.get("AIHUB_KSL_ROOT", "data/aihub_sign_language"))

TRAIN_KP_ROOTS = [
    str(AIHUB_KSL_ROOT / "수어 영상" / "1.Training" / "keypoint"),
    "/data/수어 영상/1.Training/keypoint",
]

VAL_KP_ROOTS = [
    str(AIHUB_KSL_ROOT / "수어 영상" / "2.Validation" / "[라벨]09_real_sen_keypoint" / "keypoint"),
    str(AIHUB_KSL_ROOT / "수어 영상" / "2.Validation" / "keypoint"),
    "/data/수어 영상/2.Validation/[라벨]09_real_sen_keypoint/keypoint",
    "/data/수어 영상/2.Validation/keypoint",
]

NUM_JOINTS = 137

def _safe_reshape_xy(flat_list, expected: int | None = None) -> np.ndarray:
    """
    flat_list: [x1, y1, c1, x2, y2, c2, ...]
    return: (N, 2) float32
    """
    arr = np.asarray(flat_list, dtype=np.float32)
    if arr.size == 0:
        if expected is None:
            return np.zeros((0, 2), dtype=np.float32)
        return np.zeros((expected, 2), dtype=np.float32)

    arr = arr.reshape(-1, 3)[:, :2]  # (N, 2)

    if expected is not None:
        n = arr.shape[0]
        if n > expected:
            arr = arr[:expected]
        elif n < expected:
            pad = np.zeros((expected - n, 2), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

    return arr


def _decode_single_frame(json_path: str) -> np.ndarray:
    """Decode one OpenPose keypoint JSON frame into a (NUM_JOINTS, 2) array."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_people = data.get("people", None)
    people = {}
    if isinstance(raw_people, list):
        if len(raw_people) > 0:
            people = raw_people[0]
    elif isinstance(raw_people, dict):
        people = raw_people
    # OpenPose format with fixed canonical slots.
    pose = _safe_reshape_xy(people.get("pose_keypoints_2d", []), expected=25)
    face = _safe_reshape_xy(people.get("face_keypoints_2d", []), expected=70)
    lh = _safe_reshape_xy(people.get("hand_left_keypoints_2d", []), expected=21)
    rh = _safe_reshape_xy(people.get("hand_right_keypoints_2d", []), expected=21)

    # Canonical order: body and face first, then both hands.
    # pose(25) + face(70) + lh(21) + rh(21) = 137
    kpts = np.concatenate([pose, face, lh, rh], axis=0)

    if kpts.shape != (NUM_JOINTS, 2):
        raise ValueError(f"canonical keypoint shape mismatch: {kpts.shape}")

    return kpts.astype(np.float32)  # (J, 2)


def _find_video_dir(video_id: str) -> str:
    """Find the keypoint directory for an AIHub KSL-Guide video ID."""
    m = re.search(r"REAL(\d+)_", video_id)
    if not m:
        raise ValueError(f"could not parse REAL signer ID from video_id: {video_id}")
    real_num = m.group(1)  # "17" or "18"

    cand_dirs: list[str] = []

    for root in TRAIN_KP_ROOTS:
        cand_dirs.append(os.path.join(root, real_num, video_id))
        cand_dirs.append(os.path.join(root, "keypoint", real_num, video_id))

    for root in VAL_KP_ROOTS:
        cand_dirs.append(os.path.join(root, real_num, video_id))
        cand_dirs.append(os.path.join(root, "keypoint", real_num, video_id))

    for d in cand_dirs:
        if os.path.isdir(d):
            return d

    msg = "video_dir not found for {vid}. Tried:\n  ".format(vid=video_id)
    msg += "\n  ".join(cand_dirs)
    raise FileNotFoundError(msg)


# -------------------------------------------------------
#  Public API
# -------------------------------------------------------

def load_keypoints_2d_for_video(video_id: str) -> np.ndarray:
    """Load all keypoint frames for a video as a (T, NUM_JOINTS, 2) array."""
    video_dir = _find_video_dir(video_id)

    all_files = [f for f in os.listdir(video_dir) if f.endswith("_keypoints.json")]
    if not all_files:
        raise FileNotFoundError(f"No keypoint jsons in {video_dir}")

    def _frame_index(name: str) -> int:
        m = re.search(r"_([0-9]{12})_keypoints\.json$", name)
        if m:
            return int(m.group(1))
        return 0

    all_files.sort(key=_frame_index)

    frames: list[np.ndarray] = []
    for fname in all_files:
        fpath = os.path.join(video_dir, fname)
        try:
            kpts = _decode_single_frame(fpath)  # (J, 2)
            frames.append(kpts)
        except json.JSONDecodeError:
            print(f"[WARN] JSONDecodeError in keypoint file, skip: {fpath}")
        except Exception as e:
            print(f"[WARN] Error in keypoint file {fpath}: {e}")

    if not frames:
        raise RuntimeError(f"No valid keypoint frames for video_id={video_id}")

    skel = np.stack(frames, axis=0)  # (T, J, 2)
    return skel.astype(np.float32)
