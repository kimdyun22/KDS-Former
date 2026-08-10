from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_ksl_manifest import mask_face_to_mouth
from datasets.keypoint_parser import NUM_JOINTS, _decode_single_frame
from evaluation.eval_keypoint_corruption import (
    repo_relative,
    run_name_for,
    validate_selection as validate_corruption_selection,
    validate_severity,
)
from models.kds_former import KDSFormer
from scripts.eval_test_from_valwer_selection import same_path, validate_selection as validate_test_selection


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def test_keypoint_parser_preserves_canonical_slots(tmp_path: Path) -> None:
    payload = {
        "people": [
            {
                "pose_keypoints_2d": [1.0, 2.0, 0.9] * 25,
                "face_keypoints_2d": [3.0, 4.0, 0.8] * 68,
                "hand_left_keypoints_2d": [5.0, 6.0, 0.7] * 21,
                "hand_right_keypoints_2d": [7.0, 8.0, 0.6] * 21,
            }
        ]
    }
    path = tmp_path / "frame_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    frame = _decode_single_frame(str(path))

    assert frame.shape == (NUM_JOINTS, 2)
    assert frame[0].tolist() == [1.0, 2.0]
    assert frame[25].tolist() == [3.0, 4.0]
    assert frame[25 + 68].tolist() == [0.0, 0.0]
    assert frame[95].tolist() == [5.0, 6.0]
    assert frame[116].tolist() == [7.0, 8.0]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"people": []},
        {"people": [{}]},
        {"people": {"pose_keypoints_2d": [1.0, 2.0, 0.9] * 25}},
    ],
)
def test_keypoint_parser_handles_missing_fields(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "frame_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    frame = _decode_single_frame(str(path))

    assert frame.shape == (NUM_JOINTS, 2)
    if not payload or payload.get("people") in ([], [{}]):
        assert np.all(frame == 0.0)


def test_keypoint_parser_truncates_and_pads_each_part(tmp_path: Path) -> None:
    payload = {
        "people": [
            {
                "pose_keypoints_2d": [1.0, 2.0, 0.9] * 30,
                "face_keypoints_2d": [3.0, 4.0, 0.8] * 68,
                "hand_left_keypoints_2d": [5.0, 6.0, 0.7] * 19,
                "hand_right_keypoints_2d": [7.0, 8.0, 0.6] * 25,
            }
        ]
    }
    path = tmp_path / "frame_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    frame = _decode_single_frame(str(path))

    assert frame.shape == (NUM_JOINTS, 2)
    assert frame[24].tolist() == [1.0, 2.0]
    assert frame[25 + 67].tolist() == [3.0, 4.0]
    assert frame[25 + 68].tolist() == [0.0, 0.0]
    assert frame[95 + 18].tolist() == [5.0, 6.0]
    assert frame[95 + 19].tolist() == [0.0, 0.0]
    assert frame[136].tolist() == [7.0, 8.0]


def test_mouth_only_mask_keeps_expected_slots() -> None:
    skeleton = np.ones((2, 137, 2), dtype=np.float32)
    masked = mask_face_to_mouth(skeleton)

    keep = np.zeros((137,), dtype=bool)
    keep[:25] = True
    keep[25 + 48 : 25 + 67] = True
    keep[95:137] = True

    assert np.all(masked[:, keep, :] == 1.0)
    assert np.all(masked[:, ~keep, :] == 0.0)


@pytest.mark.parametrize(
    ("name", "kwargs", "expected_params"),
    [
        (
            "KDS-Former-S",
            {
                "num_classes": 436,
                "d_hand": 128,
                "d_body": 128,
                "d_fusion": 256,
                "nhead": 8,
                "branch_layers": 2,
                "fusion_layers": 2,
                "dim_feedforward": 512,
            },
            2_586_804,
        ),
        (
            "KDS-Former-M",
            {
                "num_classes": 436,
                "d_hand": 256,
                "d_body": 256,
                "d_fusion": 512,
                "nhead": 8,
                "branch_layers": 3,
                "fusion_layers": 3,
                "dim_feedforward": 1024,
            },
            14_754_740,
        ),
    ],
)
def test_kds_former_param_counts_and_forward(name: str, kwargs: dict, expected_params: int) -> None:
    model = KDSFormer(**kwargs)
    model.eval()
    assert count_params(model) == expected_params, name

    x = torch.randn(2, 8, 137, 2)
    actual_frames = torch.tensor([8, 5])
    with torch.no_grad():
        gloss_logits = model(x, actual_frames=actual_frames)
    assert gloss_logits.shape == (2, 8, 436)


def test_kds_former_requires_num_classes() -> None:
    with pytest.raises(TypeError):
        KDSFormer()
    with pytest.raises(ValueError):
        KDSFormer(num_classes=1)


def test_public_run_names_are_stable() -> None:
    assert run_name_for("kds_s", "k1", 42) == "k1_seed42_kds_former_s"
    assert run_name_for("kds_m", "k1", 42) == "k1_seed42_kds_former_m_mouth_only"


def test_rejects_test_selected_checkpoint() -> None:
    invalid = {
        "selection_split": "test",
        "selection_metric": "test WER",
        "best_checkpoint": "model.pt",
    }
    with pytest.raises(ValueError):
        validate_test_selection(invalid)
    with pytest.raises(ValueError):
        validate_corruption_selection(invalid)


def test_valid_selection_accepts_legacy_split_key() -> None:
    selection = {
        "split": "val",
        "selection_metric": "validation WER",
        "best_checkpoint": "model.pt",
    }
    assert validate_test_selection(selection) == "model.pt"
    assert validate_corruption_selection(selection) == "model.pt"


def test_manifest_path_comparison_detects_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("{}", encoding="utf-8")
    right.write_text("{}", encoding="utf-8")
    assert same_path(left, left)
    assert not same_path(left, right)


def test_corruption_severity_rules() -> None:
    validate_severity("dropout", 0.0)
    validate_severity("dropout", 1.0)
    validate_severity("gaussian", 0.0)
    with pytest.raises(ValueError):
        validate_severity("dropout", 1.5)
    with pytest.raises(ValueError):
        validate_severity("gaussian", -0.1)


def test_explicit_checkpoint_provenance_fields() -> None:
    selection_file = None
    metadata = {
        "checkpoint_source": "explicit_checkpoint",
        "selection_split": None,
        "selection_metric": None,
        "selection_file": repo_relative(selection_file),
    }
    assert metadata["checkpoint_source"] == "explicit_checkpoint"
    assert metadata["selection_split"] is None
    assert metadata["selection_metric"] is None
    assert metadata["selection_file"] is None


def test_default_vocab_scope_is_manifest_train() -> None:
    import inspect

    from datasets.dataset_ksl_manifest import KSLManifestDataset

    sig = inspect.signature(KSLManifestDataset.__init__)
    assert sig.parameters["vocab_scope"].default == "manifest_train"
    assert "vocab_manifest_split" not in sig.parameters


def _sentence_ids(paths: list[str]) -> set[str]:
    return {match.group(1) for path in paths if (match := re.search(r"(SEN\d+)", path))}


def _signer_ids(paths: list[str]) -> set[str]:
    return {match.group(1) for path in paths if (match := re.search(r"REAL(\d{2})", path))}


def test_primary_manifests_are_present_and_nonoverlapping() -> None:
    expected = [
        "seen_signer_unseen_sentence_seed42.json",
        "seen_signer_unseen_sentence_seed43.json",
        "seen_signer_unseen_sentence_seed2024.json",
        "unseen_signer_unseen_sentence_seed42.json",
        "unseen_signer_unseen_sentence_seed43.json",
        "unseen_signer_unseen_sentence_seed2024.json",
    ]
    manifest_dir = ROOT / "manifests"

    for name in expected:
        assert (manifest_dir / name).exists()
        data = json.loads((manifest_dir / name).read_text(encoding="utf-8"))
        assert set(data) == {"train", "val", "test"}
        train = set(data["train"])
        val = set(data["val"])
        test = set(data["test"])
        assert train
        assert val
        assert test
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert all(not Path(path).is_absolute() for split in data.values() for path in split)
        assert _sentence_ids(data["train"]).isdisjoint(_sentence_ids(data["test"]))
        if name.startswith("seen_signer"):
            assert _signer_ids(data["train"]) & _signer_ids(data["test"])
        else:
            assert _signer_ids(data["train"]).isdisjoint(_signer_ids(data["test"]))


def test_metrics_reports_wer_only() -> None:
    from evaluation.eval_ksl_manifest import metrics

    result = metrics(["A B"], ["A C"])

    assert set(result) == {"WER"}
    assert result["WER"] == 0.5
