# KDS-Former

This repository provides the core implementation of KDS-Former for continuous
Korean Sign Language recognition.

## Overview

- `models/kds_former.py`: KDS-Former-S and KDS-Former-M architecture.
- `datasets/`: KSL-Guide label parsing, canonical 137-keypoint loading, and
  manifest dataset code.
- `training/train_ksl_manifest.py`: KDS-Former training entry point.
- `scripts/select_checkpoint_by_val_wer.py`: validation-WER checkpoint
  selection only.
- `scripts/eval_test_from_valwer_selection.py`: test evaluation after
  checkpoint selection.
- `evaluation/eval_ksl_manifest.py`: split evaluation with WER and
  out-of-vocabulary token statistics.
- `manifests/`: exact primary KSL-Guide manifests for seeds 42, 43, and 2024.
- `configs/`: human-readable records of final KDS-Former-S/M hyperparameter
  settings. The CLI entry points do not load these YAML files automatically.

## Environment

The experiments reported in the paper used PyTorch 2.5.1 with CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate kds-former
```

The Dockerfile starts from a PyTorch CUDA runtime image and installs only the
non-Torch Python dependencies from `requirements.txt`.

```bash
docker build -t kds-former .
```

## Verification

After setting up the environment, run the test suite to check keypoint
parsing, face-mode masking, selection-policy checks, and manifest integrity:

```bash
pytest -q
```

## Dataset Preparation

Raw AIHub KSL-Guide files are not included. After obtaining the dataset through
the official provider, set:

```bash
export AIHUB_KSL_ROOT=/path/to/aihub_ksl_guide
```

The loader expects the AIHub directory layout beneath this root, including:

```text
수어 영상/1.Training/morpheme/morpheme/
수어 영상/1.Training/keypoint/
수어 영상/2.Validation/morpheme_SEN/
수어 영상/2.Validation/[라벨]09_real_sen_keypoint/keypoint/
```

Each frame is decoded into this fixed keypoint order:

```text
Body25 + Face70 + LeftHand21 + RightHand21 = 137 keypoints
```

KDS-Former-M uses `face_mode=mouth_only`: it keeps body joints, face-local
landmarks 48-66, and both hands, while zeroing the other 51 face landmarks.

For KDS-Former-S/M with `nhead=8`, each branch encoder uses four attention
heads (`nhead // 2`) and the fusion encoder uses eight attention heads.

The six public manifests are:

```text
manifests/seen_signer_unseen_sentence_seed42.json
manifests/seen_signer_unseen_sentence_seed43.json
manifests/seen_signer_unseen_sentence_seed2024.json
manifests/unseen_signer_unseen_sentence_seed42.json
manifests/unseen_signer_unseen_sentence_seed43.json
manifests/unseen_signer_unseen_sentence_seed2024.json
```

## Training

KDS-Former-S:

```bash
python training/train_ksl_manifest.py \
  --manifest manifests/seen_signer_unseen_sentence_seed42.json \
  --protocol k1 \
  --seed 42 \
  --epochs 60 \
  --batch-size 12 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --max-frames 300 \
  --d-hand 128 \
  --d-body 128 \
  --d-fusion 256 \
  --nhead 8 \
  --branch-layers 2 \
  --fusion-layers 2 \
  --dim-feedforward 512 \
  --run-suffix _kds_former_s \
  --no-amp
```

This writes checkpoints under `checkpoints/k1_seed42_kds_former_s/`.

KDS-Former-M mouth-only:

```bash
python training/train_ksl_manifest.py \
  --manifest manifests/seen_signer_unseen_sentence_seed42.json \
  --protocol k1 \
  --seed 42 \
  --epochs 60 \
  --batch-size 12 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --max-frames 300 \
  --d-hand 256 \
  --d-body 256 \
  --d-fusion 512 \
  --nhead 8 \
  --branch-layers 3 \
  --fusion-layers 3 \
  --dim-feedforward 1024 \
  --face-mode mouth_only \
  --warmup-epochs 5 \
  --run-suffix _kds_former_m_mouth_only \
  --no-amp
```

This writes checkpoints under
`checkpoints/k1_seed42_kds_former_m_mouth_only/`.

## Validation-Based Checkpoint Selection

Checkpoint selection must use validation WER only. The selection script rejects
test-split selection.

KDS-Former-S:

```bash
python scripts/select_checkpoint_by_val_wer.py \
  --run-name k1_seed42_kds_former_s \
  --manifest manifests/seen_signer_unseen_sentence_seed42.json \
  --checkpoint-glob 'checkpoints/k1_seed42_kds_former_s/*.pt' \
  --output-dir outputs/val_wer_selection \
  --model kds_s \
  --protocol k1 \
  --seed 42
```

KDS-Former-M mouth-only:

```bash
python scripts/select_checkpoint_by_val_wer.py \
  --run-name k1_seed42_kds_former_m_mouth_only \
  --manifest manifests/seen_signer_unseen_sentence_seed42.json \
  --checkpoint-glob 'checkpoints/k1_seed42_kds_former_m_mouth_only/*.pt' \
  --output-dir outputs/val_wer_selection \
  --model kds_m \
  --protocol k1 \
  --seed 42 \
  --d-hand 256 \
  --d-body 256 \
  --d-fusion 512 \
  --branch-layers 3 \
  --fusion-layers 3 \
  --dim-feedforward 1024 \
  --face-mode mouth_only
```

## Evaluation

Test metrics are computed only after a validation-WER-selected checkpoint is
fixed.

```bash
python scripts/eval_test_from_valwer_selection.py \
  --selection outputs/val_wer_selection/k1_seed42_kds_former_s/k1_seed42_kds_former_s_val_wer_selection.json
```

The evaluator reports WER, with OOV glosses counted as errors, and OOV token
counts/rates. WER is the primary evaluation metric used throughout the paper.

## Citation

Please cite the associated KDS-Former manuscript if you use this code or the
released split manifests. Metadata is provided in `CITATION.cff`.

## License

This repository's original code is released under the MIT License.
