#!/usr/bin/env python3
"""Build primary KSL split manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AIHUB_KSL_ROOT = Path(os.environ.get("AIHUB_KSL_ROOT", "data/aihub_sign_language"))
TRAIN_ROOT = AIHUB_KSL_ROOT / "수어 영상" / "1.Training" / "morpheme" / "morpheme"
VALID_ROOT = AIHUB_KSL_ROOT / "수어 영상" / "2.Validation" / "morpheme_SEN"
PUBLIC_PROTOCOL_NAMES = {
    "k1": "seen_signer_unseen_sentence",
    "k2": "unseen_signer_unseen_sentence",
}

PATH_RE = re.compile(r"NIA_SL_(SEN\d+)_REAL(\d{2})_([LRUDF])")


def to_manifest_path(path: Path) -> str:
    return str(path.relative_to(AIHUB_KSL_ROOT))


def host_path(path: str) -> str:
    raw = Path(path)
    if raw.is_absolute():
        return str(raw).replace("/data/수어 영상", str(AIHUB_KSL_ROOT / "수어 영상"))
    return str(AIHUB_KSL_ROOT / raw)


def parse_path(path: Path | str) -> dict[str, str]:
    match = PATH_RE.search(str(path))
    if not match:
        raise ValueError(f"Cannot parse SEN/signer/view from {path}")
    return {"sen": match.group(1), "signer": match.group(2), "view": match.group(3)}


def parse_label_file(path: str) -> tuple[str, list[str]]:
    real_path = host_path(path)
    with open(real_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    name = data.get("metaData", {}).get("name", "")
    match = PATH_RE.search(name)
    video_id = f"NIA_SL_{match.group(1)}_REAL{match.group(2)}_{match.group(3)}" if match else Path(name).stem
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
        glosses.update(seq)
    gloss2idx = {gloss: idx + 1 for idx, gloss in enumerate(sorted(glosses))}
    idx2gloss = {idx: gloss for gloss, idx in gloss2idx.items()}
    return gloss2idx, idx2gloss


def write_vocab(path: Path, gloss2idx: dict[str, int], idx2gloss: dict[int, str]) -> None:
    payload = {
        "blank_id": 0,
        "vocab_size_with_blank": len(gloss2idx) + 1,
        "gloss2idx": gloss2idx,
        "idx2gloss": {str(idx): gloss for idx, gloss in idx2gloss.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_front(root: Path) -> list[str]:
    paths = []
    for path in root.rglob("*.json"):
        meta = parse_path(path)
        if meta["view"] == "F":
            paths.append(to_manifest_path(path))
    return sorted(paths)


def group_by_sentence(paths: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        grouped[parse_path(path)["sen"]].append(path)
    return {sen: sorted(items) for sen, items in grouped.items()}


def split_sentence_ids(sentence_ids: list[str], seed: int, ratios: tuple[float, float, float]) -> tuple[set[str], set[str], set[str]]:
    ids = list(sentence_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    return train_ids, val_ids, test_ids


def paths_for_ids(grouped: dict[str, list[str]], ids: set[str]) -> list[str]:
    out: list[str] = []
    for sen in sorted(ids):
        out.extend(grouped[sen])
    return sorted(out)


def sentence_set(paths: list[str]) -> set[str]:
    return {parse_path(path)["sen"] for path in paths}


def signer_set(paths: list[str]) -> set[str]:
    return {parse_path(path)["signer"] for path in paths}


def gloss_stats(paths: list[str], train_vocab: set[str] | None = None) -> dict[str, Any]:
    token_count = 0
    oov_count = 0
    gloss_counter: Counter[str] = Counter()
    for path in paths:
        _, glosses = parse_label_file(path)
        token_count += len(glosses)
        gloss_counter.update(glosses)
        if train_vocab is not None:
            oov_count += sum(1 for gloss in glosses if gloss not in train_vocab)
    return {
        "samples": len(paths),
        "sentences": len(sentence_set(paths)),
        "signers": sorted(signer_set(paths)),
        "tokens": token_count,
        "unique_glosses": len(gloss_counter),
        "oov_tokens": oov_count,
        "oov_rate": oov_count / token_count if token_count else 0.0,
    }


def save_manifest(
    protocol: str,
    seed: int,
    split_seed: int,
    manifest: dict[str, list[str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    public_name = PUBLIC_PROTOCOL_NAMES[protocol]
    manifest_path = output_dir / f"{public_name}_seed{seed}.json"
    vocab_path = output_dir / f"{public_name}_seed{seed}.train_vocab.json"
    summary_path = output_dir / f"{public_name}_seed{seed}.summary.json"

    gloss2idx, idx2gloss = build_vocab(manifest["train"])
    write_vocab(vocab_path, gloss2idx, idx2gloss)
    train_vocab = set(gloss2idx)
    summary = {
        "protocol": protocol,
        "seed": seed,
        "run_seed": seed,
        "model_seed": seed,
        "split_seed": split_seed,
        "manifest": str(manifest_path),
        "train_vocab": str(vocab_path),
        "vocab_size_without_blank": len(gloss2idx),
        "vocab_size_with_blank": len(gloss2idx) + 1,
        "splits": {split: gloss_stats(paths, train_vocab) for split, paths in manifest.items()},
        "integrity": {
            "train_val_sentence_overlap": len(sentence_set(manifest["train"]) & sentence_set(manifest["val"])),
            "train_test_sentence_overlap": len(sentence_set(manifest["train"]) & sentence_set(manifest["test"])),
            "val_test_sentence_overlap": len(sentence_set(manifest["val"]) & sentence_set(manifest["test"])),
            "train_test_signer_overlap": len(signer_set(manifest["train"]) & signer_set(manifest["test"])),
        },
    }

    if protocol == "k1":
        assert summary["integrity"]["train_test_sentence_overlap"] == 0
    if protocol == "k2":
        assert summary["integrity"]["train_test_sentence_overlap"] == 0
        assert summary["integrity"]["train_test_signer_overlap"] == 0

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build(seed: int, output_dir: Path) -> None:
    train_front = collect_front(TRAIN_ROOT)
    valid_front = collect_front(VALID_ROOT)
    train_by_sen = group_by_sentence(train_front)
    valid_by_sen = group_by_sentence(valid_front)

    k1_train_ids, k1_val_ids, k1_test_ids = split_sentence_ids(sorted(train_by_sen), seed, (0.825, 0.09, 0.085))
    save_manifest(
        "k1",
        seed,
        seed,
        {
            "train": paths_for_ids(train_by_sen, k1_train_ids),
            "val": paths_for_ids(train_by_sen, k1_val_ids),
            "test": paths_for_ids(train_by_sen, k1_test_ids),
        },
        output_dir,
    )

    shared_ids = sorted(set(train_by_sen) & set(valid_by_sen))
    k2_train_ids, k2_val_ids, k2_test_ids = split_sentence_ids(shared_ids, seed + 1000, (0.825, 0.09, 0.085))
    save_manifest(
        "k2",
        seed,
        seed + 1000,
        {
            "train": paths_for_ids(train_by_sen, k2_train_ids),
            "val": paths_for_ids(valid_by_sen, k2_val_ids),
            "test": paths_for_ids(valid_by_sen, k2_test_ids),
        },
        output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 2024])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generated_manifests"),
        help="Directory for generated manifests, vocabularies, and summaries.",
    )
    args = parser.parse_args()
    for seed in args.seeds:
        build(seed, args.output_dir)


if __name__ == "__main__":
    main()
