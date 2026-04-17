#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align each annotator TextGrid to SSL frame times for multiple pilot datasets.

This is the per-annotator alignment step:
- one output per dataset
- one output per annotator

If you want consensus-only training data, this file is not the final step.
Use the agreement table plus prepare_training_data_from_agreement.py after this.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from textgrid_utils import active_tiers_at_time, collect_boundaries, read_textgrid_intervals


# ============================================================
# USER SETTINGS
# ============================================================
DATASETS = {
    "female_sp1": {
        "embedding_path": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings_wavlm_large/female_sp1/frame_level/sp1_female.npz"),
        "annotators": {
            "AP": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Female/AP Speaker001Session1IntTaskStudioQual_100425 (3).TextGrid"),
            "TB": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Female/TB-Speaker001Session1IntTaskStudioQual_100425 (1).TextGrid"),
            "XD": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Female/XD Speaker001Session1IntTaskStudioQual_100425 (1).TextGrid"),
        },
    },
    "male_sp1": {
        "embedding_path": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings_wavlm_large/male_sp1/frame_level/sp1_male.npz"),
        "annotators": {
            "AP": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Male/AP 001-1-060310 (1).TextGrid"),
            "TB": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Male/TB-001-1-060310 (3).TextGrid"),
            "XD": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Male/XD 001-1-060310 (1).TextGrid"),
        },
    },
}

TIER_NAME_MAP = {
    "BRT": "Breathy",
    "CRK": "Creaky",
    "WHS": "Whispery",
    "Breathy": "Breathy",
    "Creaky": "Creaky",
    "Whispery": "Whispery",
}

TARGET_TIERS = ["Breathy", "Creaky", "Whispery"]
POSITIVE_VALUES = {"1", "2"}
IGNORE_BOUNDARY_MARGIN_SEC = 0.02
DROP_MULTI_LABEL_FRAMES = True

OUTPUT_ROOT = Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/aligned_by_annotator")


def canonical_active_tiers(textgrid, time_sec):
    active = active_tiers_at_time(
        textgrid,
        list(TIER_NAME_MAP.keys()),
        time_sec,
        positive_values=POSITIVE_VALUES,
    )
    canonical = set()
    for label in active:
        mapped = TIER_NAME_MAP.get(label)
        if mapped is not None:
            canonical.add(mapped)
    return canonical


def encode_state(active_labels):
    if not active_labels:
        return "none"
    return "+".join(sorted(label.lower() for label in active_labels))


def collect_mapped_boundaries(textgrid):
    return collect_boundaries(
        textgrid,
        list(TIER_NAME_MAP.keys()),
        positive_only=True,
        positive_values=POSITIVE_VALUES,
    )


def align_one_annotator(times_sec, textgrid_path: Path):
    tiers = read_textgrid_intervals(textgrid_path)
    boundaries = collect_mapped_boundaries(tiers)

    labels = []
    active_count = []
    boundary_mask = np.zeros(times_sec.shape[0], dtype=bool)

    for i, t in enumerate(times_sec):
        active = canonical_active_tiers(tiers, float(t))
        labels.append(encode_state(active))
        active_count.append(len(active))

        if IGNORE_BOUNDARY_MARGIN_SEC > 0:
            for boundary in boundaries:
                if abs(float(t) - boundary) <= IGNORE_BOUNDARY_MARGIN_SEC:
                    boundary_mask[i] = True
                    break

    labels = np.asarray(labels, dtype=object)
    active_count = np.asarray(active_count, dtype=np.int32)

    valid_mask = ~boundary_mask
    if DROP_MULTI_LABEL_FRAMES:
        valid_mask &= active_count <= 1

    return labels, valid_mask, boundary_mask, active_count


def process_dataset(dataset_name: str, cfg: dict):
    emb = np.load(cfg["embedding_path"], allow_pickle=True)
    times_sec = emb["times_sec"].astype(np.float32)

    out_dir = OUTPUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for annotator_name, textgrid_path in cfg["annotators"].items():
        labels, valid_mask, boundary_mask, active_count = align_one_annotator(times_sec, textgrid_path)

        npz_path = out_dir / f"{annotator_name}_aligned_labels.npz"
        csv_path = out_dir / f"{annotator_name}_aligned_labels.csv"
        json_path = out_dir / f"{annotator_name}_aligned_labels.json"

        np.savez_compressed(
            npz_path,
            times_sec=times_sec,
            labels=labels,
            valid_mask=valid_mask,
            boundary_mask=boundary_mask,
            active_count=active_count,
        )

        pd.DataFrame(
            {
                "time_sec": times_sec,
                "label": labels,
                "valid_mask": valid_mask,
                "boundary_mask": boundary_mask,
                "active_count": active_count,
            }
        ).to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "annotator": annotator_name,
                    "textgrid_path": str(textgrid_path),
                    "embedding_path": str(cfg["embedding_path"]),
                    "ignore_boundary_margin_sec": IGNORE_BOUNDARY_MARGIN_SEC,
                    "drop_multi_label_frames": DROP_MULTI_LABEL_FRAMES,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "annotator": annotator_name,
                "textgrid_path": str(textgrid_path),
                "embedding_path": str(cfg["embedding_path"]),
                "num_frames": int(len(times_sec)),
                "num_valid_frames": int(valid_mask.sum()),
                "num_boundary_drops": int(boundary_mask.sum()),
                "output_npz": str(npz_path),
                "output_csv": str(csv_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "alignment_summary.csv", index=False)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name, cfg in DATASETS.items():
        print("=" * 60)
        print(f"ALIGNING DATASET: {dataset_name}")
        print("=" * 60)
        process_dataset(dataset_name, cfg)

    print("=" * 60)
    print("PER-ANNOTATOR ALIGNMENT COMPLETE")
    print(f"Saved to: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
