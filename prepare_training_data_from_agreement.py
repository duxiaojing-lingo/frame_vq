#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare train-ready SSL frame data by combining:
  1) extracted frame-level embeddings
  2) inter-rater agreement tables

Important:
- Your current agreement CSV's `final_mask` is not a true agreement mask.
- This script derives a real consensus label from the annotator states.
- By default it keeps frames where at least 2 of 3 annotators agree.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================
DATASETS = {
    "female_sp1": {
        "agreement_csv": Path(
            "/Users/dududu/PhD_Code/frame_level_vq/Output/agreement/female_sp1/framewise_agreement_table.csv"
        ),
        "embedding_npz": Path(
            "/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings_wavlm_large/female_sp1/frame_level/sp1_female.npz"
        ),
    },
    "male_sp1": {
        "agreement_csv": Path(
            "/Users/dududu/PhD_Code/frame_level_vq/Output/agreement/male_sp1/framewise_agreement_table.csv"
        ),
        "embedding_npz": Path(
            "/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings_wavlm_large/male_sp1/frame_level/sp1_male.npz"
        ),
    },
}

ANNOTATOR_STATE_COLUMNS = ["AP_state", "TB_state", "XD_state"]

# Keep frames where at least this many annotators agree on the same non-none label.
MIN_VOTES_FOR_CONSENSUS = 2

# If True, discard labels like "breathy+creaky".
DROP_MULTI_LABEL_STATES = True

# Maximum allowed mismatch between agreement time and embedding time.
ALIGN_TOLERANCE_SEC = 0.011

OUTPUT_ROOT = Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/training_data")


def is_multi_label(label: str) -> bool:
    return "+" in label


def derive_consensus(row: pd.Series):
    states = [str(row[col]).strip().lower() for col in ANNOTATOR_STATE_COLUMNS]
    counts = Counter(states)

    if "none" in counts:
        del counts["none"]

    if not counts:
        return "none", 0, False, False

    top_label, top_votes = counts.most_common(1)[0]
    unanimous = len(set(states)) == 1 and top_label != "none"
    majority = top_votes >= MIN_VOTES_FOR_CONSENSUS

    return top_label, int(top_votes), bool(majority), bool(unanimous)


def nearest_time_match(source_times, target_times, tolerance_sec):
    """
    Map each source time to the nearest target time index within tolerance.
    Returns an array of target indices, with -1 for unmatched rows.
    """
    target_times = np.asarray(target_times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)

    idx = np.searchsorted(target_times, source_times)
    matched = np.full(source_times.shape[0], -1, dtype=np.int64)

    for i, (t, j) in enumerate(zip(source_times, idx)):
        candidates = []
        if j < len(target_times):
            candidates.append(j)
        if j > 0:
            candidates.append(j - 1)

        if not candidates:
            continue

        best = min(candidates, key=lambda k: abs(target_times[k] - t))
        if abs(target_times[best] - t) <= tolerance_sec:
            matched[i] = best

    return matched


def prepare_dataset(tag: str, cfg: dict):
    agreement_csv = cfg["agreement_csv"]
    embedding_npz = cfg["embedding_npz"]

    print("=" * 60)
    print(f"Preparing training data: {tag}")
    print("=" * 60)

    if not agreement_csv.exists():
        print(f"Missing agreement CSV: {agreement_csv}")
        return

    if not embedding_npz.exists():
        print(f"Missing embedding NPZ: {embedding_npz}")
        print("Skipping for now. Extract embeddings first, then rerun.")
        return

    out_dir = OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    agreement_df = pd.read_csv(agreement_csv)
    for col in ANNOTATOR_STATE_COLUMNS:
        if col not in agreement_df.columns:
            raise ValueError(f"Missing annotator column '{col}' in {agreement_csv}")

    consensus = agreement_df.apply(derive_consensus, axis=1, result_type="expand")
    consensus.columns = [
        "consensus_label",
        "consensus_votes",
        "consensus_mask",
        "unanimous_mask",
    ]
    agreement_df = pd.concat([agreement_df, consensus], axis=1)

    if DROP_MULTI_LABEL_STATES:
        agreement_df["single_label_mask"] = ~agreement_df["consensus_label"].map(is_multi_label)
    else:
        agreement_df["single_label_mask"] = True

    agreement_df["training_mask"] = (
        agreement_df["valid_mask"].astype(bool)
        & agreement_df["consensus_mask"].astype(bool)
        & agreement_df["single_label_mask"].astype(bool)
    )

    emb = np.load(embedding_npz)
    embedding_times = emb["times_sec"].astype(np.float32)
    agreement_times = agreement_df["time_sec"].to_numpy(dtype=np.float32)

    matched_embedding_idx = nearest_time_match(
        source_times=agreement_times,
        target_times=embedding_times,
        tolerance_sec=ALIGN_TOLERANCE_SEC,
    )
    agreement_df["embedding_index"] = matched_embedding_idx
    agreement_df["matched_embedding"] = matched_embedding_idx >= 0

    agreement_df["matched_time_sec"] = np.nan
    valid_rows = matched_embedding_idx >= 0
    agreement_df.loc[valid_rows, "matched_time_sec"] = embedding_times[matched_embedding_idx[valid_rows]]
    agreement_df["time_diff_sec"] = np.nan
    agreement_df.loc[valid_rows, "time_diff_sec"] = (
        agreement_df.loc[valid_rows, "matched_time_sec"] - agreement_df.loc[valid_rows, "time_sec"]
    )

    training_df = agreement_df[
        agreement_df["training_mask"] & agreement_df["matched_embedding"]
    ].copy()

    if training_df.empty:
        print("No training frames survived filtering.")
        training_df.to_csv(out_dir / "training_frame_table.csv", index=False)
        return

    keep_idx = training_df["embedding_index"].to_numpy(dtype=np.int64)
    labels = training_df["consensus_label"].to_numpy(dtype=object)

    to_save = {
        "embedding_index": keep_idx,
        "times_sec": embedding_times[keep_idx],
        "labels": labels,
    }

    for key in emb.files:
        if key == "times_sec":
            continue
        to_save[key] = emb[key][keep_idx]

    np.savez_compressed(out_dir / "training_frames.npz", **to_save)
    training_df.to_csv(out_dir / "training_frame_table.csv", index=False)

    summary = {
        "dataset": tag,
        "agreement_csv": str(agreement_csv),
        "embedding_npz": str(embedding_npz),
        "min_votes_for_consensus": MIN_VOTES_FOR_CONSENSUS,
        "drop_multi_label_states": DROP_MULTI_LABEL_STATES,
        "align_tolerance_sec": ALIGN_TOLERANCE_SEC,
        "num_agreement_rows": int(len(agreement_df)),
        "num_matched_rows": int(agreement_df["matched_embedding"].sum()),
        "num_training_rows": int(len(training_df)),
        "label_counts": training_df["consensus_label"].value_counts().to_dict(),
    }

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Matched rows: {summary['num_matched_rows']} / {summary['num_agreement_rows']}")
    print(f"Training rows kept: {summary['num_training_rows']}")
    print("Label counts:")
    print(training_df["consensus_label"].value_counts())
    print(f"Saved to: {out_dir}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tag, cfg in DATASETS.items():
        prepare_dataset(tag, cfg)


if __name__ == "__main__":
    main()
