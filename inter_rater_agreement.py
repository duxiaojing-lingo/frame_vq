#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from textgrid_utils import active_tiers_at_time, collect_boundaries, read_textgrid_intervals


# ============================================================
# USER SETTINGS
# ============================================================

DATASETS = {
    "female_sp1": [
        Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Female/AP Speaker001Session1IntTaskStudioQual_100425 (3).TextGrid"),
        Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Female/TB-Speaker001Session1IntTaskStudioQual_100425 (1).TextGrid"),
        Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Female/XD Speaker001Session1IntTaskStudioQual_100425 (1).TextGrid"),
    ],
    "male_sp1": [
        Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Male/AP 001-1-060310 (1).TextGrid"),
        Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Male/TB-001-1-060310 (3).TextGrid"),
        Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/Male/XD 001-1-060310 (1).TextGrid"),
    ],
}

ANNOTATOR_NAMES = ["AP", "TB", "XD"]

OUTPUT_ROOT = Path("/Users/dududu/PhD_Code/frame_level_vq/Output/agreement")


# ============================================================
# LABEL SETTINGS
# ============================================================

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

GRID_STEP_SEC = 0.02
IGNORE_BOUNDARY_MARGIN_SEC = 0.02


# ============================================================
# FUNCTIONS
# ============================================================

def canonical_active_tiers(textgrid, time):
    active = active_tiers_at_time(
        textgrid,
        list(TIER_NAME_MAP.keys()),
        time,
        positive_values=POSITIVE_VALUES,
    )

    canonical = set()
    for label in active:
        if label in TIER_NAME_MAP:
            canonical.add(TIER_NAME_MAP[label])

    return canonical


def encode_state(active_labels):
    if not active_labels:
        return "none"
    return "+".join(sorted(label.lower() for label in active_labels))


def cohen_kappa(labels_a, labels_b):
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    categories = sorted(set(labels_a.tolist()) | set(labels_b.tolist()))
    if not categories:
        return np.nan

    observed = np.mean(labels_a == labels_b)

    expected = 0.0
    for category in categories:
        pa = np.mean(labels_a == category)
        pb = np.mean(labels_b == category)
        expected += pa * pb

    if np.isclose(1.0 - expected, 0.0):
        return np.nan
    return (observed - expected) / (1.0 - expected)


def build_time_grid(textgrids):
    xmax = 0.0
    boundaries = []

    for tg in textgrids:
        for tier_name in TIER_NAME_MAP.keys():
            intervals = tg.get(tier_name, [])
            if intervals:
                xmax = max(xmax, max(interval.end for interval in intervals))

        boundaries.extend(
            collect_boundaries(
                tg,
                list(TIER_NAME_MAP.keys()),
                positive_only=True,
                positive_values=POSITIVE_VALUES,
            )
        )

    times = np.arange(0.0, xmax, GRID_STEP_SEC, dtype=np.float32) + (GRID_STEP_SEC / 2.0)

    valid_mask = np.ones(times.shape[0], dtype=bool)
    if IGNORE_BOUNDARY_MARGIN_SEC > 0:
        for boundary in boundaries:
            valid_mask &= np.abs(times - boundary) > IGNORE_BOUNDARY_MARGIN_SEC

    return times, valid_mask


def compute_majority_label(states):
    counts = Counter(states)
    top_label, top_count = counts.most_common(1)[0]
    return top_label, top_count


# ============================================================
# CORE PIPELINE
# ============================================================

def run_agreement(textgrid_paths, output_dir):

    textgrids = [read_textgrid_intervals(path) for path in textgrid_paths]
    times, valid_mask = build_time_grid(textgrids)

    per_annotator_states = {}
    per_annotator_binary = {tier: {} for tier in TARGET_TIERS}

    for annotator_name, textgrid in zip(ANNOTATOR_NAMES, textgrids):
        states = []

        for t in times:
            active = canonical_active_tiers(textgrid, float(t))
            states.append(encode_state(active))

        per_annotator_states[annotator_name] = np.asarray(states, dtype=object)

        for tier_name in TARGET_TIERS:
            binary = np.array(
                [
                    tier_name in canonical_active_tiers(textgrid, float(t))
                    for t in times
                ],
                dtype=bool,
            )
            per_annotator_binary[tier_name][annotator_name] = binary

    state_matrix = np.vstack([per_annotator_states[name] for name in ANNOTATOR_NAMES]).T
    non_none_mask = np.any(state_matrix != "none", axis=1)
    final_mask = valid_mask & non_none_mask

    # =====================
    # PAIRWISE
    # =====================
    pairwise_rows = []
    for a, b in combinations(ANNOTATOR_NAMES, 2):
        x = per_annotator_states[a][final_mask]
        y = per_annotator_states[b][final_mask]

        pairwise_rows.append({
            "annotator_a": a,
            "annotator_b": b,
            "num_points": int(final_mask.sum()),
            "observed_agreement": float(np.mean(x == y)),
            "cohen_kappa": float(cohen_kappa(x, y)),
        })

    # =====================
    # SAVE
    # =====================
    pd.DataFrame(pairwise_rows).to_csv(output_dir / "pairwise_state_agreement.csv", index=False)

    frame_df = pd.DataFrame({"time_sec": times})
    for name in ANNOTATOR_NAMES:
        frame_df[f"{name}_state"] = per_annotator_states[name]

    frame_df["valid_mask"] = valid_mask
    frame_df["non_none_mask"] = non_none_mask
    frame_df["final_mask"] = final_mask

    frame_df.to_csv(output_dir / "framewise_agreement_table.csv", index=False)

    print(f"Saved to: {output_dir}")


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    for tag, paths in DATASETS.items():
        print(f"\nProcessing {tag}")

        output_dir = OUTPUT_ROOT / tag
        output_dir.mkdir(parents=True, exist_ok=True)

        run_agreement(paths, output_dir)


if __name__ == "__main__":
    main()