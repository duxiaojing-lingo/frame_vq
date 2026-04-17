#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# USER SETTINGS
# ============================================================
REPRESENTATION = "frame"
LAYERS_TO_TEST = ["layer_00", "layer_02", "layer_03", "layer_04", "layer_05"]
LABELS = ["breathy", "creaky", "whispery"]
ALIGN_TOLERANCE_SEC = 0.011
RANDOM_STATE = 42

CONTROLLED_MANIFESTS = [
    Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/controlled_embeddings/laver_controlled/manifest.csv"),
    Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/controlled_embeddings/nolan_controlled/manifest.csv"),
]

PILOT_DATASETS = {
    "female_sp1": {
        "frame_npz": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings/female_sp1/frame_level/sp1_female.npz"),
        "segment_npz": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings/female_sp1/segment_level/sp1_female.npz"),
        "agreement_csv": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/agreement/female_sp1/framewise_agreement_table.csv"),
    },
    "male_sp1": {
        "frame_npz": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings/male_sp1/frame_level/sp1_male.npz"),
        "segment_npz": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/embeddings/male_sp1/segment_level/sp1_male.npz"),
        "agreement_csv": Path("/Users/dududu/PhD_Code/frame_level_vq/Output/agreement/male_sp1/framewise_agreement_table.csv"),
    },
}

OUTPUT_DIR = Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/controlled_to_pilot_classifier")


# ============================================================
# UTIL
# ============================================================

def is_single_label(label: str) -> bool:
    return label not in {"", "none"} and "+" not in label


def nearest_time_match(source_times, target_times, tolerance_sec):
    target_times = np.asarray(target_times)
    source_times = np.asarray(source_times)

    idx = np.searchsorted(target_times, source_times)
    matched = np.full(len(source_times), -1)

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


def get_npz_times_and_features(npz_path: Path, layer_key: str, representation: str):
    data = np.load(npz_path, allow_pickle=True)

    if representation == "frame":
        return data["times_sec"], data[layer_key]

    if representation == "segment":
        starts = data["segment_start_sec"]
        ends = data["segment_end_sec"]
        times = (starts + ends) / 2.0
        feats = data[f"{layer_key}_mean"]
        return times, feats

    raise ValueError


# ============================================================
# TRAINING DATA
# ============================================================

def load_controlled_training(layer_key, representation):
    X_parts, y_parts = [], []

    for manifest_path in CONTROLLED_MANIFESTS:
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["status"] == "ok"]

        npz_col = "frame_npz" if representation == "frame" else "segment_npz"

        for row in manifest.itertuples(index=False):
            npz_path = Path(getattr(row, npz_col))
            _, feats = get_npz_times_and_features(npz_path, layer_key, representation)

            label = str(row.voice_quality_label).strip().lower()
            if label not in LABELS:
                continue

            X_parts.append(feats)
            y_parts.append(np.full(feats.shape[0], label))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    return X, y


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for layer_key in LAYERS_TO_TEST:
        print(f"\n========== {layer_key} ==========")

        X_train, y_train = load_controlled_training(layer_key, REPRESENTATION)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)

        for dataset_name, cfg in PILOT_DATASETS.items():
            print(f"\n--- {dataset_name} ---")

            npz_path = cfg["frame_npz"]
            emb_times, X_test = get_npz_times_and_features(npz_path, layer_key, REPRESENTATION)

            agreement = pd.read_csv(cfg["agreement_csv"])
            agreement["time_sec"] = agreement["time_sec"].astype(float)

            idx = nearest_time_match(agreement["time_sec"], emb_times, ALIGN_TOLERANCE_SEC)
            agreement["embedding_index"] = idx
            agreement["matched"] = idx >= 0

            X_test = scaler.transform(X_test)
            y_pred_all = clf.predict(X_test)

            for annotator_col in ["AP_state", "TB_state", "XD_state"]:
                labels = agreement[annotator_col].astype(str).str.lower()
                mask = agreement["matched"] & agreement["valid_mask"]
                mask &= labels.map(is_single_label)

                eval_df = agreement.loc[mask].copy()
                if eval_df.empty:
                    continue

                eval_df["true"] = labels[mask]
                eval_df["pred"] = y_pred_all[eval_df["embedding_index"]]

                # =====================
                # GLOBAL METRICS
                # =====================
                acc = accuracy_score(eval_df["true"], eval_df["pred"])
                bal = balanced_accuracy_score(eval_df["true"], eval_df["pred"])

                # =====================
                # PER-CLASS METRICS
                # =====================
                report = classification_report(
                    eval_df["true"],
                    eval_df["pred"],
                    labels=LABELS,
                    output_dict=True,
                    zero_division=0,
                )

                cm = confusion_matrix(
                    eval_df["true"],
                    eval_df["pred"],
                    labels=LABELS,
                )

                # save confusion matrix
                pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
                    OUTPUT_DIR / f"{layer_key}_{dataset_name}_{annotator_col}_cm.csv"
                )

                # save predictions
                eval_df.to_csv(
                    OUTPUT_DIR / f"{layer_key}_{dataset_name}_{annotator_col}_predictions.csv",
                    index=False,
                )

                # =====================
                # PRINT BEST CLASS
                # =====================
                best_class = None
                best_recall = -1

                for cls in LABELS:
                    if cls in report:
                        r = report[cls]["recall"]
                        print(f"{cls:10s} recall={r:.3f} precision={report[cls]['precision']:.3f}")

                        if r > best_recall:
                            best_recall = r
                            best_class = cls

                        all_rows.append(
                            {
                                "layer": layer_key,
                                "dataset": dataset_name,
                                "annotator": annotator_col,
                                "class": cls,
                                "recall": r,
                                "precision": report[cls]["precision"],
                                "f1": report[cls]["f1-score"],
                            }
                        )

                print(f">>> BEST CLASS: {best_class} (recall={best_recall:.3f})")

    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / "per_class_results.csv", index=False)

    print("\nDONE")


if __name__ == "__main__":
    main()