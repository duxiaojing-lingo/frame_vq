#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-speaker voice-quality classification on consensus-labeled SSL frames.

This script:
1) loads train-ready frame embeddings for female and male speakers
2) trains on one speaker and tests on the other
3) compares predictions against human consensus labels
4) reports results for each SSL layer
5) also runs one-vs-rest binary tasks

Important:
- The human consensus labels are the ground truth here.
- The embeddings are the input features, not a competing label source.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# USER SETTINGS
# ============================================================
DATASETS = {
    "female_sp1": Path(
        "/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/training_data/female_sp1/training_frames.npz"
    ),
    "male_sp1": Path(
        "/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/training_data/male_sp1/training_frames.npz"
    ),
}

LAYERS_TO_TEST = ["layer_00", "layer_02", "layer_03", "layer_04", "layer_05"]
LABELS = ["breathy", "creaky", "whispery"]
BINARY_TASKS = ["breathy", "creaky", "whispery"]

OUTPUT_DIR = Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/cross_speaker_classifier")
RANDOM_STATE = 42


def load_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    return {
        "times_sec": data["times_sec"].astype(np.float32),
        "labels": data["labels"].astype(str),
        **{key: data[key].astype(np.float32) for key in data.files if key.startswith("layer_")},
    }


def fit_predict_logreg(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)

    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_z, y_train)
    y_pred = clf.predict(X_test_z)
    return y_pred


def save_confusion(cm, labels, title, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_multiclass(train_name, test_name, train_data, test_data):
    rows = []

    for layer_key in LAYERS_TO_TEST:
        if layer_key not in train_data or layer_key not in test_data:
            continue

        X_train = train_data[layer_key]
        y_train = train_data["labels"]
        X_test = test_data[layer_key]
        y_test = test_data["labels"]

        y_pred = fit_predict_logreg(X_train, y_train, X_test)

        report = classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        rows.append(
            {
                "train_speaker": train_name,
                "test_speaker": test_name,
                "layer": layer_key,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test, y_pred, labels=LABELS, average="weighted", zero_division=0)),
                "breathy_f1": float(report.get("breathy", {}).get("f1-score", 0.0)),
                "creaky_f1": float(report.get("creaky", {}).get("f1-score", 0.0)),
                "whispery_f1": float(report.get("whispery", {}).get("f1-score", 0.0)),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(
        by=["balanced_accuracy", "macro_f1", "accuracy"],
        ascending=False,
    )

    if not result_df.empty:
        best_layer = result_df.iloc[0]["layer"]
        y_pred_best = fit_predict_logreg(
            train_data[best_layer],
            train_data["labels"],
            test_data[best_layer],
        )
        cm = confusion_matrix(test_data["labels"], y_pred_best, labels=LABELS)
        save_confusion(
            cm,
            LABELS,
            title=f"{train_name} -> {test_name} ({best_layer})",
            out_path=OUTPUT_DIR / f"{train_name}_to_{test_name}_{best_layer}_confusion.png",
        )

        report_text = classification_report(
            test_data["labels"],
            y_pred_best,
            labels=LABELS,
            zero_division=0,
            digits=3,
        )
        (OUTPUT_DIR / f"{train_name}_to_{test_name}_{best_layer}_report.txt").write_text(
            report_text,
            encoding="utf-8",
        )

    return result_df


def evaluate_binary(train_name, test_name, train_data, test_data):
    rows = []

    for target_label in BINARY_TASKS:
        y_train = (train_data["labels"] == target_label).astype(int)
        y_test = (test_data["labels"] == target_label).astype(int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            rows.append(
                {
                    "train_speaker": train_name,
                    "test_speaker": test_name,
                    "target_label": target_label,
                    "layer": "",
                    "n_train_positive": int(y_train.sum()),
                    "n_test_positive": int(y_test.sum()),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "f1": np.nan,
                    "note": "Skipped because train/test lacks both classes.",
                }
            )
            continue

        for layer_key in LAYERS_TO_TEST:
            if layer_key not in train_data or layer_key not in test_data:
                continue

            scaler = StandardScaler()
            X_train_z = scaler.fit_transform(train_data[layer_key])
            X_test_z = scaler.transform(test_data[layer_key])

            clf = LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            )
            clf.fit(X_train_z, y_train)
            y_pred = clf.predict(X_test_z)

            rows.append(
                {
                    "train_speaker": train_name,
                    "test_speaker": test_name,
                    "target_label": target_label,
                    "layer": layer_key,
                    "n_train_positive": int(y_train.sum()),
                    "n_test_positive": int(y_test.sum()),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "note": "",
                }
            )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = {name: load_dataset(path) for name, path in DATASETS.items()}

    all_multi = []
    all_binary = []

    directions = [
        ("female_sp1", "male_sp1"),
        ("male_sp1", "female_sp1"),
    ]

    for train_name, test_name in directions:
        multi_df = evaluate_multiclass(
            train_name,
            test_name,
            loaded[train_name],
            loaded[test_name],
        )
        binary_df = evaluate_binary(
            train_name,
            test_name,
            loaded[train_name],
            loaded[test_name],
        )

        multi_df.to_csv(OUTPUT_DIR / f"{train_name}_to_{test_name}_multiclass.csv", index=False)
        binary_df.to_csv(OUTPUT_DIR / f"{train_name}_to_{test_name}_binary.csv", index=False)

        all_multi.append(multi_df)
        all_binary.append(binary_df)

        print("=" * 60)
        print(f"{train_name} -> {test_name}")
        print("=" * 60)
        print("Multiclass results:")
        print(multi_df.to_string(index=False))
        print()
        print("Best binary results by target:")
        if not binary_df.empty:
            best_binary = (
                binary_df.dropna(subset=["balanced_accuracy"])
                .sort_values(["target_label", "balanced_accuracy", "f1"], ascending=[True, False, False])
                .groupby("target_label", as_index=False)
                .head(1)
            )
            print(best_binary.to_string(index=False))

    combined_multi = pd.concat(all_multi, ignore_index=True)
    combined_binary = pd.concat(all_binary, ignore_index=True)

    combined_multi.to_csv(OUTPUT_DIR / "all_multiclass_results.csv", index=False)
    combined_binary.to_csv(OUTPUT_DIR / "all_binary_results.csv", index=False)

    summary = {
        "datasets": {k: str(v) for k, v in DATASETS.items()},
        "layers_tested": LAYERS_TO_TEST,
        "labels": LABELS,
        "binary_tasks": BINARY_TASKS,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
