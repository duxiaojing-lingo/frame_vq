#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# ============================================================
# USER SETTINGS
# ============================================================
DATASETS = {
    "female_sp1": {
        "npz": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/training_data/female_sp1/training_frames.npz"),
        "csv": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/training_data/female_sp1/training_frame_table.csv"),
    },
    "male_sp1": {
        "npz": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/training_data/male_sp1/training_frames.npz"),
        "csv": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/training_data/male_sp1/training_frame_table.csv"),
    },
}

OUTPUT_ROOT = Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/pca_lda")

LAYER_KEY = "layer_24"
RANDOM_STATE = 42


# ============================================================
# PLOTTING
# ============================================================
def save_scatter(df, x_col, y_col, title, out_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="label",
        style="label",
        s=70,
        alpha=0.85,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_strip(df, x_col, title, out_path):
    plt.figure(figsize=(8, 4))
    sns.stripplot(
        data=df,
        x=x_col,
        y="label",
        orient="h",
        size=6,
        alpha=0.8,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# CORE FUNCTION
# ============================================================
def run_one_dataset(tag, npz_path, csv_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    X = data[LAYER_KEY].astype(np.float32)
    y = data["labels"].astype(str)
    times = data["times_sec"].astype(np.float32)

    unique_labels = np.unique(y)

    # =====================
    # STANDARDIZE
    # =====================
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # =====================
    # PCA
    # =====================
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(Xz)

    pca_df = pd.DataFrame({
        "time_sec": times,
        "label": y,
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
    })

    pca_df.to_csv(output_dir / f"{tag}_pca_coords.csv", index=False)

    save_scatter(
        pca_df,
        "pc1",
        "pc2",
        f"PCA ({tag})",
        output_dir / f"{tag}_pca.png"
    )

    # =====================
    # LDA (SAFE VERSION)
    # =====================
    n_classes = len(unique_labels)
    n_components = min(n_classes - 1, 2)

    if n_components >= 1:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(Xz, y)

        lda_df = pd.DataFrame({
            "time_sec": times,
            "label": y,
            "ld1": X_lda[:, 0],
        })

        if n_components >= 2:
            lda_df["ld2"] = X_lda[:, 1]

            save_scatter(
                lda_df,
                "ld1",
                "ld2",
                f"LDA ({tag})",
                output_dir / f"{tag}_lda.png"
            )
        else:
            save_strip(
                lda_df,
                "ld1",
                f"LDA ({tag})",
                output_dir / f"{tag}_lda.png"
            )

        lda_df.to_csv(output_dir / f"{tag}_lda_coords.csv", index=False)

    # =====================
    # CENTROIDS (PER DATASET)
    # =====================
    centroid_rows = []
    for label in unique_labels:
        mask = y == label
        centroid_rows.append({
            "label": label,
            "count": int(mask.sum()),
            "pc1_mean": float(pca_df.loc[mask, "pc1"].mean()),
            "pc2_mean": float(pca_df.loc[mask, "pc2"].mean()),
        })

    pd.DataFrame(centroid_rows).to_csv(
        output_dir / f"{tag}_centroids.csv",
        index=False
    )

    # =====================
    # SUMMARY
    # =====================
    summary = {
        "dataset": tag,
        "n_frames": int(X.shape[0]),
        "label_counts": pd.Series(y).value_counts().to_dict(),
        "pca_variance": pca.explained_variance_ratio_.tolist(),
    }

    with open(output_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # =====================
    # PRINT
    # =====================
    print(f"\n=== {tag} ===")
    print(pd.Series(y).value_counts())
    print("PCA variance:", pca.explained_variance_ratio_)


# ============================================================
# MAIN
# ============================================================
def main():
    for tag, paths in DATASETS.items():
        print(f"\nProcessing {tag}")
        out_dir = OUTPUT_ROOT / tag

        run_one_dataset(
            tag,
            paths["npz"],
            paths["csv"],
            out_dir
        )

    print("\nDONE: PCA + LDA complete")


if __name__ == "__main__":
    main()