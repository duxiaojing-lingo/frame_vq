#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract wav2vec-style SSL embeddings for the controlled Laver and Nolan data.

This mirrors the pilot embedding pipeline, but keeps the controlled materials
in their own output tree so they can later be used as a reference/prototype
training set.
"""

from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd
import librosa
import torch
from transformers import AutoProcessor, AutoModel


# =========================
# USER SETTINGS
# =========================
DATASETS = {
    "laver_controlled": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/controlled/laver"),
    "nolan_controlled": Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Data/controlled/nolan"),
}

OUTPUT_ROOT = Path("/Users/dududu/PhD_Code/frame_level_vq/pilot/Output/controlled_embeddings")

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
LAYERS_TO_SAVE = LAYERS_TO_SAVE = [0, 2, 3, 4, 5]

SAVE_FRAME_LEVEL = True
SAVE_SEGMENT_LEVEL = True

SEGMENT_MS = 200
SEGMENT_HOP_MS = 100

TRIM_SILENCE = False
SILENCE_TOP_DB = 35
DO_PEAK_NORMALIZE = False

MAX_CHUNK_SECONDS = 10
PRINT_EVERY = 1


# =========================
# DEVICE
# =========================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# =========================
# HELPERS
# =========================
def list_audio_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def safe_relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def make_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def infer_voice_quality_from_name(path: Path) -> str:
    name = path.stem.lower()
    if "breathy" in name:
        return "breathy"
    if "creaky" in name or "creak" in name:
        return "creaky"
    if "whispery" in name or "whisper" in name:
        return "whispery"
    return "unknown"


def infer_source_from_dataset(dataset_name: str) -> str:
    if "laver" in dataset_name.lower():
        return "laver"
    if "nolan" in dataset_name.lower():
        return "nolan"
    return "unknown"


def load_audio(path: Path, target_sr: int):
    y, sr = librosa.load(path.as_posix(), sr=target_sr, mono=True)
    y = y.astype(np.float32)

    if TRIM_SILENCE:
        y, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)

    if DO_PEAK_NORMALIZE:
        peak = np.max(np.abs(y)) if len(y) > 0 else 0.0
        if peak > 0:
            y = y / peak

    return y, target_sr


def get_model_inputs(processor, y, sr, device):
    batch = processor(y, sampling_rate=sr, return_tensors="pt", padding=False)

    model_inputs = {}
    if "input_values" in batch:
        model_inputs["input_values"] = batch["input_values"].to(device)
    elif "input_features" in batch:
        model_inputs["input_features"] = batch["input_features"].to(device)
    else:
        raise ValueError(
            "Processor output has neither 'input_values' nor 'input_features'. "
            f"Available keys: {list(batch.keys())}"
        )

    if "attention_mask" in batch:
        model_inputs["attention_mask"] = batch["attention_mask"].to(device)

    return model_inputs


def infer_frame_hop_seconds(model, sr=16000):
    conv_stride = getattr(model.config, "conv_stride", None)

    if conv_stride is None:
        fe = getattr(model, "feature_extractor", None)
        if fe is not None and hasattr(fe, "conv_layers"):
            strides = []
            for layer in fe.conv_layers:
                try:
                    strides.append(int(layer.conv.stride[0]))
                except Exception:
                    pass
            if len(strides) > 0:
                conv_stride = strides

    if conv_stride is None:
        raise ValueError("Could not infer conv_stride from model/config.")

    samples_per_frame = int(np.prod(conv_stride))
    hop_sec = samples_per_frame / sr
    return hop_sec, samples_per_frame


def extract_hidden_states_full(y, sr, processor, model, device):
    model_inputs = get_model_inputs(processor, y, sr, device)

    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)

    return [h.squeeze(0).detach().cpu().numpy().astype(np.float32) for h in outputs.hidden_states]


def extract_hidden_states_chunked(y, sr, processor, model, device, max_chunk_seconds):
    chunk_samples = int(max_chunk_seconds * sr)
    if chunk_samples <= 0:
        raise ValueError("max_chunk_seconds must be > 0.")

    all_layers = None
    min_chunk_samples = 400

    starts = list(range(0, len(y), chunk_samples))
    for i, start in enumerate(starts):
        end = min(len(y), start + chunk_samples)

        # Avoid sending an extremely short tail chunk into wav2vec2.
        if i > 0 and (end - start) < min_chunk_samples:
            prev_start = starts[i - 1]
            prev_end = start
            for layer_parts in all_layers:
                if layer_parts:
                    layer_parts.pop()
            y_chunk = y[prev_start:len(y)]
            chunk_hidden = extract_hidden_states_full(y_chunk, sr, processor, model, device)

            for j, arr in enumerate(chunk_hidden):
                all_layers[j].append(arr)
            break

        y_chunk = y[start:end]
        chunk_hidden = extract_hidden_states_full(y_chunk, sr, processor, model, device)

        if all_layers is None:
            all_layers = [[] for _ in range(len(chunk_hidden))]

        for i, arr in enumerate(chunk_hidden):
            all_layers[i].append(arr)

    return [np.concatenate(parts, axis=0) for parts in all_layers]


def extract_hidden_states(y, sr, processor, model, device, max_chunk_seconds=None):
    if max_chunk_seconds is None:
        return extract_hidden_states_full(y, sr, processor, model, device)
    return extract_hidden_states_chunked(y, sr, processor, model, device, max_chunk_seconds)


def build_times(num_frames, frame_hop_sec):
    return (np.arange(num_frames, dtype=np.float32) + 0.5) * np.float32(frame_hop_sec)


def pool_segments(layer_array, frame_hop_sec, segment_ms=200, hop_ms=100):
    T, _ = layer_array.shape
    seg_frames = max(1, int(round((segment_ms / 1000.0) / frame_hop_sec)))
    hop_frames = max(1, int(round((hop_ms / 1000.0) / frame_hop_sec)))

    starts = []
    ends = []
    means = []
    stds = []

    if T <= seg_frames:
        x = layer_array
        starts.append(0.0)
        ends.append(T * frame_hop_sec)
        means.append(x.mean(axis=0))
        stds.append(x.std(axis=0))
    else:
        last_start = T - seg_frames
        for s in range(0, last_start + 1, hop_frames):
            e = s + seg_frames
            x = layer_array[s:e]
            starts.append(s * frame_hop_sec)
            ends.append(e * frame_hop_sec)
            means.append(x.mean(axis=0))
            stds.append(x.std(axis=0))

    return (
        np.asarray(starts, dtype=np.float32),
        np.asarray(ends, dtype=np.float32),
        np.asarray(means, dtype=np.float32),
        np.asarray(stds, dtype=np.float32),
    )


def save_frame_level_npz(out_path, times_sec, selected_layers, hidden_states):
    to_save = {"times_sec": times_sec}
    for layer_idx in selected_layers:
        to_save[f"layer_{layer_idx:02d}"] = hidden_states[layer_idx]
    make_parent(out_path)
    np.savez_compressed(out_path, **to_save)


def save_segment_level_npz(out_path, frame_hop_sec, selected_layers, hidden_states):
    to_save = {
        "frame_hop_sec": np.float32(frame_hop_sec),
        "segment_ms": np.int32(SEGMENT_MS),
        "segment_hop_ms": np.int32(SEGMENT_HOP_MS),
    }

    first = True
    for layer_idx in selected_layers:
        starts, ends, means, stds = pool_segments(
            hidden_states[layer_idx],
            frame_hop_sec=frame_hop_sec,
            segment_ms=SEGMENT_MS,
            hop_ms=SEGMENT_HOP_MS,
        )
        if first:
            to_save["segment_start_sec"] = starts
            to_save["segment_end_sec"] = ends
            first = False
        to_save[f"layer_{layer_idx:02d}_mean"] = means
        to_save[f"layer_{layer_idx:02d}_std"] = stds

    make_parent(out_path)
    np.savez_compressed(out_path, **to_save)


def run_dataset(dataset_name: str, input_dir: Path, processor, model, frame_hop_sec):
    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list_audio_files(input_dir)
    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files found under {input_dir}")

    manifest_rows = []
    run_config = {
        "dataset_name": dataset_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "target_sr": TARGET_SR,
        "layers_to_save": LAYERS_TO_SAVE,
        "save_frame_level": SAVE_FRAME_LEVEL,
        "save_segment_level": SAVE_SEGMENT_LEVEL,
        "segment_ms": SEGMENT_MS,
        "segment_hop_ms": SEGMENT_HOP_MS,
        "trim_silence": TRIM_SILENCE,
        "do_peak_normalize": DO_PEAK_NORMALIZE,
        "max_chunk_seconds": MAX_CHUNK_SECONDS,
        "approx_frame_hop_sec": frame_hop_sec,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"DATASET      : {dataset_name}")
    print(f"INPUT_DIR    : {input_dir}")
    print(f"OUTPUT_DIR   : {output_dir}")
    print("=" * 60)

    for idx, wav_path in enumerate(audio_files, start=1):
        rel = safe_relpath(wav_path, input_dir)
        stem_safe = Path(rel).with_suffix("")
        print(f"[{idx}/{len(audio_files)}] {rel}")

        row = {
            "dataset": dataset_name,
            "source": infer_source_from_dataset(dataset_name),
            "relpath": rel,
            "audio_path": str(wav_path),
            "voice_quality_label": infer_voice_quality_from_name(wav_path),
            "status": "ok",
            "error": "",
            "duration_sec": np.nan,
            "num_audio_samples": np.nan,
            "num_ssl_frames": np.nan,
            "frame_hop_sec": frame_hop_sec,
            "frame_npz": "",
            "segment_npz": "",
        }

        try:
            y, sr = load_audio(wav_path, TARGET_SR)
            row["duration_sec"] = len(y) / sr
            row["num_audio_samples"] = len(y)

            hidden_states = extract_hidden_states(
                y=y,
                sr=sr,
                processor=processor,
                model=model,
                device=DEVICE,
                max_chunk_seconds=MAX_CHUNK_SECONDS,
            )

            max_layer = max(LAYERS_TO_SAVE)
            if max_layer >= len(hidden_states):
                raise ValueError(
                    f"Requested layer {max_layer}, but model returned only {len(hidden_states)} hidden states."
                )

            T = hidden_states[LAYERS_TO_SAVE[0]].shape[0]
            row["num_ssl_frames"] = T
            times_sec = build_times(T, frame_hop_sec)

            if SAVE_FRAME_LEVEL:
                frame_npz = output_dir / "frame_level" / f"{stem_safe}.npz"
                save_frame_level_npz(frame_npz, times_sec, LAYERS_TO_SAVE, hidden_states)
                row["frame_npz"] = str(frame_npz)

            if SAVE_SEGMENT_LEVEL:
                seg_npz = output_dir / "segment_level" / f"{stem_safe}.npz"
                save_segment_level_npz(seg_npz, frame_hop_sec, LAYERS_TO_SAVE, hidden_states)
                row["segment_npz"] = str(seg_npz)

        except Exception as e:
            row["status"] = "failed"
            row["error"] = f"{type(e).__name__}: {e}"
            print("FAILED")
            traceback.print_exc()

        manifest_rows.append(row)
        if idx % PRINT_EVERY == 0:
            pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Manifest saved to: {output_dir / 'manifest.csv'}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONTROLLED SSL EMBEDDING EXTRACTION")
    print("=" * 60)
    print(f"MODEL_NAME  : {MODEL_NAME}")
    print(f"DEVICE      : {DEVICE}")
    print(f"TARGET_SR   : {TARGET_SR}")
    print(f"LAYERS      : {LAYERS_TO_SAVE}")
    print("=" * 60)

    print("Loading processor/model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    frame_hop_sec, samples_per_frame = infer_frame_hop_seconds(model, sr=TARGET_SR)
    print(f"Approx frame hop: {frame_hop_sec:.6f} sec ({samples_per_frame} samples)")

    for dataset_name, input_dir in DATASETS.items():
        run_dataset(
            dataset_name=dataset_name,
            input_dir=input_dir,
            processor=processor,
            model=model,
            frame_hop_sec=frame_hop_sec,
        )

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
