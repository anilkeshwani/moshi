#!/usr/bin/env python

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from tqdm import tqdm


# Constants
BLOCK_SIZE = 100_000
MLS_SIZES = {"train": 10_808_037, "dev": 3_807, "test": 3_769}
DEVICE = torch.device("cuda")

# Mimi params
MIMI_SAMPLE_RATE = 24000
MIMI_FRAME_RATE = 12.5
MIMI_FRAME_SIZE = int(MIMI_SAMPLE_RATE / MIMI_FRAME_RATE)

# Paths
_MLS_SEGMENTS_PATH = "/mnt/scratch-artemis/shared/datasets/MLS/{}/segments.txt"
_MLS_AUDIO_DIR = "/mnt/scratch-artemis/shared/datasets/MLS/{}/audio"
_OUTPUT_DIR = "/mnt/scratch-artemis/anilkeshwani/mimi-mls/{}"

# Split to process
SPLIT = "test"
MLS_SPLIT_SIZE = MLS_SIZES[SPLIT]
MLS_SEGMENTS = _MLS_SEGMENTS_PATH.format(SPLIT)
MLS_AUDIO_DIR = Path(_MLS_AUDIO_DIR.format(SPLIT))
OUTPUT_DIR = Path(_OUTPUT_DIR.format(SPLIT))


def mls_id_to_path(mls_id: str, audio_dir: Path, suffix: str = ".flac") -> Path:
    speaker_id, book_id, file_specifier = mls_id.removesuffix(suffix).split("_")
    return (audio_dir / speaker_id / book_id / mls_id).with_suffix(suffix)


def load_and_preprocess(mls_id: str) -> tuple[str, torch.Tensor] | None:
    try:
        wav, sr = torchaudio.load(mls_id_to_path(mls_id, MLS_AUDIO_DIR))
        if wav.size(0) > 1:
            wav = wav[:1]
        if sr != MIMI_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, MIMI_SAMPLE_RATE)
        return mls_id, wav.squeeze(0)  # [T]
    except Exception as e:
        warnings.warn(f"Skipping {mls_id} due to error: {e}")
        return None


def pad_batch(wavs: list[torch.Tensor], frame_multiple: int) -> torch.Tensor:
    max_len = max(w.shape[0] for w in wavs)
    max_len = ((max_len + frame_multiple - 1) // frame_multiple) * frame_multiple
    padded = torch.zeros(len(wavs), 1, max_len)
    for i, w in enumerate(wavs):
        padded[i, 0, : w.shape[0]] = w
    return padded


@torch.inference_mode()
def mimi_encode_mls(idx_block: int, out_file: Path, batch_size: int = 64):
    # Load model
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=DEVICE)
    mimi.set_num_codebooks(8)
    mimi.eval()

    # Load IDs
    with open(MLS_SEGMENTS, "r") as f:
        all_ids = [line.strip().split(None, 1)[0] for line in f]
    assert len(all_ids) == MLS_SPLIT_SIZE

    start_idx = idx_block * BLOCK_SIZE
    end_idx = min((idx_block + 1) * BLOCK_SIZE, MLS_SPLIT_SIZE)
    mls_ids = all_ids[start_idx:end_idx]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(out_file, "x") as f:
        for i in tqdm(range(0, len(mls_ids), batch_size), desc=f"Block {idx_block}"):
            batch_ids = mls_ids[i : i + batch_size]
            results = [load_and_preprocess(mid) for mid in batch_ids]
            results = [r for r in results if r is not None]
            if not results:
                continue

            ids, wavs = zip(*results)
            wav_batch = pad_batch(list(wavs), MIMI_FRAME_SIZE).to(DEVICE)  # [B, 1, T]
            try:
                codes = mimi.encode(wav_batch).cpu().int()  # [B, 8, T]
            except Exception as e:
                warnings.warn(f"Batch encoding failed at idx {i}: {e}")
                continue

            for j, mls_id in enumerate(ids):
                sample = {"ID": mls_id}
                for rvq_idx in range(codes.shape[1]):
                    sample[f"RVQ_{rvq_idx}"] = codes[j, rvq_idx].tolist()
                f.write(json.dumps(sample) + "\n")


def main():
    idx_block = int(sys.argv[1])
    if idx_block < 0 or idx_block * BLOCK_SIZE >= MLS_SPLIT_SIZE:
        raise ValueError(f"Invalid block index {idx_block}. Must be in [0, {MLS_SPLIT_SIZE // BLOCK_SIZE}]")
    out_file = OUTPUT_DIR / f"mimi_mls_{idx_block}.jsonl"
    mimi_encode_mls(idx_block, out_file)


if __name__ == "__main__":
    main()
