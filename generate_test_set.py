#!/usr/bin/env python3
"""

⚠️ Note:
This version does NOT download any samples automatically.
Please manually place your real and fake image, audio, and video samples
inside the following directories before running this script:

    seed/images/
    seed/audio/
    seed/videos/

After that, run:
    python generate_test_set.py
"""

import os
import sys
import csv
import random
import math
import shutil
from pathlib import Path
import numpy as np
import cv2
import librosa

# Optional dependencies
try:
    import soundfile as sf
    SF_AVAILABLE = True
except Exception:
    SF_AVAILABLE = False

# ------------------ Directories ------------------
BASE_DIR = Path(__file__).resolve().parent
SEED_DIR = BASE_DIR / "seed"
TEST_DIR = BASE_DIR / "tests"
GEN_DIR = TEST_DIR / "generated"

IMG_DIR = SEED_DIR / "images"
AUD_DIR = SEED_DIR / "audio"
VID_DIR = SEED_DIR / "videos"

CSV_PATH = TEST_DIR / "test_labels.csv"

for d in [IMG_DIR, AUD_DIR, VID_DIR, GEN_DIR / "images", GEN_DIR / "audio", GEN_DIR / "videos"]:
    os.makedirs(d, exist_ok=True)


# ------------------ Helper Functions ------------------
def save_audio(y, sr, out_path):
    """Save audio robustly using soundfile if available, else fallback."""
    if SF_AVAILABLE:
        sf.write(out_path, y, sr)
    else:
        try:
            librosa.output.write_wav(out_path, y, sr)
        except Exception as e:
            print(f"[!] Audio write failed for {out_path}: {e}")


# ------------------ Augmentations ------------------
def aug_image(img, tag):
    """Apply small random augmentations to an image."""
    if tag == "bright_up":
        img = np.clip(img * 1.2, 0, 255).astype(np.uint8)
    elif tag == "bright_down":
        img = np.clip(img * 0.8, 0, 255).astype(np.uint8)
    elif tag == "flip":
        img = cv2.flip(img, 1)
    elif tag == "blur":
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif tag == "noise":
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def aug_audio(y, sr, tag):
    """Apply light augmentation to audio."""
    if tag == "pitch_up":
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
    elif tag == "pitch_down":
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2)
    elif tag == "stretch":
        return librosa.effects.time_stretch(y, 1.1)
    elif tag == "noise":
        return y + np.random.normal(0, 0.005, y.shape)
    return y


def aug_video_brightness(input_path, output_path, factor=1.1):
    """Apply brightness change to a video."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[!] Could not open video: {input_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
        out.write(frame)
    cap.release()
    out.release()


# ------------------ Generate dataset ------------------
def generate_dataset(per_source=2):
    print("=== Step 2: Generating augmented dataset ===")
    rows = []

    # ---- IMAGES ----
    for img_file in IMG_DIR.glob("*.jpg"):
        label = 1 if "fake" in img_file.stem else 0
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        for tag in ["bright_up", "flip", "blur", "noise"][:per_source]:
            aug = aug_image(img, tag)
            out_path = GEN_DIR / "images" / f"{img_file.stem}_{tag}.jpg"
            cv2.imwrite(str(out_path), aug)
            rows.append([str(out_path), "image", label])

    # ---- AUDIO ----
    for aud_file in AUD_DIR.glob("*.wav"):
        label = 1 if "fake" in aud_file.stem else 0
        try:
            y, sr = librosa.load(str(aud_file), sr=None)
            for tag in ["pitch_up", "noise", "stretch"][:per_source]:
                y2 = aug_audio(y, sr, tag)
                out_path = GEN_DIR / "audio" / f"{aud_file.stem}_{tag}.wav"
                save_audio(y2, sr, out_path)
                rows.append([str(out_path), "audio", label])
        except Exception as e:
            print(f"[!] Audio augment error: {e}")

    # ---- VIDEOS ----
    for vid_file in VID_DIR.glob("*.mp4"):
        label = 1 if "fake" in vid_file.stem else 0
        out_path = GEN_DIR / "videos" / f"{vid_file.stem}_bright.mp4"
        aug_video_brightness(vid_file, out_path, factor=1.1)
        rows.append([str(out_path), "video", label])

    # ---- CSV WRITE ----
    CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"[✓] Generated {len(rows)} augmented samples.")
    print(f"[✓] CSV file saved at: {CSV_PATH}")


# ------------------ Main ------------------
if __name__ == "__main__":
    print("=== Generating Deepfake Test Dataset ===")

    # ✅ User must manually add samples
    if not any(SEED_DIR.rglob("*.*")):
        print("[!] No files found in 'seed/' directory.")
        print("Please manually add your image (.jpg), audio (.wav), and video (.mp4) samples in:")
        print("   seed/images/")
        print("   seed/audio/")
        print("   seed/videos/")
        sys.exit(1)
    else:
        print("[✓] Seed folder detected with files. Proceeding with dataset generation...")

    generate_dataset()
    print("\nAll done! You can now run:")
    print("  python test_runner.py --config tests/test_labels.csv")
