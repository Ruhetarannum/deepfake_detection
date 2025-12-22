"""
Run:
  - python test_runner.py --config tests/test_labels.csv
"""

import argparse
import csv
import os
import sys
import time
import math
import tempfile
import shutil
import statistics
from collections import defaultdict

# tracemalloc fallback for Python memory measurement
import tracemalloc

# image/audio/video processing
import cv2
import numpy as np

# for audio processing (already used in your app)
import librosa

# Import your detector classes from your app file.
# If your main file is named something else, update this import.
try:
    from app import DeepfakeDetector, AudioDeepfakeDetector
except Exception as e:
    print("Error importing detectors from app.py. Ensure test_runner.py is in same folder as your app file and that app.py defines DeepfakeDetector and AudioDeepfakeDetector.")
    print("Import error:", e)
    sys.exit(1)


# -----------------------
# Utilities & perturbations
# -----------------------
def adjust_brightness(image, factor=1.0):
    img = image.astype(np.float32) * factor
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def add_gaussian_noise(image, mean=0, var=10):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def gaussian_blur(image, ksize=3):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)

def measure_time(func, *args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    t1 = time.time()
    return result, (t1 - t0)


# -----------------------
# Metrics (no sklearn)
# -----------------------
def compute_confusion_counts(y_true, y_pred):
    TP = FP = TN = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: TP += 1
        if t == 0 and p == 1: FP += 1
        if t == 0 and p == 0: TN += 1
        if t == 1 and p == 0: FN += 1
    return TP, FP, TN, FN

def precision_recall_f1(y_true, y_pred):
    TP, FP, TN, FN = compute_confusion_counts(y_true, y_pred)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    return accuracy, precision, recall, f1, (TP, FP, TN, FN)


# -----------------------
# Test implementations
# -----------------------
class Tester:
    def __init__(self, image_model_path, audio_model_path, tmpdir=None):
        self.image_detector = None
        self.audio_detector = None

        try:
            self.image_detector = DeepfakeDetector(image_model_path)
        except Exception as e:
            print("Warning: Could not load image/video model:", e)

        try:
            self.audio_detector = AudioDeepfakeDetector(audio_model_path)
        except Exception as e:
            print("Warning: Could not load audio model:", e)

        self.tmpdir = tmpdir or tempfile.mkdtemp(prefix="test_runner_")
        self.results = {"image": [], "audio": [], "video": []}

    def cleanup(self):
        try:
            shutil.rmtree(self.tmpdir)
        except Exception:
            pass

    def predict_image(self, path):
        if self.image_detector is None:
            raise RuntimeError("Image detector not loaded")
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError("Could not read image: " + path)
        (pred, conf), t = measure_time(self.image_detector.predict, img)
        label = 1 if pred > 0.5 else 0
        return label, float(conf), t

    def predict_audio(self, path):
        if self.audio_detector is None:
            raise RuntimeError("Audio detector not loaded")
        (pred, conf), t = measure_time(self.audio_detector.predict, path)
        label = 1 if pred > 0.5 else 0
        return label, float(conf), t

    def predict_video(self, path, max_frames=10):
        if self.image_detector is None:
            raise RuntimeError("Image detector not loaded")
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            raise RuntimeError("Could not read video frames: " + path)
        step = max(1, total_frames // max_frames)
        indices = list(range(0, total_frames, step))[:max_frames]
        times, confidences, preds = [], [], []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            (pred, conf), t = measure_time(self.image_detector.predict, frame)
            times.append(t)
            confidences.append(conf)
            preds.append(1 if pred > 0.5 else 0)
        cap.release()
        if not preds:
            raise RuntimeError("No valid frames for video: " + path)
        avg_conf = float(np.mean(confidences))
        avg_pred = float(np.mean(preds))
        label = 1 if avg_pred > 0.5 else 0
        return label, avg_conf, float(np.mean(times))

    def run_accuracy_test(self, cases, verbose=True):
        outputs = {"image": [], "audio": [], "video": []}
        time_stats = defaultdict(list)
        for c in cases:
            path = c["path"]
            modality = c.get("modality", "image")
            true_label = int(c.get("label", 0))
            try:
                if modality == "image":
                    pred_label, conf, t = self.predict_image(path)
                elif modality == "audio":
                    pred_label, conf, t = self.predict_audio(path)
                elif modality == "video":
                    pred_label, conf, t = self.predict_video(path)
                else:
                    continue
                outputs[modality].append((path, true_label, pred_label, conf, t))
                time_stats[modality].append(t)
                if verbose:
                    print(f"[{modality}] {os.path.basename(path)} -> true={true_label} pred={pred_label} conf={conf:.3f} time={t:.3f}s")
            except Exception as e:
                print(f"ERROR on {path} ({modality}): {e}")
        metrics = {}
        for mod in ["image", "audio", "video"]:
            rows = outputs[mod]
            if not rows:
                metrics[mod] = None
                continue
            y_true = [r[1] for r in rows]
            y_pred = [r[2] for r in rows]
            confs = [r[3] for r in rows]
            times = [r[4] for r in rows]
            acc, prec, rec, f1, conf_counts = precision_recall_f1(y_true, y_pred)
            metrics[mod] = {
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "counts": conf_counts,
                "avg_confidence": float(np.mean(confs)) if confs else None,
                "time_mean": float(np.mean(times)) if times else None,
                "time_median": float(np.median(times)) if times else None,
                "time_p95": float(np.percentile(times, 95)) if times else None,
                "cases": rows
            }
        return metrics

    # Confidence stability test
    def run_confidence_stability(self, cases, n_perturb=4):
        stability_results = {"image": [], "audio": [], "video": []}
        for c in cases:
            path = c["path"]
            modality = c.get("modality", "image")
            try:
                if modality == "image":
                    img = cv2.imread(path)
                    if img is None:
                        raise RuntimeError("Could not load image")
                    (base_pred, base_conf, _) = self.predict_image(path)
                    base_label = 1 if base_pred > 0.5 else 0
                    confs = [base_conf]
                    flip_count = 0
                    perturbs = [
                        adjust_brightness(img, 0.8),
                        adjust_brightness(img, 1.2),
                        gaussian_blur(img, 3),
                        add_gaussian_noise(img, var=20)
                    ][:n_perturb]
                    for p in perturbs:
                        (pred_val, conf_val), _ = measure_time(self.image_detector.predict, p)
                        pred_label_int = 1 if pred_val > 0.5 else 0
                        confs.append(conf_val)
                        if pred_label_int != base_label:
                            flip_count += 1
                    stability_results["image"].append({
                        "path": path, "base_label": base_label,
                        "mean_conf": float(np.mean(confs)),
                        "std_conf": float(np.std(confs)),
                        "flip_count": flip_count, "n_perturb": len(confs)-1
                    })
                elif modality == "audio":
                    (base_pred, base_conf, _) = self.predict_audio(path)
                    base_label = 1 if base_pred > 0.5 else 0
                    confs = [base_conf]
                    flip_count = 0
                    y, sr = librosa.load(path, sr=None)
                    import soundfile as sf
                    for var in [0.001, 0.005, 0.01][:n_perturb]:
                        noisy = y + np.random.normal(0, var, y.shape)
                        tmpf = os.path.join(self.tmpdir, f"pert_{os.path.basename(path)}_{var:.3f}.wav")
                        sf.write(tmpf, noisy, sr)
                        (pred_label, conf, _) = self.predict_audio(tmpf)
                        confs.append(conf)
                        if pred_label != base_label:
                            flip_count += 1
                        os.unlink(tmpf)
                    stability_results["audio"].append({
                        "path": path, "base_label": base_label,
                        "mean_conf": float(np.mean(confs)),
                        "std_conf": float(np.std(confs)),
                        "flip_count": flip_count, "n_perturb": len(confs)-1
                    })
                elif modality == "video":
                    label, conf, t = self.predict_video(path, max_frames=6)
                    stability_results["video"].append({
                        "path": path, "mean_conf": conf, "std_conf": 0.0,
                        "flip_count": 0, "n_frames_tested": 6
                    })
            except Exception as e:
                print("Stability test error for", path, e)
        return stability_results

    # Performance tests
    def run_performance_test(self, cases, iterations=3):
        perf = {"image": [], "audio": [], "video": []}
        for mod in ["image", "audio", "video"]:
            mod_cases = [c for c in cases if c.get("modality") == mod]
            if not mod_cases:
                continue
            times = []
            tracemalloc.start()
            snapshot_before = tracemalloc.take_snapshot()
            subset = mod_cases[:max(1, min(5, len(mod_cases)))]
            for it in range(iterations):
                for c in subset:
                    try:
                        p0 = time.time()
                        if mod == "image":
                            _ = self.predict_image(c["path"])
                        elif mod == "audio":
                            _ = self.predict_audio(c["path"])
                        else:
                            _ = self.predict_video(c["path"])
                        p1 = time.time()
                        times.append(p1 - p0)
                    except Exception as e:
                        print("Perf error:", e)
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            mem_used = sum([stat.size_diff for stat in stats[:10]])
            tracemalloc.stop()
            perf[mod] = {
                "times_mean": float(np.mean(times)) if times else None,
                "times_median": float(np.median(times)) if times else None,
                "times_p95": float(np.percentile(times, 95)) if times else None,
                "mem_used_bytes": int(mem_used)
            }
        return perf


# -----------------------
# Helpers for loading test cases
# -----------------------
def load_cases_from_csv(csv_path):
    cases = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            path = row[0].strip()
            modality = row[1].strip() if len(row) > 1 and row[1].strip() else "image"
            label = int(row[2].strip()) if len(row) > 2 and row[2].strip() else 0
            cases.append({"path": path, "modality": modality, "label": label})
    return cases

def discover_cases_auto(base_dir="tests"):
    cases = []
    for mod, sub in [("image","images"), ("audio","audio"), ("video","videos")]:
        p = os.path.join(base_dir, sub)
        if not os.path.exists(p):
            continue
        for fname in os.listdir(p):
            path = os.path.join(p, fname)
            lower = fname.lower()
            label = 1 if "fake" in lower else 0
            cases.append({"path": path, "modality": mod, "label": label})
    return cases


# -----------------------
# CLI entrypoint
# -----------------------
def run_all_tests(config_csv=None):
    image_model_path = r"C:\Users\ruhet\Downloads\DeepFake Audio and Video-20251109T104526Z-1-001\DeepFake Audio and Video\deepfake-det\deepfake-det\Models\best_mobilenet_lstm_model.keras"
    audio_model_path = r"C:\Users\ruhet\Downloads\DeepFake Audio and Video-20251109T104526Z-1-001\DeepFake Audio and Video\deepfake-det\deepfake-det\Models\deepfake_audio_detector.h5"

    if config_csv and os.path.exists(config_csv):
        cases = load_cases_from_csv(config_csv)
    else:
        cases = discover_cases_auto()
        if not cases:
            print("No test cases found. Provide a --config CSV or place test files under './tests/images', './tests/audio', './tests/videos'.")
            return

    tester = Tester(image_model_path=image_model_path, audio_model_path=audio_model_path)
    try:
        print("\n=== Running Accuracy Tests ===")
        metrics = tester.run_accuracy_test(cases, verbose=True)
        print("\n=== Accuracy Summary ===")
        for mod, m in metrics.items():
            if m is None:
                print(f"[{mod}] No cases run.")
            else:
                print(f"[{mod}] Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} Rec={m['recall']:.3f} F1={m['f1']:.3f} AvgConf={m['avg_confidence']:.3f} TimeMean={m['time_mean']:.3f}s")

        print("\n=== Running Confidence Stability Tests ===")
        stability = tester.run_confidence_stability(cases)
        for mod, res in stability.items():
            if not res:
                print(f"[{mod}] No stability results.")
            else:
                for r in res:
                    if mod == "image":
                        print(f"[{mod}] {os.path.basename(r['path'])} mean_conf={r['mean_conf']:.3f} std={r['std_conf']:.3f} flips={r['flip_count']}/{r['n_perturb']}")
                    elif mod == "audio":
                        print(f"[{mod}] {os.path.basename(r['path'])} mean_conf={r['mean_conf']:.3f} std={r['std_conf']:.3f} flips={r['flip_count']}/{r['n_perturb']}")
                    else:
                        print(f"[{mod}] {os.path.basename(r['path'])} mean_conf={r['mean_conf']:.3f} std={r['std_conf']:.3f} flips={r['flip_count']}/{r['n_frames_tested']}")

        print("\n=== Running Performance Tests ===")
        perf = tester.run_performance_test(cases)
        for mod, p in perf.items():
            if not p:
                print(f"[{mod}] No perf results.")
            else:
                mem_bytes = p.get("mem_used_bytes")
                mem_str = f"{mem_bytes/1024/1024:.2f} MB" if mem_bytes else "N/A"
                print(f"[{mod}] time_mean={p['times_mean']:.3f}s median={p['times_median']:.3f}s p95={p['times_p95']:.3f}s mem_delta={mem_str}")

    finally:
        tester.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test runner for Deepfake detection models")
    parser.add_argument("--config", type=str, default=None, help="CSV of test cases: path,modality,label")
    args = parser.parse_args()
    run_all_tests(config_csv=args.config)
