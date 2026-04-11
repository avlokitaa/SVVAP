"""
Deepfake Detector — Dynamic Pseudo-Fake Generation Strategy
============================================================
Role    : Hackathon audio-visual sync detector
Strategy: Train on 50-100 real videos; generate pseudo-fakes on-the-fly
          so the model learns physics of speech, not just memorized frames.

Dependencies
------------
    pip install librosa opencv-python-headless numpy scikit-learn torch

Usage
-----
    # Train:
    from deepfake_detector import train_on_seed_data
    train_on_seed_data("path/to/real_videos/")

    # Demo a single file:
    from deepfake_detector import analyze_single_file
    score = analyze_single_file("path/to/test_video.mp4")
    print(f"Sync score: {score:.3f}  ({'REAL' if score > 0.5 else 'FAKE'})")
"""

import os
from typing import Optional
import random
from moviepy import VideoFileClip
import numpy as np
import librosa
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Constants ────────────────────────────────────────────────────────────────

FPS            = 30                     # Target video frame rate
HOP_LENGTH     = int(22050 / FPS)       # ~735 samples → 1 audio chunk per frame
SR             = 22050                  # Standard sample rate (librosa default)
N_MFCC         = 13                     # Number of MFCC coefficients
MOUTH_SIZE     = (64, 64)               # Mouth crop resolution
INPUT_DIM      = (64 * 64) + N_MFCC + N_MFCC   # 4096 + 13 + 13 = 4122

MODEL_PATH     = "deepfake_model.pt"
SCALER_PATH    = "deepfake_scaler.pkl"

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Temporal-shift range (seconds) for pseudo-fake augmentation
SHIFT_MIN_SEC  = 0.2
SHIFT_MAX_SEC  = 0.5

# ─── Audio Extraction ─────────────────────────────────────────────────────────

def extract_audio_features(video_path: str):
    """
    Extract 13 MFCCs + 13 Delta-MFCCs from a video file.
    Uses MoviePy to safely extract audio to bypass Windows FFmpeg issues.
    """
    temp_audio_path = f"{video_path}_temp.wav"
    
    try:
        # 1. Safely extract audio using MoviePy
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                return None
            clip.audio.write_audiofile(temp_audio_path)
            
        # 2. Load the pure audio file with Librosa
        y, sr = librosa.load(temp_audio_path, sr=SR, mono=True)
        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        deltas = librosa.feature.delta(mfccs)
        features = np.vstack([mfccs, deltas]).T
        
        # 3. Cleanup temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return features
        
    except Exception as e:
        print(f"[Audio] Error on {video_path}: {e}")
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        return None

def shift_audio_features(features: np.ndarray, shift_sec: float) -> np.ndarray:
    """
    Temporal-shift technique: offset audio feature rows by `shift_sec` seconds.
    Rows that would fall outside the array are zero-padded.
    """
    shift_frames = int(shift_sec * FPS)
    if shift_frames == 0:
        return features
    shifted = np.zeros_like(features)
    if shift_frames > 0:
        shifted[shift_frames:] = features[:-shift_frames]
    else:
        shifted[:shift_frames] = features[-shift_frames:]
    return shifted

# ─── Video / Mouth-crop Extraction ────────────────────────────────────────────

def _get_mouth_roi(frame: np.ndarray):
    """
    Locate the mouth region in a BGR frame using Haar cascade.
    Returns a (64, 64) grayscale crop or None if no face detected.
    """
    face_cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml")

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        # Fallback: use the bottom-centre of the frame as an estimate
        h, w = frame.shape[:2]
        roi = gray[int(h * 0.55):int(h * 0.85),
                   int(w * 0.25):int(w * 0.75)]
        return cv2.resize(roi, MOUTH_SIZE) if roi.size > 0 else None

    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    # Crop the lower third of the face (mouth region)
    mouth_y1 = y + int(fh * 0.55)
    mouth_y2 = y + fh
    mouth_roi = gray[mouth_y1:mouth_y2, x:x + fw]
    if mouth_roi.size == 0:
        return None
    return cv2.resize(mouth_roi, MOUTH_SIZE)


def extract_mouth_frames(video_path: str):
    """
    Extract mouth crops for every frame in a video.

    Returns
    -------
    np.ndarray of shape (n_frames, 64, 64) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Video] Cannot open {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = _get_mouth_roi(frame)
        if roi is not None:
            frames.append(roi.astype(np.float32) / 255.0)
        else:
            # Keep a blank frame to preserve alignment
            frames.append(np.zeros(MOUTH_SIZE, dtype=np.float32))

    cap.release()
    return np.array(frames) if frames else None

# ─── Feature Vector Builder ───────────────────────────────────────────────────

def build_paired_samples(mouth_frames: np.ndarray,
                          audio_features: np.ndarray) -> np.ndarray:
    """
    Concatenate (mouth crop) + (MFCCs) + (Deltas) per frame.

    Aligns the two sequences by length (takes min) and returns an array
    of shape (n_aligned_frames, INPUT_DIM = 4122).
    """
    n = min(len(mouth_frames), len(audio_features))
    mouth_flat = mouth_frames[:n].reshape(n, -1)    # (n, 4096)
    audio_feat = audio_features[:n]                 # (n, 26)
    return np.concatenate([mouth_flat, audio_feat], axis=1)   # (n, 4122)

# ─── Dynamic Pseudo-Fake Generator ────────────────────────────────────────────

def generate_training_samples(video_paths: list):
    """
    For every real video in `video_paths` produce:
      • 1 real sample  (label = 1)
      • 1 temporal-shift fake  (label = 0)  — audio shifted 0.2–0.5 s
      • 1 cross-modal swap fake  (label = 0) — audio from a random other video

    Returns
    -------
    X : np.ndarray (n_samples, INPUT_DIM)
    y : np.ndarray (n_samples,)
    """
    X_all, y_all = [], []

    # Pre-extract audio features for all videos so swaps can be drawn quickly
    print("[Dataset] Extracting features from seed videos …")
    mouth_map = {}
    audio_map = {}
    for vp in video_paths:
        m = extract_mouth_frames(vp)
        a = extract_audio_features(vp)
        if m is not None and a is not None:
            mouth_map[vp] = m
            audio_map[vp] = a

    valid_paths = list(mouth_map.keys())
    print(f"[Dataset] {len(valid_paths)} usable videos out of {len(video_paths)}")

    for vp in valid_paths:
        m = mouth_map[vp]
        a = audio_map[vp]

        # ── Real sample ──────────────────────────────────────────────────
        real_samples = build_paired_samples(m, a)
        X_all.append(real_samples)
        y_all.append(np.ones(len(real_samples)))

        # ── Technique 1: Temporal Shift ──────────────────────────────────
        shift_sec   = random.uniform(SHIFT_MIN_SEC, SHIFT_MAX_SEC)
        a_shifted   = shift_audio_features(a, shift_sec)
        fake_t1     = build_paired_samples(m, a_shifted)
        X_all.append(fake_t1)
        y_all.append(np.zeros(len(fake_t1)))

        # ── Technique 2: Cross-Modal Swap ────────────────────────────────
        donor = random.choice([p for p in valid_paths if p != vp])
        a_swapped = audio_map[donor]
        fake_t2   = build_paired_samples(m, a_swapped)
        X_all.append(fake_t2)
        y_all.append(np.zeros(len(fake_t2)))

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"[Dataset] Total samples: {len(X)}  "
          f"(real={int(y.sum())}, fake={int((1-y).sum())})")
    return X, y

# ─── Model Definition (PyTorch) ───────────────────────────────────────────────

class AudioVisualSyncNet(nn.Module):
    """
    Shallow 3-layer neural network:
        Linear(256, relu) → Dropout(0.4)
        Linear(64,  relu) → Dropout(0.3)
        Linear(1,  sigmoid)

    Kept intentionally shallow to avoid memorising the tiny seed dataset.
    L2 weight decay and Dropout force the model to learn generalisable features.
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(input_dim: int = INPUT_DIM) -> AudioVisualSyncNet:
    model = AudioVisualSyncNet(input_dim).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] AudioVisualSyncNet — {total_params:,} parameters  (device: {DEVICE})")
    return model

# ─── Public API ───────────────────────────────────────────────────────────────

def train_on_seed_data(video_folder: str,
                       epochs: int = 30,
                       batch_size: int = 64,
                       val_split: float = 0.15) -> AudioVisualSyncNet:
    """
    Train the sync detector on all .mp4 / .avi / .mov files found in
    `video_folder`.  Saves the trained model and feature scaler to disk.

    Parameters
    ----------
    video_folder : str
        Directory of real videos (50-100 recommended for the hackathon).
    epochs       : int  — training epochs (early-stop may terminate sooner).
    batch_size   : int
    val_split    : float — fraction held out for validation.

    Returns
    -------
    Trained AudioVisualSyncNet
    """
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    video_paths = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not video_paths:
        raise FileNotFoundError(f"No video files found in '{video_folder}'")

    X, y = generate_training_samples(video_paths)

    # Normalise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Train] Scaler saved → {SCALER_PATH}")

    # Build PyTorch datasets
    X_tensor = torch.from_numpy(X_scaled)
    y_tensor  = torch.from_numpy(y.astype(np.float32))
    dataset   = TensorDataset(X_tensor, y_tensor)

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = build_model()
    criterion = nn.BCELoss()
    # weight_decay acts as L2 regularisation (λ = 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6)

    # ── Early-stopping state ──────────────────────────────────────────────────
    best_val_auc   = 0.0
    patience_count = 0
    patience_limit = 6
    best_state     = None

    def _binary_auc(labels, preds):
        """Simple AUC approximation using sklearn if available."""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(labels, preds)
        except Exception:
            return 0.0

    for epoch in range(1, epochs + 1):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= train_size

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds   = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        val_loss /= val_size
        val_auc   = _binary_auc(all_labels, all_preds)

        scheduler.step(val_loss)

        print(f"[Train] Epoch {epoch:03d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"[Train] Early stopping at epoch {epoch} "
                      f"(best val AUC: {best_val_auc:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state": model.state_dict(),
                "input_dim":   INPUT_DIM}, MODEL_PATH)
    print(f"[Train] Model saved → {MODEL_PATH}")
    print(f"[Train] Best val AUC: {best_val_auc:.4f}")
    return model


def analyze_single_file(video_path: str,
                         model: Optional[AudioVisualSyncNet] = None,
                         scaler: Optional[StandardScaler] = None,
                         verbose: bool = True) -> float:
    """
    Compute a Sync Score in [0.0, 1.0] for a single video.

    Score → 1.0  means the audio-visual sync looks REAL.
    Score → 0.0  means the sync pattern looks FAKE / mismatched.

    Loads a saved model from disk if `model` is not provided.

    Parameters
    ----------
    video_path : str — path to an .mp4 / .avi / .mov file
    model      : optional pre-loaded AudioVisualSyncNet
    scaler     : optional pre-loaded StandardScaler
    verbose    : print the result to stdout

    Returns
    -------
    float — mean sync score across all frames
    """
    # Load model/scaler if not provided
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. "
                "Run train_on_seed_data() first.")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model = AudioVisualSyncNet(checkpoint.get("input_dim", INPUT_DIM)).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])

    if scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"No scaler found at '{SCALER_PATH}'. "
                "Run train_on_seed_data() first.")
        scaler = joblib.load(SCALER_PATH)

    # Feature extraction
    mouth_frames   = extract_mouth_frames(video_path)
    audio_features = extract_audio_features(video_path)

    if mouth_frames is None or audio_features is None:
        print(f"[Analyze] Could not extract features from {video_path}")
        return 0.5   # Return uncertain score on failure

    X = build_paired_samples(mouth_frames, audio_features)
    X_scaled = scaler.transform(X).astype(np.float32)
    X_tensor  = torch.from_numpy(X_scaled).to(DEVICE)

    # Per-frame scores, then average
    model.eval()
    with torch.no_grad():
        frame_scores = model(X_tensor).cpu().numpy().flatten()

    sync_score = float(np.mean(frame_scores))

    if verbose:
        verdict    = "REAL" if sync_score > 0.5 else "FAKE"
        confidence = abs(sync_score - 0.5) * 200   # 0-100%
        print(f"\n{'─'*50}")
        print(f"  File        : {os.path.basename(video_path)}")
        print(f"  Sync Score  : {sync_score:.4f}   ({verdict})")
        print(f"  Confidence  : {confidence:.1f}%")
        print(f"  Frames used : {len(frame_scores)}")
        print(f"{'─'*50}\n")

    return sync_score

# ─── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "train":
        train_on_seed_data(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "analyze":
        score = analyze_single_file(sys.argv[2])
        sys.exit(0 if score > 0.5 else 1)

    else:
        print(__doc__)
        print("CLI usage:")
        print("  python deepfake_detector.py train  <video_folder>")
        print("  python deepfake_detector.py analyze <video_file>")