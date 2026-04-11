from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import shutil
import os
import sys
import tempfile
import random
import numpy as np
from pathlib import Path

# --- MEDIAPIPE TASKS API (v0.10+) ---
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_face_landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,  # VIDEO mode enables per-frame timestamp scanning
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)

# Add the project root (parent of Backend/) to the path so Python can find 'gaze'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gaze.gaze_physics import process_api_payload

# ── Import AV-Sync detector from sibling folder ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "avsync"))
try:
    from avsync.audio import analyze_single_file as _av_analyze
    _AV_AVAILABLE = True
except ImportError as e:
    print(f"[AV Sync] Import failed: {e}")
    _AV_AVAILABLE = False

# ------------------ APP SETUP ------------------
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ HELPERS ------------------
def get_av_sync_score(video_path: str) -> float:
    try:
        from avsync.audio import analyze_single_file
        return analyze_single_file(video_path, verbose=False)
    except Exception as e:
        print(f"AV Sync Error: {e}")
        return 0.5 

def save_upload_to_tempfile(file: UploadFile) -> str:
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(prefix="svvap_", suffix=suffix, delete=False) as temp:
        shutil.copyfileobj(file.file, temp)
        return temp.name

# ------------------ REAL MEDIA PIPE EXTRACTION ------------------
def extract_binocular_gaze(video_path: str) -> dict:
    # New mediapipe Tasks API (v0.10+): FaceLandmarker in VIDEO mode replaces FaceMesh
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Fallback to 30fps if unknown

    landmarks = None
    # Scan up to the first 30 frames to find a clear face
    with mp_vision.FaceLandmarker.create_from_options(_face_landmarker_options) as landmarker:
        for frame_idx in range(30):
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # VIDEO mode requires a monotonically increasing timestamp in milliseconds
            timestamp_ms = int((frame_idx / fps) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]  # List of NormalizedLandmark objects
                break  # Face found! Break the loop.

    cap.release()

    if not landmarks:
        raise ValueError("No face detected in the first 30 frames of the video.")

    # Iris landmark indices: 468 (left iris center), 473 (right iris center)
    lp = landmarks[468]
    rp = landmarks[473]

    # MediaPipe X/Y are normalized [0.0 to 1.0].
    # We subtract 0.5 to put the nose roughly at coordinate (0,0) for the 3D graph
    p_l = [lp.x - 0.5, lp.y - 0.5, lp.z * 5]  # Multiply Z to give the face some 3D depth
    p_r = [rp.x - 0.5, rp.y - 0.5, rp.z * 5]

    # HACKATHON SHORTCUT: Assume the person is looking roughly at the camera lens
    # The camera lens is sitting in front of them on the Z-axis.
    camera_lens = [0.0, 0.0, 10.0]

    # Calculate the mathematical Direction Vector (Destination - Origin)
    g_l = [camera_lens[0] - p_l[0], camera_lens[1] - p_l[1], camera_lens[2] - p_l[2]]
    g_r = [camera_lens[0] - p_r[0], camera_lens[1] - p_r[1], camera_lens[2] - p_r[2]]

    # Add a tiny bit of random noise to simulate natural human micro-saccades,
    # which prevents perfectly parallel lines from causing a division-by-zero in edge cases.
    g_l[0] += random.uniform(-0.005, 0.005)
    g_r[0] += random.uniform(-0.005, 0.005)

    return {
        "left_pupil": p_l,
        "right_pupil": p_r,
        "left_gaze": g_l,
        "right_gaze": g_r
    }
# ------------------ MERGED API ENDPOINT ------------------
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_file_path = None

    try:
        # 1. Save uploaded video
        temp_file_path = save_upload_to_tempfile(file)

        # 2. Extract Base Metrics
        cap = cv2.VideoCapture(temp_file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 3. RUN MEDIAPIPE EXTRACTION
        try:
            extracted_coords = extract_binocular_gaze(temp_file_path)
        except Exception as e:
            return {"status": "error", "message": f"Landmark extraction failed: {str(e)}"}

        # 4. RUN EXPERT A: The Geometric Physics Engine
        physics_results = process_api_payload(extracted_coords)

        # 5. RUN EXPERT B: The AV Sync Engine
        av_score = get_av_sync_score(temp_file_path)

        # 6. ENSEMBLE METAMODEL
        gaze_is_fake = physics_results.get("is_deepfake", False)
        final_is_deepfake = bool(gaze_is_fake) or (av_score < 0.7)

        # Helper: coerce any numpy scalar/array to native Python
        def _py(v):
            if v is None:
                return None
            if hasattr(v, 'tolist'):
                return v.tolist()
            if hasattr(v, 'item'):
                return v.item()
            return v

        return {
            "status": "success",
            "metadata": {
                "filename": file.filename,
                "frames": int(frame_count)
            },
            "analysis": {
                "is_deepfake":    bool(final_is_deepfake),
                "logical_gaze":   bool(physics_results.get("logical_gaze")),
                "distance":       float(_py(physics_results.get("distance", 0))),
                "vergence_point": _py(physics_results.get("vergence_point"))
            },
            "raw_coordinates": {
                k: _py(v) for k, v in extracted_coords.items()
            },
            "av_sync_data": {
                "score": float(av_score)
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)