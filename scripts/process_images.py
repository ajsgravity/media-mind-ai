"""
process_images.py — YOLO object detection and face detection via MediaPipe.
Produces per-image metadata (tags, face encodings, locations).
Uses mediapipe for face detection (no dlib/CMake needed).
"""

import os
import pickle
import numpy as np
import cv2
from PIL import Image

from config import (
    YOLO_MODEL_NAME, IMAGE_METADATA_PATH,
)

# ─── Lazy-loaded models ───────────────────────────────────────────────────────
_yolo_model = None
_face_detector = None
_face_embedder = None


def _load_yolo():
    """Lazy-load YOLOv8 model."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("[YOLO] Loading model...")
        _yolo_model = YOLO(YOLO_MODEL_NAME)
        print("[YOLO] Model loaded.")
    return _yolo_model


def _load_face_detector():
    """Lazy-load MediaPipe face detector."""
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0=short-range, 1=full-range
            min_detection_confidence=0.5
        )
        print("[FACE] MediaPipe face detector loaded.")
    return _face_detector


def _load_face_embedder():
    """Lazy-load a simple face embedding model using insightface."""
    global _face_embedder
    if _face_embedder is None:
        try:
            from insightface.app import FaceAnalysis
            _face_embedder = FaceAnalysis(
                name="buffalo_s",
                providers=["CPUExecutionProvider"]
            )
            _face_embedder.prepare(ctx_id=-1, det_size=(640, 640))
            print("[FACE] InsightFace embedder loaded.")
        except Exception as e:
            print(f"[FACE] InsightFace not available ({e}), using MediaPipe-only mode.")
            _face_embedder = "mediapipe_only"
    return _face_embedder


def detect_objects(image_path):
    """
    Run YOLOv8 on an image and return detected class names.

    Returns:
        list of tag strings (e.g., ["person", "dog", "car"])
    """
    model = _load_yolo()
    try:
        results = model(image_path, verbose=False)
        tags = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                tag = model.names[cls_id]
                tags.add(tag)
        return list(tags)
    except Exception as e:
        print(f"[YOLO] Error on {image_path}: {e}")
        return []


def detect_faces(image_path):
    """
    Detect faces and compute embeddings.
    Uses MediaPipe for detection, InsightFace for 512-d embeddings (if available).

    Returns:
        face_encodings: list of numpy arrays (512-d each from InsightFace, or empty)
        face_locations: list of (top, right, bottom, left) tuples
    """
    face_encodings = []
    face_locations = []

    # Try InsightFace first (gives both detection and embeddings)
    embedder = _load_face_embedder()
    if embedder != "mediapipe_only":
        try:
            img = cv2.imread(image_path)
            if img is None:
                return [], []
            faces = embedder.get(img)
            for face in faces:
                bbox = face.bbox.astype(int)
                top, right, bottom, left = bbox[1], bbox[2], bbox[3], bbox[0]
                face_locations.append((top, right, bottom, left))
                face_encodings.append(face.embedding)
            return face_encodings, face_locations
        except Exception as e:
            print(f"[FACE] InsightFace error on {image_path}: {e}")

    # Fallback: MediaPipe detection only (no embeddings)
    detector = _load_face_detector()
    try:
        img = cv2.imread(image_path)
        if img is None:
            return [], []
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(img_rgb)

        if results.detections:
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                top = int(bb.ymin * h)
                left = int(bb.xmin * w)
                bottom = int((bb.ymin + bb.height) * h)
                right = int((bb.xmin + bb.width) * w)
                face_locations.append((top, right, bottom, left))
                # Create a simple embedding from the cropped face (pixel-based hash)
                face_crop = img_rgb[max(0, top):bottom, max(0, left):right]
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (64, 64)).flatten().astype(np.float32)
                    face_resized = face_resized / (np.linalg.norm(face_resized) + 1e-8)
                    face_encodings.append(face_resized)
    except Exception as e:
        print(f"[FACE] MediaPipe error on {image_path}: {e}")

    return face_encodings, face_locations


def process_single_image(image_path, source_video=None, timestamp=None):
    """
    Process one image: YOLO tags + face detection.

    Returns:
        metadata dict for this image
    """
    tags = detect_objects(image_path)
    face_encodings, face_locations = detect_faces(image_path)

    meta = {
        "path": os.path.normpath(image_path),
        "tags": tags,
        "face_encodings": [enc.tolist() if isinstance(enc, np.ndarray) else enc
                           for enc in face_encodings],
        "face_locations": face_locations,
        "source_video": source_video,
        "timestamp": timestamp,
    }
    return meta


def load_image_metadata():
    """Load existing image metadata from disk."""
    if os.path.exists(IMAGE_METADATA_PATH):
        with open(IMAGE_METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_image_metadata(metadata):
    """Save image metadata dict to disk."""
    with open(IMAGE_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


def process_images(image_paths, video_frame_info=None):
    """
    Process a list of images through YOLO + face detection.

    Args:
        image_paths: list of image file paths
        video_frame_info: optional dict mapping frame_path -> (source_video, timestamp)

    Returns:
        metadata: dict mapping file_path -> metadata dict
    """
    if video_frame_info is None:
        video_frame_info = {}

    existing_meta = load_image_metadata()
    count = 0

    for path in image_paths:
        norm_path = os.path.normpath(path)
        if norm_path in existing_meta:
            continue

        src_video, ts = video_frame_info.get(norm_path, (None, None))
        meta = process_single_image(path, source_video=src_video, timestamp=ts)
        existing_meta[norm_path] = meta
        count += 1

        if count % 50 == 0:
            print(f"[PROCESS] Processed {count}/{len(image_paths)} images...")
            save_image_metadata(existing_meta)  # checkpoint

    save_image_metadata(existing_meta)
    print(f"[PROCESS] Done. Processed {count} new images. Total: {len(existing_meta)}")
    return existing_meta
