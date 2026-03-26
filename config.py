"""
Central configuration for the AI Media Intelligence System.
All paths, model settings, and processing parameters live here.
"""

import os

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default media directories to scan (override via CLI)
MEDIA_DIRS = [
    os.path.join(BASE_DIR, "Photos"),
    os.path.join(BASE_DIR, "videos"),
]

# Storage directories
INDEX_DIR = os.path.join(BASE_DIR, "index")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
VIDEO_FRAMES_DIR = os.path.join(INDEX_DIR, "video_frames")

# ─── Supported Extensions ─────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

# ─── CLIP Model ───────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512

# ─── YOLO Model ───────────────────────────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n.pt"  # nano variant for speed

# ─── Face Detection (MediaPipe + InsightFace) ─────────────────────────────
INSIGHTFACE_MODEL = "buffalo_s"     # lightweight face analysis model
FACE_DETECTION_CONFIDENCE = 0.5     # min confidence for MediaPipe
FACE_DISTANCE_THRESHOLD = 0.6       # max distance to consider same person

# ─── Face Clustering (DBSCAN) ─────────────────────────────────────────────────
DBSCAN_EPS = 0.6
DBSCAN_MIN_SAMPLES = 2

# ─── Video Processing ─────────────────────────────────────────────────────────
VIDEO_FRAME_RATE = 1  # extract 1 frame per second
SCENE_DIFF_THRESHOLD = 30.0  # mean pixel diff to detect scene change

# ─── Search ────────────────────────────────────────────────────────────────────
SEARCH_TOP_K = 20
SIMILARITY_THRESHOLD = 0.95  # for duplicate detection

# ─── FAISS Index File ─────────────────────────────────────────────────────────
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
INDEX_MAPPING_PATH = os.path.join(METADATA_DIR, "index_mapping.pkl")

# ─── Metadata Files ───────────────────────────────────────────────────────────
PROCESSED_FILES_PATH = os.path.join(METADATA_DIR, "processed_files.pkl")
IMAGE_METADATA_PATH = os.path.join(METADATA_DIR, "image_metadata.pkl")
FACE_CLUSTERS_PATH = os.path.join(METADATA_DIR, "face_clusters.pkl")
FACE_LABELS_PATH = os.path.join(METADATA_DIR, "face_labels.pkl")

# ─── Ensure directories exist ─────────────────────────────────────────────────
for d in [INDEX_DIR, METADATA_DIR, VIDEO_FRAMES_DIR]:
    os.makedirs(d, exist_ok=True)
