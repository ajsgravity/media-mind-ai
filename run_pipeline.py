"""
run_pipeline.py — Orchestrator for the full indexing pipeline.
Usage:
    python run_pipeline.py                           # incremental index
    python run_pipeline.py --media-dirs ./Photos     # custom dirs
    python run_pipeline.py --rebuild                 # full rebuild
"""

import argparse
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MEDIA_DIRS, PROCESSED_FILES_PATH
from scripts.scan import scan_media, load_processed_files, save_processed_files
from scripts.process_videos import process_all_videos
from scripts.embeddings import (
    embed_images, load_faiss_index, save_faiss_index, add_to_index
)
from scripts.process_images import process_images
from scripts.face_clustering import cluster_faces


def run_pipeline(media_dirs=None, rebuild=False):
    """
    Run the full indexing pipeline.

    Steps:
        1. Scan for new media files
        2. Extract frames from videos
        3. Generate CLIP embeddings + update FAISS index
        4. Run YOLO + face detection
        5. Cluster faces
    """
    start_time = time.time()
    print("=" * 60)
    print("  AI Media Intelligence — Indexing Pipeline")
    print("=" * 60)

    # ── Step 1: Scan ──────────────────────────────────────────────────
    print("\n[1/5] Scanning for media files...")
    if rebuild:
        # Remove processed files tracker to force full re-scan
        if os.path.exists(PROCESSED_FILES_PATH):
            os.remove(PROCESSED_FILES_PATH)

    images, videos = scan_media(media_dirs=media_dirs, incremental=not rebuild)

    if not images and not videos:
        print("\nNo new media files found. Index is up to date!")
        return

    # ── Step 2: Extract video frames ──────────────────────────────────
    print("\n[2/5] Extracting video frames...")
    video_frame_info = {}  # frame_path -> (source_video, timestamp)
    frame_paths = []

    if videos:
        all_frames = process_all_videos(videos)
        for frame_path, src_video, timestamp in all_frames:
            frame_paths.append(frame_path)
            video_frame_info[frame_path] = (src_video, timestamp)
    else:
        print("  No new videos to process.")

    # Combine image paths: original images + extracted frames
    all_image_paths = images + frame_paths
    print(f"  Total images to process: {len(all_image_paths)}")

    # ── Step 3: Generate embeddings + update FAISS ────────────────────
    print("\n[3/5] Generating CLIP embeddings...")
    embeddings, valid_paths = embed_images(all_image_paths)

    print("  Updating FAISS index...")
    index, mapping = load_faiss_index()
    index, mapping = add_to_index(index, mapping, embeddings, valid_paths)
    save_faiss_index(index, mapping)

    # ── Step 4: YOLO + Face detection ─────────────────────────────────
    print("\n[4/5] Running object detection & face recognition...")
    process_images(all_image_paths, video_frame_info=video_frame_info)

    # ── Step 5: Cluster faces ─────────────────────────────────────────
    print("\n[5/5] Clustering faces...")
    clusters = cluster_faces()

    # ── Mark files as processed ───────────────────────────────────────
    processed = load_processed_files()
    for p in images:
        processed.add(os.path.normpath(p))
    for p in videos:
        processed.add(os.path.normpath(p))
    # Also mark frames as processed so they aren't re-embedded
    for p in frame_paths:
        processed.add(os.path.normpath(p))
    save_processed_files(processed)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Images indexed: {len(valid_paths)}")
    print(f"  Total in FAISS: {index.ntotal}")
    n_people = len([k for k in clusters if k >= 0])
    print(f"  People detected: {n_people}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Media Intelligence Indexing Pipeline")
    parser.add_argument(
        "--media-dirs", nargs="+", default=None,
        help="Directories to scan for media (default: config.MEDIA_DIRS)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild the entire index from scratch"
    )
    args = parser.parse_args()

    run_pipeline(media_dirs=args.media_dirs, rebuild=args.rebuild)
