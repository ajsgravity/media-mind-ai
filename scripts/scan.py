"""
scan.py — Recursive media scanner with incremental tracking.
Finds new images and videos that haven't been processed yet.
"""

import os
import pickle
from config import (
    MEDIA_DIRS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, PROCESSED_FILES_PATH
)


def load_processed_files():
    """Load set of already-processed file paths."""
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "rb") as f:
            return pickle.load(f)
    return set()


def save_processed_files(processed: set):
    """Persist the set of processed file paths."""
    with open(PROCESSED_FILES_PATH, "wb") as f:
        pickle.dump(processed, f)


def scan_media(media_dirs=None, incremental=True):
    """
    Recursively scan directories for media files.

    Args:
        media_dirs: list of directory paths to scan (defaults to config)
        incremental: if True, skip already-processed files

    Returns:
        images: list of new image file paths
        videos: list of new video file paths
    """
    if media_dirs is None:
        media_dirs = MEDIA_DIRS

    processed = load_processed_files() if incremental else set()

    images = []
    videos = []

    for root_dir in media_dirs:
        if not os.path.isdir(root_dir):
            print(f"[SCAN] Warning: directory not found: {root_dir}")
            continue

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                fpath = os.path.normpath(os.path.join(dirpath, fname))
                if fpath in processed:
                    continue

                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    images.append(fpath)
                elif ext in VIDEO_EXTENSIONS:
                    videos.append(fpath)

    print(f"[SCAN] Found {len(images)} new images, {len(videos)} new videos")
    return images, videos


if __name__ == "__main__":
    imgs, vids = scan_media()
    for p in imgs[:10]:
        print(f"  IMG: {p}")
    for p in vids[:10]:
        print(f"  VID: {p}")
