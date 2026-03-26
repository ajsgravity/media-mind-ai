"""
process_videos.py — Extract frames from video files using OpenCV.
Supports constant FPS extraction and basic scene-change detection.
"""

import os
import cv2
import numpy as np
from config import VIDEO_FRAMES_DIR, VIDEO_FRAME_RATE, SCENE_DIFF_THRESHOLD


def extract_frames(video_path, fps=None, scene_detect=False):
    """
    Extract frames from a video file.

    Args:
        video_path: path to the video file
        fps: frames per second to extract (default from config)
        scene_detect: if True, also extract on scene changes

    Returns:
        list of (frame_path, source_video, timestamp_sec) tuples
    """
    if fps is None:
        fps = VIDEO_FRAME_RATE

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VIDEO] Error: cannot open {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback

    frame_interval = max(1, int(video_fps / fps))

    # Create output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(VIDEO_FRAMES_DIR, video_name)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    frame_idx = 0
    saved_count = 0
    prev_frame_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        should_save = (frame_idx % frame_interval == 0)

        # Scene-change detection
        if scene_detect and not should_save and prev_frame_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = np.mean(np.abs(gray.astype(float) - prev_frame_gray.astype(float)))
            if diff > SCENE_DIFF_THRESHOLD:
                should_save = True

        if should_save:
            timestamp = frame_idx / video_fps
            frame_filename = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(out_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            results.append((
                os.path.normpath(frame_path),
                os.path.normpath(video_path),
                round(timestamp, 2)
            ))
            saved_count += 1

        # Update prev frame for scene detection
        if scene_detect:
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_idx += 1

    cap.release()
    print(f"[VIDEO] Extracted {saved_count} frames from {os.path.basename(video_path)}")
    return results


def process_all_videos(video_paths, scene_detect=False):
    """
    Process a list of video files and extract frames.

    Returns:
        all_frames: list of (frame_path, source_video, timestamp) tuples
    """
    all_frames = []
    for vpath in video_paths:
        frames = extract_frames(vpath, scene_detect=scene_detect)
        all_frames.extend(frames)
    print(f"[VIDEO] Total frames extracted: {len(all_frames)}")
    return all_frames


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        frames = extract_frames(sys.argv[1])
        for fp, src, ts in frames[:5]:
            print(f"  {fp}  (from {src} @ {ts}s)")
