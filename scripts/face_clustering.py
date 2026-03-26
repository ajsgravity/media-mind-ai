"""
face_clustering.py — Cluster detected faces using DBSCAN.
Supports labeling individuals and incremental re-clustering.
"""

import os
import pickle
import numpy as np
from sklearn.cluster import DBSCAN

from config import (
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    FACE_CLUSTERS_PATH, FACE_LABELS_PATH, IMAGE_METADATA_PATH
)


def load_face_clusters():
    """Load existing face clusters from disk."""
    if os.path.exists(FACE_CLUSTERS_PATH):
        with open(FACE_CLUSTERS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_face_clusters(clusters):
    """Save face clusters to disk."""
    with open(FACE_CLUSTERS_PATH, "wb") as f:
        pickle.dump(clusters, f)


def load_face_labels():
    """Load cluster ID → name mapping."""
    if os.path.exists(FACE_LABELS_PATH):
        with open(FACE_LABELS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_face_labels(labels):
    """Save cluster labels to disk."""
    with open(FACE_LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)


def cluster_faces(image_metadata=None):
    """
    Cluster all face encodings from image metadata using DBSCAN.

    Returns:
        clusters: dict mapping cluster_id -> list of
                  {"path": ..., "face_location": ..., "face_idx": ...}
    """
    # Load metadata if not provided
    if image_metadata is None:
        if os.path.exists(IMAGE_METADATA_PATH):
            with open(IMAGE_METADATA_PATH, "rb") as f:
                image_metadata = pickle.load(f)
        else:
            print("[FACE-CLUSTER] No image metadata found.")
            return {}

    # Collect all face encodings with their source info
    all_encodings = []
    face_info = []  # parallel list: (image_path, face_idx, face_location)

    for path, meta in image_metadata.items():
        encodings = meta.get("face_encodings", [])
        locations = meta.get("face_locations", [])
        for idx, enc in enumerate(encodings):
            all_encodings.append(np.array(enc))
            loc = locations[idx] if idx < len(locations) else None
            face_info.append({
                "path": path,
                "face_idx": idx,
                "face_location": loc,
                "source_video": meta.get("source_video"),
                "timestamp": meta.get("timestamp"),
            })

    if len(all_encodings) < 2:
        print(f"[FACE-CLUSTER] Only {len(all_encodings)} face(s) found, skipping clustering.")
        if len(all_encodings) == 1:
            clusters = {0: [face_info[0]]}
            save_face_clusters(clusters)
            return clusters
        return {}

    # Run DBSCAN with cosine metric (normalize first for InsightFace embeddings)
    X = np.array(all_encodings, dtype=np.float64)
    # L2-normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    labels = clustering.fit_predict(X)

    # Group by cluster
    clusters = {}
    for label, info in zip(labels, face_info):
        cluster_id = int(label)  # -1 = noise
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(info)

    n_clusters = len([k for k in clusters if k >= 0])
    n_noise = len(clusters.get(-1, []))
    print(f"[FACE-CLUSTER] Found {n_clusters} people, {n_noise} unmatched faces.")

    save_face_clusters(clusters)
    return clusters


def label_cluster(cluster_id, name):
    """Assign a name to a face cluster."""
    labels = load_face_labels()
    labels[cluster_id] = name
    save_face_labels(labels)
    print(f"[FACE-CLUSTER] Cluster {cluster_id} labeled as '{name}'")


def get_labeled_clusters():
    """
    Return clusters with labels applied.

    Returns:
        dict: { display_name: [face_info_list] }
    """
    clusters = load_face_clusters()
    labels = load_face_labels()

    result = {}
    for cid, faces in clusters.items():
        if cid == -1:
            name = "Unknown"
        elif cid in labels:
            name = labels[cid]
        else:
            name = f"Person {cid + 1}"
        result[name] = {"cluster_id": cid, "faces": faces}

    return result


if __name__ == "__main__":
    clusters = cluster_faces()
    labels = load_face_labels()
    for cid, faces in clusters.items():
        name = labels.get(cid, f"Cluster {cid}")
        print(f"  {name}: {len(faces)} photos")
