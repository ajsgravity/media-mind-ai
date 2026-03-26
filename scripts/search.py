"""
search.py — Text, image, and tag-based search engine.
Uses CLIP embeddings + FAISS for semantic search.
"""

import os
import pickle
import numpy as np

from config import SEARCH_TOP_K, SIMILARITY_THRESHOLD, IMAGE_METADATA_PATH
from scripts.embeddings import (
    embed_text, embed_images, load_faiss_index, search_index
)


def text_search(query, top_k=None):
    """
    Search for images matching a natural language query.

    Args:
        query: text string (e.g., "photos of friends at the beach")
        top_k: number of results to return

    Returns:
        list of (path, score) tuples, sorted by relevance
    """
    if top_k is None:
        top_k = SEARCH_TOP_K

    query_embedding = embed_text(query)
    index, mapping = load_faiss_index()
    results = search_index(index, mapping, query_embedding, top_k=top_k)
    return results


def image_search(image_path, top_k=None):
    """
    Find images similar to a given image.

    Args:
        image_path: path to the query image
        top_k: number of results to return

    Returns:
        list of (path, score) tuples, sorted by similarity
    """
    if top_k is None:
        top_k = SEARCH_TOP_K + 1  # +1 because the image itself will match

    embeddings, valid_paths = embed_images([image_path])
    if embeddings.shape[0] == 0:
        return []

    index, mapping = load_faiss_index()
    results = search_index(index, mapping, embeddings, top_k=top_k)

    # Remove the query image itself from results
    norm_query = os.path.normpath(image_path)
    results = [(p, s) for p, s in results if os.path.normpath(p) != norm_query]

    return results[:SEARCH_TOP_K]


def tag_search(tag_query, image_metadata=None):
    """
    Search for images by YOLO-detected tags.

    Args:
        tag_query: tag string or list of tags
        image_metadata: pre-loaded metadata dict (optional)

    Returns:
        list of image paths containing the tag(s)
    """
    if image_metadata is None:
        if os.path.exists(IMAGE_METADATA_PATH):
            with open(IMAGE_METADATA_PATH, "rb") as f:
                image_metadata = pickle.load(f)
        else:
            return []

    if isinstance(tag_query, str):
        tags_to_find = {tag_query.lower()}
    else:
        tags_to_find = {t.lower() for t in tag_query}

    results = []
    for path, meta in image_metadata.items():
        image_tags = {t.lower() for t in meta.get("tags", [])}
        if tags_to_find & image_tags:  # intersection
            results.append(path)

    return results


def find_duplicates(threshold=None):
    """
    Find groups of near-duplicate images using embedding similarity.

    Returns:
        list of lists, each inner list is a group of duplicate paths
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    index, mapping = load_faiss_index()
    if index.ntotal == 0:
        return []

    n = index.ntotal
    # Search each vector against the index
    # We do this in batches to avoid memory issues
    duplicates_map = {}  # path -> group_id
    groups = []
    group_id = 0

    batch_size = 100
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Reconstruct vectors for this batch
        vectors = np.zeros((end - start, index.d), dtype="float32")
        for i in range(start, end):
            vectors[i - start] = index.reconstruct(i)

        scores, indices = index.search(vectors, min(10, n))

        for i in range(end - start):
            path_i = mapping[start + i]
            if path_i in duplicates_map:
                continue

            dups = []
            for j in range(len(indices[i])):
                idx_j = int(indices[i][j])
                score_j = float(scores[i][j])
                if idx_j == start + i:
                    continue
                if score_j >= threshold and idx_j < len(mapping):
                    path_j = mapping[idx_j]
                    if path_j not in duplicates_map:
                        dups.append(path_j)

            if dups:
                group = [path_i] + dups
                for p in group:
                    duplicates_map[p] = group_id
                groups.append(group)
                group_id += 1

    print(f"[SEARCH] Found {len(groups)} duplicate groups.")
    return groups


def get_all_tags(image_metadata=None):
    """
    Get all unique tags from the metadata.

    Returns:
        sorted list of tag strings
    """
    if image_metadata is None:
        if os.path.exists(IMAGE_METADATA_PATH):
            with open(IMAGE_METADATA_PATH, "rb") as f:
                image_metadata = pickle.load(f)
        else:
            return []

    tags = set()
    for meta in image_metadata.values():
        tags.update(t.lower() for t in meta.get("tags", []))
    return sorted(tags)
