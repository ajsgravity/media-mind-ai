"""
embeddings.py — CLIP embedding generation and FAISS index management.
Supports incremental indexing (append without rebuild).
"""

import os
import pickle
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

from config import (
    CLIP_MODEL_NAME, CLIP_EMBEDDING_DIM,
    FAISS_INDEX_PATH, INDEX_MAPPING_PATH
)

# ─── Globals (lazy-loaded) ────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None
_clip_tokenizer = None
_device = None


def _load_clip():
    """Lazy-load CLIP model and processor."""
    global _clip_model, _clip_processor, _clip_tokenizer, _device
    if _clip_model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] Loading model on {_device}...")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
        print("[CLIP] Model loaded.")
    return _clip_model, _clip_processor, _clip_tokenizer, _device


def embed_images(image_paths, batch_size=32):
    """
    Generate CLIP embeddings for a list of image paths.

    Returns:
        embeddings: numpy array of shape (N, CLIP_EMBEDDING_DIM)
        valid_paths: list of paths that were successfully processed
    """
    model, processor, _, device = _load_clip()
    all_embeddings = []
    valid_paths = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                paths.append(p)
            except Exception as e:
                print(f"[CLIP] Skipping {p}: {e}")
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # transformers v5 may return a structured object instead of tensor
            if hasattr(outputs, 'pooler_output'):
                feats = outputs.pooler_output
            elif hasattr(outputs, 'image_embeds'):
                feats = outputs.image_embeds
            elif isinstance(outputs, torch.Tensor):
                feats = outputs
            else:
                # Last resort: try forward and extract image_embeds
                dummy_text = processor(text=[""], return_tensors="pt", padding=True).to(device)
                full_out = model(**{**inputs, **dummy_text})
                feats = full_out.image_embeds
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
        all_embeddings.append(feats.cpu().numpy())
        valid_paths.extend(paths)

        if (i // batch_size) % 10 == 0:
            print(f"[CLIP] Embedded {len(valid_paths)}/{len(image_paths)} images...")

    if all_embeddings:
        embeddings = np.vstack(all_embeddings).astype("float32")
    else:
        embeddings = np.empty((0, CLIP_EMBEDDING_DIM), dtype="float32")

    print(f"[CLIP] Generated {embeddings.shape[0]} embeddings.")
    return embeddings, valid_paths


def embed_text(query_text):
    """
    Generate a CLIP embedding for a text query.

    Returns:
        numpy array of shape (1, CLIP_EMBEDDING_DIM)
    """
    model, _, tokenizer, device = _load_clip()
    inputs = tokenizer([query_text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            feats = outputs.pooler_output
        elif hasattr(outputs, 'text_embeds'):
            feats = outputs.text_embeds
        elif isinstance(outputs, torch.Tensor):
            feats = outputs
        else:
            dummy_img = processor(images=[Image.new('RGB', (64, 64))], return_tensors="pt").to(device)
            full_out = model(**{**inputs, **dummy_img})
            feats = full_out.text_embeds
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


# ─── FAISS Index Management ───────────────────────────────────────────────────

def load_faiss_index():
    """Load existing FAISS index and mapping, or create new ones."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(INDEX_MAPPING_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(INDEX_MAPPING_PATH, "rb") as f:
            mapping = pickle.load(f)
        print(f"[FAISS] Loaded index with {index.ntotal} vectors.")
        return index, mapping
    else:
        index = faiss.IndexFlatIP(CLIP_EMBEDDING_DIM)  # inner product (cosine on normalized)
        mapping = []
        print("[FAISS] Created new empty index.")
        return index, mapping


def save_faiss_index(index, mapping):
    """Save FAISS index and mapping to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(INDEX_MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)
    print(f"[FAISS] Saved index with {index.ntotal} vectors.")


def add_to_index(index, mapping, embeddings, paths):
    """
    Append new embeddings to the FAISS index.

    Args:
        index: FAISS index
        mapping: list of file paths (parallel with FAISS rows)
        embeddings: numpy array (N, dim)
        paths: list of file paths corresponding to embeddings
    """
    if embeddings.shape[0] == 0:
        return index, mapping
    index.add(embeddings)
    mapping.extend(paths)
    return index, mapping


def search_index(index, mapping, query_embedding, top_k=20):
    """
    Search the FAISS index.

    Args:
        query_embedding: numpy array (1, dim)
        top_k: number of results

    Returns:
        list of (path, score) tuples
    """
    if index.ntotal == 0:
        return []
    scores, indices = index.search(query_embedding, min(top_k, index.ntotal))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(mapping) and idx >= 0:
            results.append((mapping[idx], float(score)))
    return results
