"""
app.py — Streamlit UI for the AI Media Intelligence System.
A mini Google Photos: search, browse, face albums, video moments, duplicates.
"""

import os
import sys
import base64
import pickle

import streamlit as st
import numpy as np
from PIL import Image

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    IMAGE_METADATA_PATH, FAISS_INDEX_PATH, INDEX_MAPPING_PATH,
    FACE_CLUSTERS_PATH, FACE_LABELS_PATH, SEARCH_TOP_K, SIMILARITY_THRESHOLD,
    BASE_DIR
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Media Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark premium theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #8892b0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    .image-card {
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    .image-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .tag-chip {
        display: inline-block;
        padding: 2px 10px;
        margin: 2px;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #ccd6f6 !important;
    }
    
    .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #ccd6f6 !important;
        padding: 0.7rem 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_data
def load_metadata():
    """Load image metadata from disk."""
    if os.path.exists(IMAGE_METADATA_PATH):
        with open(IMAGE_METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return {}


@st.cache_data
def load_index_mapping():
    """Load FAISS index mapping."""
    if os.path.exists(INDEX_MAPPING_PATH):
        with open(INDEX_MAPPING_PATH, "rb") as f:
            return pickle.load(f)
    return []


@st.cache_data
def load_clusters():
    """Load face clusters."""
    if os.path.exists(FACE_CLUSTERS_PATH):
        with open(FACE_CLUSTERS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def load_labels():
    """Load face labels (not cached — mutable)."""
    if os.path.exists(FACE_LABELS_PATH):
        with open(FACE_LABELS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_labels(labels):
    """Save face labels."""
    with open(FACE_LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)


def is_valid_image(path):
    """Check if an image path exists and is loadable."""
    if not os.path.exists(path):
        return False
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False


def display_image_grid(image_paths, scores=None, cols=4, show_info=True):
    """Display images in a responsive grid with optional scores and metadata."""
    metadata = load_metadata()
    
    rows = [image_paths[i:i+cols] for i in range(0, len(image_paths), cols)]
    score_rows = None
    if scores:
        score_rows = [scores[i:i+cols] for i in range(0, len(scores), cols)]
    
    for row_idx, row in enumerate(rows):
        columns = st.columns(cols)
        for col_idx, img_path in enumerate(row):
            with columns[col_idx]:
                if not os.path.exists(img_path):
                    st.warning(f"Not found")
                    continue
                
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    
                    # Score badge
                    if score_rows and col_idx < len(score_rows[row_idx]):
                        score = score_rows[row_idx][col_idx]
                        st.markdown(
                            f'<span class="score-badge">Score: {score:.3f}</span>',
                            unsafe_allow_html=True
                        )
                    
                    if show_info:
                        meta = metadata.get(os.path.normpath(img_path), {})
                        tags = meta.get("tags", [])
                        if tags:
                            tag_html = " ".join(
                                f'<span class="tag-chip">{t}</span>' for t in tags
                            )
                            st.markdown(tag_html, unsafe_allow_html=True)
                        
                        # Video source info
                        src = meta.get("source_video")
                        ts = meta.get("timestamp")
                        if src:
                            st.caption(f"📹 {os.path.basename(src)} @ {ts}s")
                    
                    # Similar images button
                    btn_key = f"similar_{img_path}_{row_idx}_{col_idx}"
                    if st.button("🔍 Find Similar", key=btn_key):
                        st.session_state["similar_query"] = img_path
                        st.session_state["page"] = "Similar Images"
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error: {e}")


# ─── Sidebar Navigation ──────────────────────────────────────────────────────

st.sidebar.markdown("## 🧠 AI Media Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Search", "👤 Face Albums", "🏷️ Browse Tags",
     "🎬 Video Moments", "📋 Duplicates", "📊 Dashboard"],
    index=0
)

# Override page if redirected
if "page" in st.session_state:
    redirect = st.session_state.pop("page")
    page_map = {
        "Similar Images": "🔍 Search",
    }
    page = page_map.get(redirect, page)


# ─── Dashboard Page ──────────────────────────────────────────────────────────

if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Overview of your indexed media</p>', unsafe_allow_html=True)
    
    metadata = load_metadata()
    mapping = load_index_mapping()
    clusters = load_clusters()
    
    # Metrics
    total_indexed = len(mapping)
    total_meta = len(metadata)
    n_people = len([k for k in clusters if k >= 0])
    n_faces_total = sum(len(v) for v in clusters.values())
    
    # Count unique tags
    all_tags = set()
    for m in metadata.values():
        all_tags.update(m.get("tags", []))
    
    # Count video frames vs images
    n_video_frames = sum(1 for m in metadata.values() if m.get("source_video"))
    n_images = total_meta - n_video_frames
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_indexed}</div>
            <div class="metric-label">Indexed Items</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_people}</div>
            <div class="metric-label">People Found</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(all_tags)}</div>
            <div class="metric-label">Unique Tags</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_video_frames}</div>
            <div class="metric-label">Video Frames</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not mapping:
        st.info("No media indexed yet. Run `python run_pipeline.py` to index your media.")


# ─── Search Page ──────────────────────────────────────────────────────────────

elif page == "🔍 Search":
    st.markdown('<h1 class="main-header">🔍 Intelligent Search</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Search your media with natural language</p>',
        unsafe_allow_html=True
    )
    
    # Check for similar image redirect
    similar_query = st.session_state.pop("similar_query", None)
    
    if similar_query:
        st.markdown(f"### 🔗 Images similar to:")
        if os.path.exists(similar_query):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(Image.open(similar_query), width=300)
        
        from scripts.search import image_search
        results = image_search(similar_query, top_k=SEARCH_TOP_K)
        
        if results:
            paths = [r[0] for r in results]
            scores = [r[1] for r in results]
            display_image_grid(paths, scores=scores)
        else:
            st.info("No similar images found.")
    else:
        query = st.text_input(
            "🔎 What are you looking for?",
            placeholder="e.g., 'friends at the beach', 'sunset', 'dog playing'..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Results to show", 4, 50, SEARCH_TOP_K)
        with col2:
            search_type = st.radio("Search type", ["Semantic (CLIP)", "Tag-based"], horizontal=True)
        
        if query:
            if search_type == "Semantic (CLIP)":
                from scripts.search import text_search
                with st.spinner("Searching with AI..."):
                    results = text_search(query, top_k=top_k)
                
                if results:
                    st.success(f"Found {len(results)} results for: **{query}**")
                    paths = [r[0] for r in results]
                    scores = [r[1] for r in results]
                    display_image_grid(paths, scores=scores)
                else:
                    st.warning("No results found. Try a different query or index more media.")
            else:
                from scripts.search import tag_search
                results = tag_search(query)
                if results:
                    st.success(f"Found {len(results)} images with tag: **{query}**")
                    display_image_grid(results[:top_k])
                else:
                    st.warning(f"No images found with tag '{query}'.")


# ─── Face Albums Page ─────────────────────────────────────────────────────────

elif page == "👤 Face Albums":
    st.markdown('<h1 class="main-header">👤 Face Albums</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">People detected and grouped automatically</p>',
        unsafe_allow_html=True
    )
    
    clusters = load_clusters()
    labels = load_labels()
    
    if not clusters:
        st.info("No face clusters found. Run the indexing pipeline first.")
    else:
        # Sidebar: label management
        with st.sidebar:
            st.markdown("### 🏷️ Label a Person")
            cluster_ids = sorted([k for k in clusters.keys() if k >= 0])
            if cluster_ids:
                selected_id = st.selectbox(
                    "Cluster to label",
                    cluster_ids,
                    format_func=lambda x: labels.get(x, f"Person {x + 1}")
                )
                new_label = st.text_input("Name", value=labels.get(selected_id, ""))
                if st.button("💾 Save Label"):
                    labels[selected_id] = new_label
                    save_labels(labels)
                    st.success(f"Labeled as '{new_label}'")
                    st.rerun()
        
        # Display each cluster
        for cid in sorted(clusters.keys()):
            if cid == -1:
                name = "❓ Unknown Faces"
            else:
                name = labels.get(cid, f"Person {cid + 1}")
            
            faces = clusters[cid]
            unique_paths = list(dict.fromkeys(f["path"] for f in faces))
            
            with st.expander(f"**{name}** — {len(unique_paths)} photos", expanded=(cid >= 0)):
                display_image_grid(unique_paths[:20], cols=5, show_info=False)


# ─── Browse Tags Page ─────────────────────────────────────────────────────────

elif page == "🏷️ Browse Tags":
    st.markdown('<h1 class="main-header">🏷️ Browse by Tags</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Objects detected by YOLOv8</p>',
        unsafe_allow_html=True
    )
    
    from scripts.search import get_all_tags, tag_search
    
    tags = get_all_tags()
    
    if not tags:
        st.info("No tags found. Run the indexing pipeline first.")
    else:
        # Tag cloud
        tag_html = " ".join(f'<span class="tag-chip">{t}</span>' for t in tags)
        st.markdown(tag_html, unsafe_allow_html=True)
        st.markdown("")
        
        selected_tag = st.selectbox("Select a tag to browse", [""] + tags)
        
        if selected_tag:
            results = tag_search(selected_tag)
            st.success(f"Found {len(results)} images with tag: **{selected_tag}**")
            display_image_grid(results[:40])


# ─── Video Moments Page ──────────────────────────────────────────────────────

elif page == "🎬 Video Moments":
    st.markdown('<h1 class="main-header">🎬 Video Moments</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Search within your videos</p>',
        unsafe_allow_html=True
    )
    
    metadata = load_metadata()
    
    # Get unique videos
    video_frames = {}
    for path, meta in metadata.items():
        src = meta.get("source_video")
        if src:
            if src not in video_frames:
                video_frames[src] = []
            video_frames[src].append({
                "frame_path": path,
                "timestamp": meta.get("timestamp", 0)
            })
    
    if not video_frames:
        st.info("No video frames indexed yet. Add videos to your media directory and run the pipeline.")
    else:
        # Video search
        video_query = st.text_input(
            "🔎 Search within videos",
            placeholder="e.g., 'celebration', 'outdoor scene'..."
        )
        
        if video_query:
            from scripts.search import text_search
            with st.spinner("Searching video frames..."):
                results = text_search(video_query, top_k=50)
            
            # Filter to only video frames
            video_results = []
            for path, score in results:
                meta = metadata.get(os.path.normpath(path), {})
                if meta.get("source_video"):
                    video_results.append((path, score, meta))
            
            if video_results:
                st.success(f"Found {len(video_results)} video moments")
                for path, score, meta in video_results[:20]:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if os.path.exists(path):
                            st.image(Image.open(path), use_container_width=True)
                    with col2:
                        src_video = meta["source_video"]
                        ts = meta.get("timestamp", 0)
                        st.markdown(f"**📹 {os.path.basename(src_video)}**")
                        st.markdown(f"⏱️ Timestamp: **{ts}s** ({int(ts//60)}:{int(ts%60):02d})")
                        st.markdown(f'<span class="score-badge">Relevance: {score:.3f}</span>',
                                   unsafe_allow_html=True)
                        
                        # Play video from timestamp
                        if os.path.exists(src_video):
                            if st.button(f"▶️ Play from {ts}s", key=f"play_{path}"):
                                st.video(src_video, start_time=int(ts))
                    st.markdown("---")
            else:
                st.warning("No matching video moments found.")
        else:
            # Browse all videos
            st.markdown("### 📂 Indexed Videos")
            for vid_path, frames in video_frames.items():
                frames_sorted = sorted(frames, key=lambda x: x["timestamp"])
                with st.expander(f"📹 {os.path.basename(vid_path)} — {len(frames_sorted)} frames"):
                    # Show sample frames
                    sample_paths = [f["frame_path"] for f in frames_sorted[:12]]
                    display_image_grid(sample_paths, cols=4, show_info=False)
                    
                    if os.path.exists(vid_path):
                        st.video(vid_path)


# ─── Duplicates Page ──────────────────────────────────────────────────────────

elif page == "📋 Duplicates":
    st.markdown('<h1 class="main-header">📋 Duplicate Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Find near-identical images in your collection</p>',
        unsafe_allow_html=True
    )
    
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.85, max_value=1.0, value=SIMILARITY_THRESHOLD,
        step=0.01, help="Higher = stricter matching"
    )
    
    if st.button("🔍 Find Duplicates"):
        from scripts.search import find_duplicates
        with st.spinner("Scanning for duplicates..."):
            groups = find_duplicates(threshold=threshold)
        
        if groups:
            st.warning(f"Found {len(groups)} groups of potential duplicates")
            for i, group in enumerate(groups):
                with st.expander(f"Group {i+1} — {len(group)} images"):
                    display_image_grid(group, cols=len(group), show_info=False)
        else:
            st.success("No duplicates found! 🎉")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="color: #4a5568; font-size: 0.75rem; text-align: center;">'
    'AI Media Intelligence v1.0<br>Powered by CLIP • YOLOv8 • FAISS</p>',
    unsafe_allow_html=True
)
