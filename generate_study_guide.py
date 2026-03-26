"""
generate_study_guide.py — Generates a comprehensive PDF study guide for the
AI Media Intelligence System project.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fpdf import FPDF


class StudyGuidePDF(FPDF):
    """Custom PDF with header/footer styling."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "AI Media Intelligence System - Study Guide", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(40, 40, 120)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(40, 40, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(60, 60, 150)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_section(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, code):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        for line in code.strip().split("\n"):
            self.cell(0, 5, "  " + line, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(55, 6, key + ":")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    def qa_pair(self, question, answer):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 120)
        self.multi_cell(0, 5.5, "Q: " + question)
        self.ln(1)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, "A: " + answer)
        self.ln(4)


def build_pdf():
    pdf = StudyGuidePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ═══════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(40, 40, 120)
    pdf.cell(0, 15, "AI Media Intelligence System", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Complete Study Guide & Viva Reference", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, "A Local AI-Powered Google Photos Alternative", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Tech Stack: Python, CLIP, YOLOv8, FAISS, InsightFace, MediaPipe, Streamlit",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ═══════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    toc = [
        "1. Project Overview & Objectives",
        "2. System Architecture & Data Flow",
        "3. Tech Stack Deep Dive",
        "   3.1 CLIP (Contrastive Language-Image Pretraining)",
        "   3.2 YOLOv8 (Object Detection)",
        "   3.3 FAISS (Vector Similarity Search)",
        "   3.4 InsightFace & MediaPipe (Face Recognition)",
        "   3.5 DBSCAN (Face Clustering)",
        "   3.6 OpenCV (Video Processing)",
        "   3.7 Streamlit (Web UI)",
        "4. Module-by-Module Breakdown",
        "   4.1 config.py",
        "   4.2 scripts/scan.py",
        "   4.3 scripts/process_videos.py",
        "   4.4 scripts/embeddings.py",
        "   4.5 scripts/process_images.py",
        "   4.6 scripts/face_clustering.py",
        "   4.7 scripts/search.py",
        "   4.8 run_pipeline.py",
        "   4.9 app.py (Streamlit UI)",
        "5. Key Concepts & Theory",
        "   5.1 Embeddings & Vector Spaces",
        "   5.2 Cosine Similarity vs Euclidean Distance",
        "   5.3 Contrastive Learning",
        "   5.4 Transfer Learning",
        "   5.5 Incremental Indexing",
        "6. How to Run the System",
        "7. Potential Viva / Interview Questions & Answers",
    ]
    for item in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. PROJECT OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("1. Project Overview & Objectives")

    pdf.body_text(
        "This project is a locally-running, AI-powered media intelligence system inspired by "
        "Google Photos. It scans local folders for images and videos, automatically understands "
        "their content using deep learning models, and enables fast semantic search using "
        "natural language queries. Everything runs offline on the user's machine with no "
        "cloud dependency."
    )

    pdf.section_title("Core Objectives")
    objectives = [
        "Scan local folders recursively for images (.jpg, .jpeg, .png) and videos (.mp4, .avi, .mov)",
        "Generate semantic embeddings using CLIP for every image/frame",
        "Detect objects automatically using YOLOv8 (auto-tagging)",
        "Detect and cluster faces using InsightFace + DBSCAN",
        "Store embeddings in FAISS vector database for sub-100ms search",
        "Enable natural language search (e.g. 'photos of friends at the beach')",
        "Find visually similar images by comparing embeddings",
        "Extract video frames and index them with timestamps",
        "Provide a beautiful Streamlit web UI for browsing, searching, and organizing",
        "Support incremental indexing (only process new files, no rebuild needed)",
    ]
    for obj in objectives:
        pdf.bullet(obj)

    pdf.section_title("Performance Goals")
    pdf.bullet("Handle 5000+ images efficiently")
    pdf.bullet("Search latency < 100 ms using FAISS")
    pdf.bullet("Efficient incremental updates without full re-indexing")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. SYSTEM ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("2. System Architecture & Data Flow")

    pdf.section_title("Architecture Diagram (Text)")
    pdf.code_block(
        "User's Media Folders (Photos/, Videos/)\n"
        "        |\n"
        "        v\n"
        "[1] Scanner (scan.py)\n"
        "    - Walks directories recursively\n"
        "    - Filters by extension (.jpg, .png, .mp4, etc.)\n"
        "    - Tracks processed files (incremental)\n"
        "        |\n"
        "        v\n"
        "[2] Video Processor (process_videos.py)\n"
        "    - Extracts frames at 1 FPS using OpenCV\n"
        "    - Scene-change detection (optional)\n"
        "    - Stores frames with video source + timestamp\n"
        "        |\n"
        "        v\n"
        "[3] CLIP Embedding (embeddings.py)\n"
        "    - Loads CLIP vision model\n"
        "    - Generates 512-d embedding per image/frame\n"
        "    - Stores in FAISS IndexFlatIP\n"
        "        |\n"
        "        v\n"
        "[4] Object Detection + Face Detection (process_images.py)\n"
        "    - YOLOv8 nano: detects objects -> tags\n"
        "    - InsightFace: detects faces -> 512-d face embeddings\n"
        "        |\n"
        "        v\n"
        "[5] Face Clustering (face_clustering.py)\n"
        "    - Normalizes face embeddings\n"
        "    - DBSCAN with cosine metric\n"
        "    - Groups faces into person clusters\n"
        "        |\n"
        "        v\n"
        "[6] Streamlit UI (app.py)\n"
        "    - Search, Face Albums, Tags, Video Moments, Duplicates"
    )

    pdf.section_title("Data Flow Summary")
    pdf.body_text(
        "1. SCAN: Find all new media files not yet processed.\n"
        "2. EXTRACT: Pull frames from videos at 1 FPS.\n"
        "3. EMBED: Generate CLIP 512-dimensional embeddings for all images and frames.\n"
        "4. TAG: Run YOLOv8 object detection to produce tags (person, dog, car, etc.).\n"
        "5. FACE: Detect faces and compute 512-d face embeddings using InsightFace.\n"
        "6. CLUSTER: Group face embeddings into people using DBSCAN.\n"
        "7. STORE: Save FAISS index, metadata pickle files, and processed file tracker.\n"
        "8. SEARCH: Convert text query to CLIP embedding, search FAISS, return top-K.\n"
        "9. UI: Display results in Streamlit with grid view, albums, and video playback."
    )

    pdf.section_title("Storage Layout")
    pdf.code_block(
        "project/\n"
        "  index/\n"
        "    faiss_index.bin       <- FAISS binary index file\n"
        "    video_frames/         <- extracted video frames organized by video\n"
        "  metadata/\n"
        "    index_mapping.pkl     <- FAISS row -> file path mapping\n"
        "    processed_files.pkl   <- set of already-indexed file paths\n"
        "    image_metadata.pkl    <- per-image YOLO tags + face data\n"
        "    face_clusters.pkl     <- cluster_id -> list of face entries\n"
        "    face_labels.pkl       <- cluster_id -> human-assigned name"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. TECH STACK DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("3. Tech Stack Deep Dive")

    # --- 3.1 CLIP ---
    pdf.section_title("3.1 CLIP (Contrastive Language-Image Pretraining)")
    pdf.body_text(
        "CLIP is a model developed by OpenAI (2021) that learns visual concepts from natural "
        "language supervision. It was trained on 400 million (image, text) pairs from the internet."
    )
    pdf.sub_section("How CLIP Works")
    pdf.body_text(
        "CLIP has two encoders:\n"
        "  - Image Encoder: A Vision Transformer (ViT) or ResNet that converts images into "
        "a 512-dimensional vector.\n"
        "  - Text Encoder: A Transformer that converts text descriptions into a 512-dimensional vector.\n\n"
        "During training, CLIP uses contrastive learning: it learns to maximize the cosine "
        "similarity between matching (image, text) pairs while minimizing it for non-matching pairs. "
        "This means images of 'a dog on a beach' will have embeddings close to the text "
        "'a dog on a beach' in the shared 512-d vector space."
    )
    pdf.sub_section("Why CLIP is Used Here")
    pdf.bullet("Enables natural language search over images without manual labeling")
    pdf.bullet("One model handles both image and text embeddings in the same space")
    pdf.bullet("Zero-shot capability: works on any visual concept without fine-tuning")
    pdf.bullet("Model used: openai/clip-vit-base-patch32 (ViT-B/32 variant)")

    pdf.sub_section("Key Parameters")
    pdf.key_value("Model", "openai/clip-vit-base-patch32")
    pdf.key_value("Embedding Dim", "512")
    pdf.key_value("Image Input", "224x224 pixels (auto-resized)")
    pdf.key_value("Library", "transformers (HuggingFace)")

    # --- 3.2 YOLO ---
    pdf.add_page()
    pdf.section_title("3.2 YOLOv8 (You Only Look Once v8)")
    pdf.body_text(
        "YOLO is a real-time object detection model by Ultralytics. 'You Only Look Once' means "
        "the model processes the entire image in a single forward pass, making it extremely fast "
        "compared to region-based detectors (like Faster R-CNN)."
    )
    pdf.sub_section("How YOLO Works")
    pdf.body_text(
        "1. The image is divided into a grid of cells.\n"
        "2. Each cell predicts bounding boxes and class probabilities.\n"
        "3. Non-maximum suppression (NMS) removes duplicate detections.\n"
        "4. Output: list of (class_name, confidence, bounding_box) per detection.\n\n"
        "YOLOv8 uses a CSPDarknet backbone, PANet neck, and a decoupled head. The 'nano' variant "
        "(yolov8n) has only 3.2M parameters, making it suitable for CPU inference."
    )
    pdf.sub_section("Role in This Project")
    pdf.bullet("Auto-tags images with detected objects (person, car, dog, cat, etc.)")
    pdf.bullet("80 object classes from the COCO dataset")
    pdf.bullet("Tags stored in metadata for tag-based browsing and filtering")

    pdf.sub_section("Key Parameters")
    pdf.key_value("Model", "yolov8n.pt (nano, 6.2 MB)")
    pdf.key_value("Classes", "80 (COCO dataset)")
    pdf.key_value("Speed", "~15ms per image on GPU, ~100ms on CPU")

    # --- 3.3 FAISS ---
    pdf.section_title("3.3 FAISS (Facebook AI Similarity Search)")
    pdf.body_text(
        "FAISS is a library by Meta/Facebook for efficient similarity search and clustering "
        "of dense vectors. It is designed to handle billions of vectors and provides sub-millisecond "
        "search latency."
    )
    pdf.sub_section("How FAISS Works (IndexFlatIP)")
    pdf.body_text(
        "We use IndexFlatIP (Inner Product), which performs exact nearest-neighbor search "
        "using dot product. Since our vectors are L2-normalized, the inner product equals "
        "cosine similarity.\n\n"
        "How search works:\n"
        "1. User types a text query.\n"
        "2. CLIP encodes the query into a 512-d vector.\n"
        "3. FAISS computes dot product of this vector with all stored vectors.\n"
        "4. Returns top-K vectors with highest similarity scores.\n\n"
        "For larger datasets (100K+ images), you could switch to IndexIVFFlat or IndexHNSW "
        "for approximate nearest neighbors (faster but slightly less accurate)."
    )
    pdf.sub_section("Key Concepts")
    pdf.bullet("IndexFlatIP: Exact search, O(n) per query, perfect accuracy")
    pdf.bullet("IndexIVFFlat: Approximate search using inverted file index, faster for large datasets")
    pdf.bullet("Vectors must be float32 and same dimensionality")
    pdf.bullet("Supports incremental addition of vectors without rebuilding")

    # --- 3.4 InsightFace + MediaPipe ---
    pdf.add_page()
    pdf.section_title("3.4 InsightFace & MediaPipe (Face Recognition)")
    pdf.body_text(
        "Face recognition involves two steps: detection (where are the faces?) and embedding "
        "(who is this face?). We use two libraries:"
    )
    pdf.sub_section("MediaPipe (by Google)")
    pdf.bullet("Lightweight, runs on CPU with no native build dependencies")
    pdf.bullet("Provides face detection with bounding boxes in real-time")
    pdf.bullet("Used as a fallback if InsightFace is not available")

    pdf.sub_section("InsightFace (Primary)")
    pdf.bullet("Open-source face analysis toolkit using deep learning")
    pdf.bullet("Model: buffalo_s (small variant, ~125 MB)")
    pdf.bullet("Provides: face detection, face landmarks (68 + 106 point), face embeddings (512-d), "
               "gender/age estimation")
    pdf.bullet("Face embeddings are used for clustering and recognition")
    pdf.bullet("Runs via ONNX Runtime (no GPU required)")

    pdf.sub_section("Face Embedding Process")
    pdf.body_text(
        "1. InsightFace detects face bounding boxes in the image.\n"
        "2. Each face is aligned using facial landmark positions.\n"
        "3. The aligned face is passed through a recognition model (ArcFace-based).\n"
        "4. Output: 512-dimensional embedding vector per face.\n"
        "5. Similar faces (same person) have low cosine distance between embeddings."
    )

    # --- 3.5 DBSCAN ---
    pdf.section_title("3.5 DBSCAN (Density-Based Spatial Clustering)")
    pdf.body_text(
        "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised "
        "clustering algorithm that groups data points based on density, without needing to specify "
        "the number of clusters in advance."
    )
    pdf.sub_section("How DBSCAN Works")
    pdf.body_text(
        "Key parameters:\n"
        "  - eps: Maximum distance between two samples to be considered neighbors\n"
        "  - min_samples: Minimum number of points to form a dense region (cluster)\n\n"
        "Algorithm:\n"
        "1. For each point, find all points within eps distance.\n"
        "2. If a point has >= min_samples neighbors, it's a core point.\n"
        "3. Core points and their reachable neighbors form a cluster.\n"
        "4. Points not reachable from any core point are labeled as noise (-1).\n\n"
        "Why DBSCAN over K-Means for faces?\n"
        "- You don't know how many people are in the photos (no need to specify K)\n"
        "- It naturally handles noise (unrecognizable faces)\n"
        "- Works well with arbitrary-shaped clusters"
    )
    pdf.sub_section("Our Configuration")
    pdf.key_value("eps", "0.6 (cosine distance)")
    pdf.key_value("min_samples", "2")
    pdf.key_value("metric", "cosine (after L2-normalizing embeddings)")

    # --- 3.6 OpenCV ---
    pdf.add_page()
    pdf.section_title("3.6 OpenCV (Video Processing)")
    pdf.body_text(
        "OpenCV (Open Source Computer Vision Library) is used for reading video files and "
        "extracting frames. We use cv2.VideoCapture to decode video streams."
    )
    pdf.sub_section("Frame Extraction Strategy")
    pdf.body_text(
        "1. Constant FPS: Extract 1 frame per second (configurable). If video is 30fps, "
        "we take every 30th frame.\n"
        "2. Scene-Change Detection (optional): Compute mean pixel difference between consecutive "
        "frames. If difference > threshold (30.0), a scene change is detected and the frame is saved.\n"
        "3. Each frame is saved as a JPEG with metadata tracking the source video and timestamp."
    )
    pdf.sub_section("Key Functions Used")
    pdf.bullet("cv2.VideoCapture(path) - Open a video file")
    pdf.bullet("cap.read() - Read the next frame")
    pdf.bullet("cap.get(cv2.CAP_PROP_FPS) - Get the video's native frame rate")
    pdf.bullet("cv2.imwrite(path, frame) - Save a frame as an image")
    pdf.bullet("cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) - Convert to grayscale for scene detection")

    # --- 3.7 Streamlit ---
    pdf.section_title("3.7 Streamlit (Web UI)")
    pdf.body_text(
        "Streamlit is a Python library that turns data scripts into shareable web apps. "
        "It provides widgets (text inputs, buttons, sliders, image display) with zero "
        "JavaScript or HTML required.\n\n"
        "Our app has 6 pages:\n"
        "1. Search: Natural language search using CLIP + FAISS\n"
        "2. Face Albums: Auto-grouped people with labeling\n"
        "3. Browse Tags: Filter by YOLO-detected object tags\n"
        "4. Video Moments: Search within videos, play from timestamps\n"
        "5. Duplicates: Find near-identical images by embedding similarity\n"
        "6. Dashboard: Overview statistics of indexed media"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. MODULE BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("4. Module-by-Module Breakdown")

    pdf.section_title("4.1 config.py")
    pdf.body_text(
        "Central configuration file. All paths, model names, thresholds, and hyperparameters "
        "are defined here. This avoids hardcoding values across modules.\n\n"
        "Key settings:\n"
        "- MEDIA_DIRS: Default folders to scan (Photos/, videos/)\n"
        "- CLIP_MODEL_NAME: 'openai/clip-vit-base-patch32'\n"
        "- CLIP_EMBEDDING_DIM: 512\n"
        "- YOLO_MODEL_NAME: 'yolov8n.pt'\n"
        "- DBSCAN_EPS: 0.6, DBSCAN_MIN_SAMPLES: 2\n"
        "- VIDEO_FRAME_RATE: 1 FPS\n"
        "- SEARCH_TOP_K: 20\n"
        "- SIMILARITY_THRESHOLD: 0.95 (for duplicate detection)\n\n"
        "Also creates necessary directories (index/, metadata/, video_frames/) on import."
    )

    pdf.section_title("4.2 scripts/scan.py")
    pdf.body_text(
        "Recursively scans configured directories for media files.\n\n"
        "Key functions:\n"
        "- scan_media(media_dirs, incremental): Walk directories, filter by extension, "
        "skip already-processed files. Returns (images_list, videos_list).\n"
        "- load_processed_files(): Load set of processed paths from pickle.\n"
        "- save_processed_files(processed): Persist the set.\n\n"
        "Incremental behavior: Maintains a pickle set of all processed file paths. "
        "On subsequent runs, only files NOT in this set are returned as 'new'."
    )

    pdf.section_title("4.3 scripts/process_videos.py")
    pdf.body_text(
        "Extracts frames from video files using OpenCV.\n\n"
        "Key functions:\n"
        "- extract_frames(video_path, fps, scene_detect): Opens video with cv2.VideoCapture, "
        "reads frames at the specified interval, optionally checks for scene changes via "
        "pixel difference, saves frames as JPEG files.\n"
        "- process_all_videos(video_paths): Batch-processes multiple videos.\n\n"
        "Output: List of (frame_path, source_video_path, timestamp_seconds) tuples. "
        "Frames are stored in index/video_frames/<video_name>/frame_XXXXX.jpg."
    )

    pdf.add_page()
    pdf.section_title("4.4 scripts/embeddings.py")
    pdf.body_text(
        "Handles CLIP model loading, image/text embedding generation, and FAISS index management.\n\n"
        "Key functions:\n"
        "- _load_clip(): Lazy-loads CLIP model, processor, and tokenizer. Uses GPU if available.\n"
        "- embed_images(image_paths, batch_size=32): Opens images with PIL, processes in batches, "
        "generates 512-d embeddings, L2-normalizes them. Returns (embeddings_array, valid_paths).\n"
        "- embed_text(query_text): Tokenizes text, passes through CLIP text encoder, normalizes. "
        "Returns (1, 512) array.\n"
        "- load_faiss_index(): Loads existing FAISS index + mapping from disk, or creates new.\n"
        "- save_faiss_index(index, mapping): Writes index and mapping to disk.\n"
        "- add_to_index(index, mapping, embeddings, paths): Appends to existing index.\n"
        "- search_index(index, mapping, query_embedding, top_k): Returns (path, score) list.\n\n"
        "FAISS Index Type: IndexFlatIP (inner product = cosine similarity on normalized vectors).\n"
        "Mapping: A Python list where index i corresponds to FAISS row i's file path."
    )

    pdf.section_title("4.5 scripts/process_images.py")
    pdf.body_text(
        "Runs YOLO object detection and InsightFace face detection on each image.\n\n"
        "Key functions:\n"
        "- _load_yolo(): Lazy-loads YOLOv8 nano model.\n"
        "- _load_face_embedder(): Lazy-loads InsightFace FaceAnalysis (buffalo_s model).\n"
        "- detect_objects(image_path): Runs YOLO, returns list of tag strings.\n"
        "- detect_faces(image_path): Detects faces, returns (face_encodings, face_locations). "
        "Uses InsightFace if available, falls back to MediaPipe.\n"
        "- process_images(image_paths, video_frame_info): Processes all images incrementally, "
        "checkpoints every 50 images.\n\n"
        "Per-image metadata stored: path, tags, face_encodings, face_locations, source_video, timestamp."
    )

    pdf.section_title("4.6 scripts/face_clustering.py")
    pdf.body_text(
        "Groups all detected face embeddings into person clusters using DBSCAN.\n\n"
        "Process:\n"
        "1. Collects all face encodings from image_metadata.pkl.\n"
        "2. L2-normalizes all embeddings (important for cosine metric).\n"
        "3. Runs DBSCAN(eps=0.6, min_samples=2, metric='cosine').\n"
        "4. Groups results into clusters: {cluster_id: [face_info]}.\n"
        "5. Cluster -1 = unmatched/noise faces.\n\n"
        "Labeling: Users can assign names to cluster IDs (e.g., cluster 0 = 'Ajay'). "
        "Labels are stored in face_labels.pkl."
    )

    pdf.add_page()
    pdf.section_title("4.7 scripts/search.py")
    pdf.body_text(
        "Implements all search functionality.\n\n"
        "Search types:\n"
        "- text_search(query, top_k): Encodes text with CLIP -> searches FAISS -> returns "
        "top-K (path, score) results. This enables queries like 'sunset on beach'.\n"
        "- image_search(image_path, top_k): Encodes image with CLIP -> searches FAISS -> "
        "returns similar images (excluding the query image itself).\n"
        "- tag_search(tag_query): Filters image_metadata by YOLO tags. Simple set intersection.\n"
        "- find_duplicates(threshold=0.95): For each image in FAISS, finds other images with "
        "similarity >= threshold. Groups them into duplicate clusters.\n"
        "- get_all_tags(): Extracts unique tags from all metadata for the tag browser."
    )

    pdf.section_title("4.8 run_pipeline.py")
    pdf.body_text(
        "CLI orchestrator that runs the full indexing pipeline in sequence:\n\n"
        "Step 1: Scan for new media files (scan.py)\n"
        "Step 2: Extract video frames (process_videos.py)\n"
        "Step 3: Generate CLIP embeddings + update FAISS (embeddings.py)\n"
        "Step 4: Run YOLO + face detection (process_images.py)\n"
        "Step 5: Cluster faces (face_clustering.py)\n"
        "Step 6: Mark files as processed\n\n"
        "CLI flags:\n"
        "  --media-dirs: Override default scan directories\n"
        "  --rebuild: Wipe all metadata and re-index from scratch"
    )

    pdf.section_title("4.9 app.py (Streamlit UI)")
    pdf.body_text(
        "The web interface with 6 sections:\n\n"
        "1. Dashboard: Shows total indexed items, people found, unique tags, video frames.\n"
        "2. Search: Text input -> CLIP text_search or tag_search -> image grid with scores.\n"
        "3. Face Albums: Shows DBSCAN clusters as expandable albums. Users can rename clusters.\n"
        "4. Browse Tags: Tag cloud of all YOLO tags. Click to filter images by tag.\n"
        "5. Video Moments: Search within video frames. Shows frame thumbnail + source video + "
        "timestamp. Play button to open video at that timestamp.\n"
        "6. Duplicates: Adjustable similarity threshold slider. Finds near-identical images.\n\n"
        "Each image in the grid has a 'Find Similar' button -> redirects to image_search."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 5. KEY CONCEPTS & THEORY
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("5. Key Concepts & Theory")

    pdf.section_title("5.1 Embeddings & Vector Spaces")
    pdf.body_text(
        "An embedding is a learned, dense vector representation of data (image, text, face) in "
        "a high-dimensional space. Items with similar meaning are placed close together.\n\n"
        "Example: CLIP maps 'a photo of a sunset' (text) and an actual sunset photo (image) "
        "to nearby points in a 512-dimensional space. This shared space is what enables "
        "cross-modal search (text -> image).\n\n"
        "Properties of good embeddings:\n"
        "- Semantically similar items have high cosine similarity\n"
        "- Dissimilar items are far apart\n"
        "- The space is continuous (interpolation is meaningful)"
    )

    pdf.section_title("5.2 Cosine Similarity vs Euclidean Distance")
    pdf.body_text(
        "Cosine Similarity measures the angle between two vectors (ignores magnitude):\n"
        "  cos(A, B) = (A . B) / (|A| * |B|)\n"
        "  Range: -1 to 1 (1 = identical direction, 0 = orthogonal, -1 = opposite)\n\n"
        "Euclidean Distance measures the straight-line distance:\n"
        "  d(A, B) = sqrt(sum((Ai - Bi)^2))\n\n"
        "Why we use cosine similarity:\n"
        "- Embeddings from CLIP and InsightFace are high-dimensional\n"
        "- Cosine is scale-invariant (doesn't depend on vector magnitude)\n"
        "- More meaningful for semantic similarity\n"
        "- When vectors are L2-normalized, dot product = cosine similarity"
    )

    pdf.section_title("5.3 Contrastive Learning")
    pdf.body_text(
        "CLIP is trained using contrastive learning:\n"
        "- Given a batch of (image, text) pairs\n"
        "- The model learns to maximize similarity for matching pairs\n"
        "- And minimize similarity for non-matching pairs\n"
        "- Loss function: InfoNCE (cross-entropy over similarity matrix)\n\n"
        "This is what allows CLIP to understand that 'a red car' corresponds to images "
        "of red cars, without explicit object detection training."
    )

    pdf.section_title("5.4 Transfer Learning")
    pdf.body_text(
        "We use pre-trained models (CLIP, YOLOv8, InsightFace) without fine-tuning. "
        "This is transfer learning: the models were trained on massive datasets (400M pairs "
        "for CLIP, COCO for YOLO, MS1M for InsightFace) and their learned features generalize "
        "well to our use case.\n\n"
        "Benefits: No training data needed, no GPU required for training, works out-of-the-box."
    )

    pdf.section_title("5.5 Incremental Indexing")
    pdf.body_text(
        "The system supports adding new media without rebuilding the entire index:\n\n"
        "1. processed_files.pkl tracks which files have been indexed\n"
        "2. scan.py only returns files NOT in this set\n"
        "3. FAISS index.add() appends new vectors to the existing index\n"
        "4. Metadata pickle files are loaded, updated, and re-saved\n\n"
        "This means adding 10 new photos to 5000 only processes those 10 files, "
        "not all 5010. Critical for large media libraries."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. HOW TO RUN
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("6. How to Run the System")

    pdf.section_title("Prerequisites")
    pdf.bullet("Python 3.10+ installed")
    pdf.bullet("Internet connection for first run (model downloads)")
    pdf.bullet("~2 GB disk space for models (CLIP ~600 MB, InsightFace ~125 MB, YOLO ~6 MB)")

    pdf.section_title("Setup")
    pdf.code_block(
        "# Create virtual environment\n"
        "python -m venv venv\n"
        "\n"
        "# Activate it (Windows)\n"
        ".\\venv\\Scripts\\activate\n"
        "\n"
        "# Install dependencies\n"
        "pip install -r requirements.txt"
    )

    pdf.section_title("Build Index")
    pdf.code_block(
        "# Index your Photos folder\n"
        "python run_pipeline.py --media-dirs ./Photos\n"
        "\n"
        "# Or specify multiple folders\n"
        "python run_pipeline.py --media-dirs ./Photos ./MorePhotos ./Videos\n"
        "\n"
        "# Full rebuild from scratch\n"
        "python run_pipeline.py --media-dirs ./Photos --rebuild"
    )

    pdf.section_title("Launch UI")
    pdf.code_block(
        "streamlit run app.py\n"
        "# Opens at http://localhost:8501"
    )

    pdf.section_title("Dependencies (requirements.txt)")
    deps = [
        ("torch + torchvision", "PyTorch deep learning framework"),
        ("transformers", "HuggingFace library for CLIP model"),
        ("ultralytics", "YOLOv8 object detection"),
        ("mediapipe", "Google's face detection (lightweight)"),
        ("insightface", "Face recognition with 512-d embeddings"),
        ("onnxruntime", "ONNX model inference (used by InsightFace)"),
        ("faiss-cpu", "Facebook's vector similarity search"),
        ("opencv-python", "Video processing and frame extraction"),
        ("streamlit", "Web UI framework"),
        ("scikit-learn", "DBSCAN clustering algorithm"),
        ("Pillow", "Image loading and processing"),
        ("numpy", "Numerical operations"),
        ("watchdog", "File system monitoring (optional)"),
    ]
    for pkg, desc in deps:
        pdf.key_value(pkg, desc)
        pdf.ln(1)

    # ═══════════════════════════════════════════════════════════════════════
    # 7. VIVA / INTERVIEW Q&A
    # ═══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("7. Potential Viva / Interview Questions & Answers")

    pdf.qa_pair(
        "What is the purpose of this project?",
        "To build a local, offline AI-powered media intelligence system similar to Google Photos. "
        "It scans images and videos, understands their content using deep learning (CLIP for "
        "semantic understanding, YOLO for object detection, InsightFace for face recognition), "
        "stores embeddings in a FAISS vector database, and provides a Streamlit web UI for "
        "natural language search, face-based albums, and video moment browsing."
    )

    pdf.qa_pair(
        "What is CLIP and how does it enable text-based image search?",
        "CLIP (Contrastive Language-Image Pretraining) by OpenAI is a model trained on 400M "
        "image-text pairs. It has two encoders: one for images and one for text. Both produce "
        "512-dimensional vectors in a shared embedding space. When searching, the text query is "
        "encoded into a vector, and FAISS finds the images whose vectors are closest (highest "
        "cosine similarity). This means you can search 'dog on a beach' and find relevant images "
        "without any manual tagging."
    )

    pdf.qa_pair(
        "Why did you use FAISS instead of a traditional database?",
        "Traditional databases use exact matching (WHERE tag = 'dog'), which can't handle "
        "semantic similarity. FAISS is designed for high-dimensional vector similarity search. "
        "It computes cosine similarity between the query vector and all stored vectors efficiently. "
        "IndexFlatIP provides exact search; for larger datasets, approximate methods (IVF, HNSW) "
        "can search millions of vectors in milliseconds."
    )

    pdf.qa_pair(
        "What is the difference between FAISS IndexFlatIP and IndexIVFFlat?",
        "IndexFlatIP performs brute-force exact search by computing the dot product with every "
        "vector. Accuracy is 100% but speed is O(n). IndexIVFFlat first uses k-means to partition "
        "vectors into clusters, then only searches the nearest clusters. It's faster (O(sqrt(n))) "
        "but may miss some results. For our dataset size (<10K), IndexFlatIP is fast enough."
    )

    pdf.qa_pair(
        "How does face clustering work in your system?",
        "We use DBSCAN with cosine distance on face embeddings from InsightFace. "
        "First, all face embeddings (512-d) are collected from the metadata. They are L2-normalized. "
        "DBSCAN groups faces where cosine distance < 0.6 (eps parameter) with at least 2 samples. "
        "Unlike K-Means, DBSCAN doesn't need the number of clusters specified in advance, and "
        "naturally handles noise (unrecognizable faces get label -1)."
    )

    pdf.qa_pair(
        "Why DBSCAN over K-Means for face clustering?",
        "Three reasons: (1) We don't know how many people will appear in the photos, and K-Means "
        "requires specifying K upfront. (2) DBSCAN handles noise - isolated faces that don't "
        "match anyone are labeled as -1 (noise) instead of being forced into a wrong cluster. "
        "(3) DBSCAN works with any distance metric including cosine, which is more appropriate "
        "for high-dimensional face embeddings."
    )

    pdf.add_page()
    pdf.qa_pair(
        "How does incremental indexing work?",
        "We maintain a pickle file (processed_files.pkl) containing a Python set of all file paths "
        "that have been indexed. When scan.py runs, it skips any file already in this set. "
        "New embeddings are appended to the existing FAISS index via index.add(), and metadata "
        "dicts are updated and re-saved. This means adding 10 new photos to a library of 5000 "
        "only processes those 10 files."
    )

    pdf.qa_pair(
        "What is contrastive learning and how is it used in CLIP?",
        "Contrastive learning trains a model to distinguish between similar (positive) and "
        "dissimilar (negative) pairs. CLIP uses it by processing batches of (image, text) pairs. "
        "For a batch of N pairs, CLIP maximizes the dot product of the N correct (image, text) "
        "matches while minimizing the N^2 - N incorrect pairings. The loss function is InfoNCE, "
        "a form of cross-entropy applied to the similarity matrix."
    )

    pdf.qa_pair(
        "Why did you switch from face_recognition to InsightFace?",
        "The face_recognition library depends on dlib, which requires CMake and a C++ compiler "
        "to build from source. On our Python 3.14 environment, pre-built dlib wheels weren't "
        "available. InsightFace uses ONNX Runtime and installs as a pure Python package, "
        "requiring no native build tools. It also provides 512-d face embeddings (vs 128-d "
        "from dlib), which are generally more discriminative."
    )

    pdf.qa_pair(
        "How does video processing work in the system?",
        "Videos are processed using OpenCV. We open each video with cv2.VideoCapture, determine "
        "the native FPS, and extract frames at 1 frame per second (configurable). Each frame is "
        "saved as a JPEG with metadata linking it to the source video and timestamp. Optionally, "
        "scene-change detection compares consecutive frames by mean pixel difference. Extracted "
        "frames go through the same pipeline as images (CLIP embedding, YOLO tagging, face detection)."
    )

    pdf.qa_pair(
        "What is cosine similarity and why is it preferred over Euclidean distance?",
        "Cosine similarity measures the angle between two vectors: cos(A,B) = (A.B)/(|A|.|B|). "
        "Range is -1 to 1. It's preferred because: (1) It's scale-invariant - doesn't depend "
        "on vector magnitude, only direction. (2) More meaningful for high-dimensional spaces "
        "where Euclidean distance tends to converge (curse of dimensionality). (3) When vectors "
        "are L2-normalized, dot product equals cosine similarity, making FAISS IndexFlatIP "
        "equivalent to cosine search."
    )

    pdf.qa_pair(
        "How does duplicate detection work?",
        "For each image embedding in FAISS, we search for other images with cosine similarity "
        ">= 0.95 (configurable threshold). Images exceeding this threshold are grouped into "
        "duplicate clusters. Higher threshold = stricter matching (near-identical only). "
        "Lower threshold catches more variations but may group similar-but-different images."
    )

    pdf.qa_pair(
        "What is YOLOv8 and how does it differ from previous YOLO versions?",
        "YOLOv8 (2023) by Ultralytics uses a CSPDarknet backbone, PANet feature pyramid neck, "
        "and an anchor-free, decoupled detection head. Key improvements over v5: anchor-free "
        "design (no predefined aspect ratios), better accuracy-speed tradeoff, native support "
        "for classification, detection, segmentation, and pose estimation in one framework. "
        "The nano variant (yolov8n) has 3.2M parameters and detects 80 COCO object classes."
    )

    pdf.add_page()
    pdf.qa_pair(
        "What is a Vision Transformer (ViT) and how is it used in CLIP?",
        "ViT (Vision Transformer) applies the Transformer architecture to images. An image is "
        "split into fixed-size patches (32x32 for ViT-B/32), each patch is linearly embedded, "
        "and position embeddings are added. These patch tokens are processed by standard "
        "Transformer encoder layers (self-attention + FFN). The [CLS] token output becomes "
        "the image embedding. CLIP's image encoder is a ViT that produces 512-d vectors."
    )

    pdf.qa_pair(
        "What are the limitations of this system?",
        "1. CPU-only inference is slow (~10 min for 447 items); GPU would be 10x faster. "
        "2. CLIP has limited understanding of spatial relationships and counting. "
        "3. Face clustering accuracy depends on photo quality and angles. "
        "4. IndexFlatIP is O(n) per query; for 100K+ images, approximate indices are needed. "
        "5. No audio transcription for videos (could add Whisper). "
        "6. No fine-tuning on user's specific domain."
    )

    pdf.qa_pair(
        "How would you scale this to millions of images?",
        "1. Switch FAISS to IndexIVFFlat or IndexHNSW for approximate nearest neighbors. "
        "2. Use GPU acceleration for CLIP/YOLO/InsightFace inference. "
        "3. Process images in parallel using multiprocessing. "
        "4. Replace pickle metadata with SQLite for faster queries. "
        "5. Use batch processing with progress checkpoints. "
        "6. Consider FAISS IndexPQ for compressed vectors (lower memory)."
    )

    pdf.qa_pair(
        "What is L2 normalization and why is it important?",
        "L2 normalization divides each vector by its L2 norm (Euclidean length), making "
        "the vector unit-length (magnitude = 1). Formula: v_norm = v / ||v||. This is "
        "important because: (1) It makes dot product equivalent to cosine similarity. "
        "(2) FAISS IndexFlatIP computes dot products, so normalized vectors give cosine scores. "
        "(3) It removes magnitude bias, focusing comparisons on direction (semantic content)."
    )

    pdf.qa_pair(
        "What is the role of pickle files in your system?",
        "Pickle is Python's serialization format. We use it to persist:\n"
        "- processed_files.pkl: Set of indexed file paths (for incremental indexing)\n"
        "- index_mapping.pkl: List mapping FAISS row index -> file path\n"
        "- image_metadata.pkl: Dict of path -> {tags, face_encodings, face_locations, ...}\n"
        "- face_clusters.pkl: Dict of cluster_id -> list of face entries\n"
        "- face_labels.pkl: Dict of cluster_id -> human-assigned name\n\n"
        "Pickle is simple and fast for small-medium datasets. For production, SQLite would "
        "be more robust and support concurrent access."
    )

    pdf.qa_pair(
        "Explain the complete flow when a user searches for 'friends at the beach'.",
        "1. User types 'friends at the beach' in the Streamlit search bar.\n"
        "2. The text is passed to embed_text(), which tokenizes it with CLIP's text tokenizer.\n"
        "3. The tokens go through CLIP's text encoder Transformer, producing a 512-d vector.\n"
        "4. The vector is L2-normalized.\n"
        "5. FAISS search_index() computes the dot product of this vector with all 447 stored image vectors.\n"
        "6. The top-K results (highest dot product = highest cosine similarity) are returned.\n"
        "7. The mapping list converts FAISS row indices to file paths.\n"
        "8. Streamlit displays the images in a grid, sorted by relevance score.\n"
        "9. Each image shows its YOLO tags and a 'Find Similar' button."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "AI_Media_Intelligence_Study_Guide.pdf")
    pdf.output(output_path)
    print(f"\nPDF generated: {output_path}")
    print(f"Pages: {pdf.pages_count}")
    return output_path


if __name__ == "__main__":
    build_pdf()
