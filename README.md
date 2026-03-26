# 🧠 MediaMind AI

### AI-Powered Semantic Media Intelligence System

MediaMind AI is a **local AI-powered media intelligence system** that enables users to search, organize, and understand images and videos using **natural language queries**.

Inspired by Google Photos, this system runs **completely offline** and uses state-of-the-art deep learning models to provide intelligent media search and organization.

---

## 🚀 Key Features

### 🔍 Semantic Search

* Search images using natural language (e.g., *"friends at the beach"*)
* Powered by CLIP multimodal embeddings

### 🏷️ Automatic Tagging

* Detect objects like **person, car, dog, etc.**
* Uses YOLOv8 for real-time object detection

### 🙂 Face Recognition & Clustering

* Detects and groups faces automatically
* Uses InsightFace + DBSCAN clustering

### 🎥 Video Intelligence

* Extracts frames from videos (1 FPS)
* Enables search inside videos using timestamps

### ⚡ Fast Similarity Search

* Uses FAISS for high-speed vector search
* Sub-100ms query performance

### 🔁 Incremental Indexing

* Processes only new media files
* Avoids reprocessing existing data

### 🌐 Interactive Web UI

* Built with Streamlit
* Includes:

  * Search
  * Face Albums
  * Tag Browser
  * Duplicate Detection
  * Video Moments

---

## 🧠 System Architecture

```
Media Files → Scan → Process → Embed → Store → Search → UI
```

### Pipeline Flow:

1. **Scan** → Detect new images/videos
2. **Extract** → Video frames using OpenCV
3. **Embed** → Generate CLIP embeddings
4. **Tag** → Object detection using YOLOv8
5. **Face Detection** → InsightFace embeddings
6. **Cluster** → Group faces using DBSCAN
7. **Store** → Save vectors in FAISS
8. **Search** → Query via text/image

---

## 🛠️ Tech Stack

| Component        | Technology            |
| ---------------- | --------------------- |
| Language         | Python                |
| Embeddings       | CLIP (ViT-B/32)       |
| Object Detection | YOLOv8                |
| Face Recognition | InsightFace           |
| Clustering       | DBSCAN (scikit-learn) |
| Vector DB        | FAISS                 |
| Video Processing | OpenCV                |
| UI               | Streamlit             |

---

## 📂 Project Structure

```
project/
│
├── scripts/
│   ├── scan.py
│   ├── process_images.py
│   ├── process_videos.py
│   ├── embeddings.py
│   ├── face_clustering.py
│   ├── search.py
│
├── index/
│   ├── faiss_index.bin
│   ├── video_frames/
│
├── metadata/
│   ├── index_mapping.pkl
│   ├── processed_files.pkl
│   ├── image_metadata.pkl
│   ├── face_clusters.pkl
│
├── app.py
├── run_pipeline.py
└── config.py
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/media-mind-ai.git
cd media-mind-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Step 1: Index Media Files

```bash
python run_pipeline.py --media-dirs ./Photos ./Videos
```

### 🔹 Step 2: Launch Web UI

```bash
streamlit run app.py
```

👉 Open in browser:
http://localhost:8501

---

## 🔍 Example Queries

* "dog playing in park"
* "friends at night"
* "cars on road"
* "me and my friends"

---

## 🧪 Core Concepts Used

* **Embeddings & Vector Spaces**
* **Cosine Similarity**
* **Contrastive Learning (CLIP)**
* **Transfer Learning**
* **Clustering (DBSCAN)**
* **Approximate Nearest Neighbors (FAISS)**

---

## ⚡ Performance

* Handles **5000+ images efficiently**
* Search latency **< 100 ms**
* Supports incremental updates

---

## ⚠️ Limitations

* Slower performance on CPU
* CLIP struggles with complex spatial relationships
* Not optimized for very large datasets (>100K images)
* No audio processing for videos

---

## 🚀 Future Improvements

* GPU acceleration
* Audio transcription (Whisper)
* Cloud sync support
* Mobile app interface
* Scalable FAISS indexing (HNSW/IVF)
* Replace pickle with database (SQLite/PostgreSQL)

---

## 👨‍💻 Author

**Ajay Singh Rathore**

---

## ⭐ Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 💡 Inspiration

Inspired by Google Photos — built as a local, privacy-focused alternative using AI.
