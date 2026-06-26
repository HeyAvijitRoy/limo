# MVP Specification: Local Intelligent Media Organizer (LIMO)
Target Platform: Windows 10/11 (Local Execution Only)
Development Engine: Google Antigravity Agentic Orchestration

---

## 1. Executive Summary & Core Constraints
LIMO is a highly optimized, fully localized desktop media utility that implements asynchronous background facial indexation, computer vision categorization, and instant vector similarity search.

### Strict Guardrails for the Antigravity Agent:
* **Zero Network Activity:** All feature extraction, classification, and vector indexing must happen entirely on the local loopback interface or CPU/GPU. No external API calls (e.g., cloud vision APIs) are permitted.
* **Targeted Scope:** The application must operate strictly on an explicitly chosen preset user directory and its subdirectories. No global system scanning.
* **Memory Footprint:** Memory management must be bounded. Image frames must be compressed or downscaled during inference, and pointers must be cleaned up explicitly to prevent memory leaks during deep folder traversals.

---

## 2. Technical Architecture & Engine Pipeline

The backend engine handles directory traversal, face encoding, image classification, and vector storage.

### A. Core Dependencies & Libraries
The engine must be constructed using a Python 3.11+ backend utilizing:
* `opencv-python-headless`: Fast image decoding and colorspace transformations.
* `face_recognition` (built on `dlib` with `BLAS`/`LAPACK` or `CUDA` support if available): Face detection and 128-dimensional floating-point vector extraction.
* `scikit-learn` or `NumPy`: For fast localized Cosine/Euclidean distance calculations.
* `Pillow`: For lightweight thumbnail generation.
* `sqlite3`: Built-in transactional local database for state and metadata retention.

### B. The Storage Engine (SQLite Schema)
To ensure sub-millisecond querying without loading thousands of high-res images into memory, the agent must implement a local relational cache database (`limo_cache.db`).

```sql
CREATE TABLE IF NOT EXISTS media_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    absolute_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    date_modified REAL NOT NULL,
    category_label TEXT DEFAULT 'Uncategorized',
    thumbnail_blob BLOB
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    embedding_vector BLOB NOT NULL, -- 128-float array serialized via numpy.tobytes()
    FOREIGN KEY(file_id) REFERENCES media_files(file_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_file_path ON media_files(absolute_path);

```

### C. Performance & Memory Optimization Protocols

1. **Iterative Batch Processing:** When scanning subdirectories, the engine must stream file paths iteratively using `os.scandir` generators instead of building massive in-memory arrays via `os.walk`.
2. **Inference Image Downscaling:** Prior to running facial recognition detection (`face_recognition.face_locations`), high-resolution user photos must be provisionally downscaled (e.g., maximum bounding box edge of 1024px) in RAM to keep the peak working set memory under 500MB.
3. **Vectorized Matching:** Searching must compare raw floating-point vectors using matrix-based NumPy operations directly, minimizing Python-level `for` loops during distance validation.

---

## 3. UI/UX Functional Design

The UI should follow a minimalist, modern desktop interface built with **PyQt6** or **Tkinter/CustomTkinter** for rapid compilation and native performance on Windows.

### A. Window Layout Workspaces

1. **Configuration Ribbon:** * A directory input field with a native system "Browse" folder selection dialog.
* An explicit "Sync / Index Directory" execution trigger.
* A reactive progress bar showing processed vs. total files.


2. **Search & Filter Core Control Canvas:**
* A drop-zone / file selection block to upload a **Reference Face Photo**.
* A strictness/tolerance slider (maps to Euclidean distance threshold, default `0.55`).
* A drop-down selector to filter by automatic categories (e.g., Landscapes, Portraits, Documents, Screenshots).


3. **Dynamic Grid Interface:**
* A multi-column, lazy-loading responsive image grid that displays matches.
* Images must render from cached low-resolution `thumbnail_blob` objects rather than loading raw files from the storage drive.
* Double-clicking any tile triggers a native Windows shell action (`os.startfile`) to reveal the file in File Explorer.



---

## 4. Antigravity Step-by-Step Implementation Workflow

*The Antigravity Agent should execute and verify the project using these distinct phases:*

### Phase 1: Environment & Scaffolding

* Configure a local Python isolated environment.
* Verify local C++ compiler availability for building native bindings (`dlib`).
* Establish directory structure:
```
limo_project/
├── engine/
│   ├── __init__.py
│   ├── database.py       # SQLite transactions
│   └── vision_core.py    # Face extraction and categorization math
├── ui/
│   ├── __init__.py
│   └── main_window.py    # GUI view layout and event signaling
├── face_index.db         # Local binary vector storage
└── main.py               # Framework entry point

```



### Phase 2: Core Engine Validation

* Implement `database.py` connection management.
* Write file-walking algorithms inside `vision_core.py` incorporating basic categorization telemetry (e.g., checking aspect ratio or structural properties to quickly categorize screenshots vs portrait photos).
* Test embedding extraction correctness against a mock directory with 5 test images. Ensure vectors save accurately as SQLite binary BLOB variables.

### Phase 3: Fast Search Implementation

* Create a specialized search utility function that reads all target arrays from the DB in a single pass into a structured multi-dimensional NumPy array.
* Execute distance evaluations in a single operation:

$$\text{Distance} = \sqrt{\sum (V_{\text{target}} - V_{\text{stored}})^2}$$


* Return matches matching the tolerance parameters instantly.

### Phase 4: UI Development & Integration

* Build out the desktop interface matching Section 3.
* Connect the background indexing thread to the main GUI thread using thread-safe signaling models (`QThread` / `pyqtSignal` if using PyQt6) to keep the layout entirely responsive and prevent Windows "Not Responding" application locks during synchronization passes.
