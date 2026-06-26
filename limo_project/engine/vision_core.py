import os
import cv2
import numpy as np
from PIL import Image
import io
import face_recognition

def scan_directory_generator(directory):
    """
    Recursively scans the directory for image files using os.scandir generators.
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    stack = [directory]
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in extensions:
                            yield entry.path
        except PermissionError:
            continue
        except Exception:
            continue

def load_and_downscale_image(file_path, max_edge=1024):
    """
    Loads an image from path and downscales it if its max edge exceeds max_edge.
    Returns (img_bgr, original_w, original_h) or (None, 0, 0) on failure.
    """
    try:
        # Read image bytes using Python's open() to support Windows Unicode/non-ASCII paths
        with open(file_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, 0, 0
        
        h, w = img_bgr.shape[:2]
        if max(h, w) <= max_edge:
            return img_bgr, w, h
        
        scale = max_edge / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, w, h
    except Exception:
        return None, 0, 0

def classify_image(img_bgr, face_count):
    """
    Classifies image based on local heuristics:
    - Portraits: face_count > 0
    - Screenshots/Documents: High edge density and face_count == 0
    - Landscapes: horizontal aspect ratio (w/h >= 1.2) and face_count == 0
    - Uncategorized: everything else
    """
    if face_count > 0:
        return "Portraits"
    
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        aspect_ratio = w / h
        
        # Run Canny Edge Detection to calculate edge density
        # Low and high thresholds: 50, 150
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels
        
        # High edge density typically indicates text, graphics, screenshots or document scans
        if edge_density > 0.07:
            return "Screenshots/Documents"
        
        if aspect_ratio >= 1.2:
            return "Landscapes"
            
    except Exception:
        pass
        
    return "Uncategorized"

def generate_thumbnail(img_bgr, size=(200, 200)):
    """
    Generates a compressed JPEG thumbnail from BGR image.
    """
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail(size)
        
        out_bytes = io.BytesIO()
        pil_img.save(out_bytes, format="JPEG", quality=85)
        return out_bytes.getvalue()
    except Exception:
        return None

def extract_faces_and_embeddings(img_bgr):
    """
    Extracts face bounding boxes and 128-d encodings.
    """
    try:
        # face_recognition library needs RGB format
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # HOG model is faster for CPU
        face_locations = face_recognition.face_locations(img_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
        return face_locations, face_encodings
    except Exception:
        return [], []

def search_similar_faces(ref_embedding, db_embeddings, feedbacks=None, tolerance=0.55):
    """
    Performs vectorized similarity search using Euclidean Distance with online feedback learning.
    ref_embedding: numpy array of shape (128,)
    db_embeddings: list of dicts with keys: 'embedding_id', 'file_id', 'embedding', 'absolute_path', 'file_name', 'category_label', 'thumbnail_blob'
    feedbacks: dict mapping target_embedding_id (int) -> is_match (1 for 👍, 0 for 👎)
    tolerance: maximum distance threshold
    
    Returns: list of matched dict records sorted by confidence (closest distance first).
    """
    if not db_embeddings or ref_embedding is None:
        return []
        
    if feedbacks is None:
        feedbacks = {}

    # Multi-Reference Template Expansion:
    # Include the original reference + any verified positive face embeddings
    templates = [ref_embedding]
    for item in db_embeddings:
        emb_id = item.get("embedding_id")
        if emb_id in feedbacks and feedbacks[emb_id] == 1:
            templates.append(item["embedding"])

    # Stack templates into (M, 128)
    templates_matrix = np.vstack(templates)
    
    # Stack stored embeddings into (N, 128)
    embeddings_matrix = np.vstack([item["embedding"] for item in db_embeddings])
    
    # Compute Euclidean distance using broadcasting:
    # (N, 1, 128) - (1, M, 128) -> (N, M, 128) -> norm on axis 2 -> (N, M) -> min on axis 1 -> (N,)
    diff = embeddings_matrix[:, np.newaxis, :] - templates_matrix[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    min_distances = np.min(distances, axis=1)
    
    matches = []
    for i, distance in enumerate(min_distances):
        item = db_embeddings[i].copy()
        emb_id = item.get("embedding_id")
        
        # Blacklist exclusion: If user marked as not-a-match (👎), omit entirely
        if emb_id in feedbacks and feedbacks[emb_id] == 0:
            continue
            
        is_explicit_positive = (emb_id in feedbacks and feedbacks[emb_id] == 1)
        
        if distance <= tolerance or is_explicit_positive:
            item["distance"] = float(distance)
            item["confidence"] = float(max(0.0, 1.0 - distance))
            item.pop("embedding", None)
            matches.append(item)
            
    matches.sort(key=lambda x: x["distance"])
    return matches
