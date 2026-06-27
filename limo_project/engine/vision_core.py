import os
import cv2
import numpy as np
from PIL import Image
import io
import face_recognition
import json
from sklearn.neighbors import KNeighborsClassifier

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

def extract_classification_features(img_bgr, face_count):
    """
    Extracts a 9-dimensional feature vector from the image for machine learning:
    1. face_count: raw face count
    2. skin_percent: percentage of skin tone pixels
    3. center_skin_percent: percentage of skin tone concentrated in the center
    4. edge_density: Canny edge density after Gaussian blur
    5. num_colors: quantized colors count (simplification)
    6. line_count: straight line count (Hough lines)
    7. text_lines_count: horizontal morphological text blocks
    8. sky_columns: number of smooth sky columns in top 30% (0 to 3)
    9. nature_percent: percentage of nature colors in bottom 70%
    """
    try:
        h, w = img_bgr.shape[:2]
        aspect_ratio = w / h
        
        scale = 256 / max(h, w)
        small_w = int(w * scale)
        small_h = int(h * scale)
        img_small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        
        # Skin mask
        lower_skin1 = np.array([0, 20, 60], dtype=np.uint8)
        upper_skin1 = np.array([20, 150, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        lower_skin2 = np.array([170, 20, 60], dtype=np.uint8)
        upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        skin_percent = np.sum(skin_mask > 0) / skin_mask.size
        
        # Center skin
        ctr_left = int(small_w * 0.20)
        ctr_right = int(small_w * 0.80)
        ctr_top = int(small_h * 0.15)
        ctr_bottom = int(small_h * 0.85)
        center_skin_mask = skin_mask[ctr_top:ctr_bottom, ctr_left:ctr_right]
        center_skin_percent = np.sum(center_skin_mask > 0) / center_skin_mask.size if center_skin_mask.size > 0 else 0
        
        # Blur & Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color count
        quantized = (img_small // 64) * 64
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
        num_colors = len(unique_colors)
        
        # Straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        # Text lines
        text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, text_kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_lines_count = 0
        for cnt in contours:
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            aspect = w_b / h_b
            if aspect > 2.0 and 3 <= h_b <= 20 and w_b >= 30:
                text_lines_count += 1
                
        # Sky columns
        top_h = int(small_h * 0.3)
        top_region = img_small[:top_h, :]
        top_hsv = hsv[:top_h, :]
        col_w = small_w // 3 if small_w >= 3 else 1
        
        sky_columns = 0
        for i in range(3):
            col_bgr = top_region[:, i*col_w : (i+1)*col_w]
            col_hsv = top_hsv[:, i*col_w : (i+1)*col_w]
            col_gray = gray[:top_h, i*col_w : (i+1)*col_w]
            
            col_edges = cv2.Canny(col_gray, 30, 100)
            col_edges_density = np.sum(col_edges > 0) / col_edges.size if col_edges.size > 0 else 0
            
            blue_mask = cv2.inRange(col_hsv, np.array([90, 30, 80]), np.array([130, 255, 255]))
            bright_mask = cv2.inRange(col_hsv, np.array([0, 0, 180]), np.array([180, 45, 255]))
            sky_p = np.sum(cv2.bitwise_or(blue_mask, bright_mask) > 0) / col_bgr.size if col_bgr.size > 0 else 0
            
            if col_edges_density < 0.05 and sky_p > 0.15:
                sky_columns += 1
                
        # Nature percent
        bottom_hsv = hsv[top_h:, :]
        green_mask = cv2.inRange(bottom_hsv, np.array([30, 25, 40]), np.array([85, 255, 255]))
        green_percent = np.sum(green_mask > 0) / bottom_hsv.size if bottom_hsv.size > 0 else 0
        brown_mask = cv2.inRange(bottom_hsv, np.array([10, 40, 30]), np.array([25, 255, 180]))
        brown_percent = np.sum(brown_mask > 0) / bottom_hsv.size if bottom_hsv.size > 0 else 0
        nature_percent = green_percent + brown_percent
        
        return [
            float(face_count),
            float(skin_percent),
            float(center_skin_percent),
            float(edge_density),
            float(num_colors),
            float(line_count),
            float(text_lines_count),
            float(sky_columns),
            float(nature_percent)
        ]
    except Exception:
        return [float(face_count), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def normalize_features(features):
    if features is None or len(features) < 9:
        return features
    norm = list(features)
    norm[0] = min(1.0, features[0] / 5.0)   # face_count scaled to [0, 1]
    # norm[1] is skin_percent
    # norm[2] is center_skin_percent
    # norm[3] is edge_density
    norm[4] = min(1.0, features[4] / 256.0) # num_colors
    norm[5] = min(1.0, features[5] / 100.0) # line_count
    norm[6] = min(1.0, features[6] / 30.0)  # text_lines_count
    norm[7] = features[7] / 3.0             # sky_columns
    # norm[8] is nature_percent
    return norm


class CategoryClassifier:
    _manual_overrides = []
    
    @classmethod
    def train_model(cls, db_manager):
        try:
            cls._manual_overrides = db_manager.get_manual_overrides()
        except Exception:
            cls._manual_overrides = []

    @classmethod
    def predict(cls, feature_vector, default_heuristic_category):
        if not cls._manual_overrides:
            return None
            
        norm_q = normalize_features(feature_vector)
        
        best_dist = float('inf')
        best_override = None
        
        for override in cls._manual_overrides:
            norm_o = normalize_features(override["features"])
            # Compute Euclidean distance (including face_count at index 0)
            diff = np.array(norm_q) - np.array(norm_o)
            dist = np.linalg.norm(diff)
            if dist < best_dist:
                best_dist = dist
                best_override = override
                
        if best_override is not None:
            # Check thresholds
            # 1. Negative reinforcement: default heuristic matches the category user rejected/changed from
            if best_dist <= 0.35:
                if default_heuristic_category == best_override["original_category"]:
                    return best_override["category_label"]
            
            # 2. Positive reinforcement: features are extremely close to the overridden image
            if best_dist <= 0.25:
                return best_override["category_label"]
                
        return None


def classify_image(img_bgr, face_count, features_list=None):
    """
    Classifies image based on machine learning prediction or fallback heuristics:
    - Portrait: Exactly 1 face.
    - Couple: Exactly 2 faces.
    - Group: 3 or more faces.
    - Landscape: No human indicators (no faces, no skin tone) and sky/ground layers.
    - Documents: horizontal morphological text blocks.
    - Uncategorized: fallback.
    """
    features = features_list
    if features is None:
        features = extract_classification_features(img_bgr, face_count)
        
    # First determine candidate category based on strict rules/heuristics
    if face_count == 1:
        candidate = "Portrait"
    elif face_count == 2:
        candidate = "Couple"
    elif face_count >= 3:
        candidate = "Group"
    else:
        candidate = _classify_heuristics(img_bgr, features)
        
    # Check if ML model has manual overrides and can predict/correct this candidate
    ml_pred = CategoryClassifier.predict(features, candidate)
    if ml_pred is not None:
        return ml_pred
        
    return candidate


def _classify_heuristics(img_bgr, features):
    try:
        h, w = img_bgr.shape[:2]
        aspect_ratio = w / h
        
        skin_percent = features[1]
        center_skin_percent = features[2]
        edge_density = features[3]
        num_colors = features[4]
        line_count = features[5]
        text_lines_count = features[6]
        sky_columns = features[7]
        nature_percent = features[8]
        
        # 1. Document Check
        if text_lines_count >= 5:
            return "Documents"
            
        # 2. Landscape Check (Requires no humans: skin_percent <= 0.035)
        if aspect_ratio >= 0.65 and skin_percent <= 0.035:
            # Downscale and analyze colors in HSV
            scale = 128 / max(h, w)
            small_w = int(w * scale)
            small_h = int(h * scale)
            img_small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
            
            # Define nature masks
            green_mask = cv2.inRange(hsv, np.array([30, 20, 30]), np.array([90, 255, 255]))
            blue_mask = cv2.inRange(hsv, np.array([90, 20, 30]), np.array([135, 255, 255]))
            brown_yellow_mask = cv2.inRange(hsv, np.array([5, 20, 30]), np.array([30, 255, 255]))
            white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 30, 255]))
            
            nature_mask = cv2.bitwise_or(green_mask, blue_mask)
            nature_mask = cv2.bitwise_or(nature_mask, brown_yellow_mask)
            nature_mask = cv2.bitwise_or(nature_mask, white_mask)
            
            nature_ratio = np.sum(nature_mask > 0) / nature_mask.size
            
            # Require at least 10% saturated nature color (green, blue, brown/yellow) to exclude plain white/grey screens
            green_p = np.sum(green_mask > 0) / green_mask.size
            blue_p = np.sum(blue_mask > 0) / blue_mask.size
            brown_p = np.sum(brown_yellow_mask > 0) / brown_yellow_mask.size
            has_color_nature = (green_p + blue_p + brown_p) >= 0.10
            
            if sky_columns >= 1 or (nature_ratio >= 0.30 and has_color_nature):
                return "Landscape"
                
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
