import os
import tempfile
import sqlite3
import io
import numpy as np
import cv2
import pytest
from PIL import Image
from limo_project.engine.database import DatabaseManager
from limo_project.engine.vision_core import (
    load_and_downscale_image,
    classify_image,
    generate_thumbnail,
    search_similar_faces
)

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(path)
    yield db
    # Cleanup
    try:
        os.remove(path)
    except OSError:
        pass

def test_database_insert_and_cascade(temp_db):
    # Test adding a media file
    file_id = temp_db.add_media_file(
        absolute_path="C:/photos/pic1.jpg",
        file_name="pic1.jpg",
        date_modified=123456789.0,
        category_label="Landscapes",
        thumbnail_blob=b"fake_thumbnail"
    )
    assert file_id == 1
    
    # Test checking path
    record = temp_db.get_media_file_by_path("C:/photos/pic1.jpg")
    assert record is not None
    assert record["file_name"] == "pic1.jpg"
    assert record["category_label"] == "Landscapes"
    
    # Test adding embedding
    mock_vector = np.zeros(128, dtype=np.float64)
    mock_vector[0] = 1.5
    temp_db.add_face_embedding(file_id, mock_vector)
    
    # Retrieve embeddings
    embeddings = temp_db.get_all_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0]["file_id"] == file_id
    assert np.allclose(embeddings[0]["embedding"], mock_vector)
    
    # Test Cascade Delete
    temp_db.delete_media_file("C:/photos/pic1.jpg")
    record_after = temp_db.get_media_file_by_path("C:/photos/pic1.jpg")
    assert record_after is None
    
    # Embeddings should be cascade deleted
    embeddings_after = temp_db.get_all_embeddings()
    assert len(embeddings_after) == 0

def test_load_and_downscale_image():
    # Create a temporary large image
    # Size 2000 x 1000
    large_img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    # Draw some patterns to ensure it's a valid image
    cv2.circle(large_img, (1000, 500), 200, (255, 255, 255), -1)
    
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cv2.imwrite(path, large_img)
        
        resized, w, h = load_and_downscale_image(path, max_edge=1024)
        assert resized is not None
        assert w == 2000
        assert h == 1000
        # Largest dimension should be exactly 1024
        assert max(resized.shape[:2]) == 1024
        # Original ratio: 2.0. New ratio: 1024 / 512 = 2.0
        assert resized.shape[1] == 1024
        assert resized.shape[0] == 512
    finally:
        os.remove(path)

def test_classify_image_heuristics():
    # 1. Portraits check (face_count > 0)
    img_portrait = np.zeros((500, 500, 3), dtype=np.uint8)
    assert classify_image(img_portrait, face_count=2) == "Portraits"
    
    # 2. Landscapes check (face_count == 0, aspect ratio horizontal >= 1.2, low edge density)
    img_landscape = np.zeros((600, 800, 3), dtype=np.uint8) # 800/600 = 1.33
    cv2.circle(img_landscape, (400, 300), 50, (0, 255, 0), -1)
    assert classify_image(img_landscape, face_count=0) == "Landscapes"
    
    # 3. Screenshots/Documents check (high edge density, e.g. text/gridlines)
    img_text = np.zeros((600, 800, 3), dtype=np.uint8)
    # Add heavy text-like gridlines to trigger Canny edge density (> 0.07)
    for x in range(0, 800, 10):
        cv2.line(img_text, (x, 0), (x, 600), (255, 255, 255), 1)
    for y in range(0, 600, 10):
        cv2.line(img_text, (0, y), (800, y), (255, 255, 255), 1)
        
    assert classify_image(img_text, face_count=0) == "Screenshots/Documents"

def test_similarity_search_math():
    # Define reference embedding
    ref = np.zeros(128, dtype=np.float64)
    ref[0] = 1.0
    
    # Define database embeddings
    item1 = {
        "file_id": 1,
        "embedding": np.zeros(128, dtype=np.float64), # distance = 1.0 (ref[0]=1.0 vs 0.0)
        "absolute_path": "path1.jpg",
        "file_name": "path1.jpg",
        "category_label": "Portraits"
    }
    item1["embedding"][0] = 0.0
    
    item2 = {
        "file_id": 2,
        "embedding": np.zeros(128, dtype=np.float64), # distance = 0.2 (ref[0]=1.0 vs 0.8)
        "absolute_path": "path2.jpg",
        "file_name": "path2.jpg",
        "category_label": "Portraits"
    }
    item2["embedding"][0] = 0.8
    
    item3 = {
        "file_id": 3,
        "embedding": np.zeros(128, dtype=np.float64), # distance = 1.5 (ref[0]=1.0 vs 2.5)
        "absolute_path": "path3.jpg",
        "file_name": "path3.jpg",
        "category_label": "Portraits"
    }
    item3["embedding"][0] = 2.5
    
    db_list = [item1, item2, item3]
    
    # 1. Search with tolerance 0.6. Only item2 is within 0.6 (distance = 0.2)
    matches = search_similar_faces(ref, db_list, tolerance=0.6)
    assert len(matches) == 1
    assert matches[0]["file_id"] == 2
    assert pytest.approx(matches[0]["distance"]) == 0.2
    
    # 2. Search with tolerance 1.1. item2 and item1 should match.
    # item2 (distance 0.2) must come first, followed by item1 (distance 1.0)
    matches_large = search_similar_faces(ref, db_list, tolerance=1.1)
    assert len(matches_large) == 2
    assert matches_large[0]["file_id"] == 2
    assert matches_large[1]["file_id"] == 1
    
    # 3. Empty database test
    assert search_similar_faces(ref, []) == []

def test_thumbnail_generation():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 50, (0, 0, 255), -1)
    
    thumb_bytes = generate_thumbnail(img, size=(100, 100))
    assert thumb_bytes is not None
    assert isinstance(thumb_bytes, bytes)
    
    # Check if PIL can open it
    stream = io.BytesIO(thumb_bytes)
    pil_thumb = Image.open(stream)
    assert pil_thumb.format == "JPEG"
    assert max(pil_thumb.size) == 100

def test_database_face_feedback(temp_db):
    # Add media and embedding
    file_id = temp_db.add_media_file(
        absolute_path="C:/photos/pic1.jpg",
        file_name="pic1.jpg",
        date_modified=123456.0,
        category_label="Portraits",
        thumbnail_blob=b"thumb"
    )
    mock_vector = np.ones(128, dtype=np.float64)
    temp_db.add_face_embedding(file_id, mock_vector)
    
    embeddings = temp_db.get_all_embeddings()
    assert len(embeddings) == 1
    emb_id = embeddings[0]["embedding_id"]
    
    # Store feedback
    ref_path = "C:/photos/ref.jpg"
    temp_db.add_face_feedback(ref_path, emb_id, is_match=1)
    
    feedback_map = temp_db.get_feedback_for_reference(ref_path)
    assert emb_id in feedback_map
    assert feedback_map[emb_id] == 1
    
    # Cascade delete check: delete media file, should delete feedback
    temp_db.delete_media_file("C:/photos/pic1.jpg")
    feedback_map_after = temp_db.get_feedback_for_reference(ref_path)
    assert len(feedback_map_after) == 0

def test_reinforced_learning_search_math():
    # Define reference embedding
    ref = np.zeros(128, dtype=np.float64)
    ref[0] = 1.0
    
    # Database list
    # item1: distance 1.0 (far) - will have positive feedback 👍
    item1 = {
        "embedding_id": 101,
        "file_id": 1,
        "embedding": np.zeros(128, dtype=np.float64),
        "absolute_path": "path1.jpg",
        "file_name": "path1.jpg",
        "category_label": "Portraits"
    }
    item1["embedding"][0] = 0.0
    
    # item2: distance 0.2 (close) - will have negative feedback 👎
    item2 = {
        "embedding_id": 102,
        "file_id": 2,
        "embedding": np.zeros(128, dtype=np.float64),
        "absolute_path": "path2.jpg",
        "file_name": "path2.jpg",
        "category_label": "Portraits"
    }
    item2["embedding"][0] = 0.8
    
    # item3: distance 1.2 from ref (far), but distance 0.2 from item1 (close)
    # Since item1 gets positive feedback 👍, item3 should match via template expansion!
    item3 = {
        "embedding_id": 103,
        "file_id": 3,
        "embedding": np.zeros(128, dtype=np.float64),
        "absolute_path": "path3.jpg",
        "file_name": "path3.jpg",
        "category_label": "Portraits"
    }
    item3["embedding"][0] = 0.2
    
    db_list = [item1, item2, item3]
    
    # Feedback dictionary
    feedbacks = {
        101: 1, # positive 👍
        102: 0  # negative 👎 (blacklist)
    }
    
    # Search with tolerance 0.3
    matches = search_similar_faces(ref, db_list, feedbacks=feedbacks, tolerance=0.3)
    
    matched_ids = [m["embedding_id"] for m in matches]
    assert 102 not in matched_ids
    assert 101 in matched_ids
    assert 103 in matched_ids
    
    # Verify distance calculations
    match1 = [m for m in matches if m["embedding_id"] == 101][0]
    match3 = [m for m in matches if m["embedding_id"] == 103][0]
    assert pytest.approx(match1["distance"]) == 0.0
    assert pytest.approx(match3["distance"]) == 0.2
