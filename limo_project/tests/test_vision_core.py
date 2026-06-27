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
        category_label="Landscape",
        thumbnail_blob=b"fake_thumbnail"
    )
    assert file_id == 1
    
    # Test checking path
    record = temp_db.get_media_file_by_path("C:/photos/pic1.jpg")
    assert record is not None
    assert record["file_name"] == "pic1.jpg"
    assert record["category_label"] == "Landscape"
    
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
    # 1. Portraits check (face_count strict rules)
    img_portrait = np.zeros((500, 500, 3), dtype=np.uint8)
    assert classify_image(img_portrait, face_count=1) == "Portrait"
    assert classify_image(img_portrait, face_count=2) == "Couple"
    assert classify_image(img_portrait, face_count=3) == "Group"
    
    # 2. Landscapes check (face_count == 0, aspect ratio horizontal >= 1.2, top sky, bottom ground)
    img_landscape = np.zeros((600, 800, 3), dtype=np.uint8) # 800/600 = 1.33
    # Top 30% is blue sky (BGR = 255, 120, 0)
    img_landscape[:180, :] = (255, 120, 0)
    # Bottom 70% is green grass (BGR = 0, 150, 0)
    img_landscape[180:, :] = (0, 150, 0)
    assert classify_image(img_landscape, face_count=0) == "Landscape"
    
    # 3. Documents check (simulated rows of text blocks/words on a white background)
    img_text = np.zeros((600, 800, 3), dtype=np.uint8)
    img_text.fill(255) # white background
    for row in range(50, 550, 50): # 10 lines of text
        for col in range(50, 750, 80): # words along the line
            cv2.rectangle(img_text, (col, row), (col + 60, row + 10), (0, 0, 0), -1)
        
    assert classify_image(img_text, face_count=0) == "Documents"

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
        "category_label": "Portrait"
    }
    item1["embedding"][0] = 0.0
    
    item2 = {
        "file_id": 2,
        "embedding": np.zeros(128, dtype=np.float64), # distance = 0.2 (ref[0]=1.0 vs 0.8)
        "absolute_path": "path2.jpg",
        "file_name": "path2.jpg",
        "category_label": "Portrait"
    }
    item2["embedding"][0] = 0.8
    
    item3 = {
        "file_id": 3,
        "embedding": np.zeros(128, dtype=np.float64), # distance = 1.5 (ref[0]=1.0 vs 2.5)
        "absolute_path": "path3.jpg",
        "file_name": "path3.jpg",
        "category_label": "Portrait"
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
        category_label="Portrait",
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
        "category_label": "Portrait"
    }
    item1["embedding"][0] = 0.0
    
    # item2: distance 0.2 (close) - will have negative feedback 👎
    item2 = {
        "embedding_id": 102,
        "file_id": 2,
        "embedding": np.zeros(128, dtype=np.float64),
        "absolute_path": "path2.jpg",
        "file_name": "path2.jpg",
        "category_label": "Portrait"
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
        "category_label": "Portrait"
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

def test_sorting_and_grouping_logic():
    # Mock files
    item1 = {
        "absolute_path": "C:/photos/vacation/pic1.jpg",
        "file_name": "pic1.jpg",
        "date_modified": 1000.0
    }
    item2 = {
        "absolute_path": "C:/photos/vacation/pic2.jpg",
        "file_name": "pic2.jpg",
        "date_modified": 2000.0
    }
    item3 = {
        "absolute_path": "C:/photos/wedding/abc.jpg",
        "file_name": "abc.jpg",
        "date_modified": 1500.0
    }
    
    items = [item1, item2, item3]
    
    # 1. Test Date sorting
    # Newest first
    items_date_desc = sorted(items, key=lambda x: x["date_modified"], reverse=True)
    assert items_date_desc[0]["file_name"] == "pic2.jpg"
    assert items_date_desc[1]["file_name"] == "abc.jpg"
    assert items_date_desc[2]["file_name"] == "pic1.jpg"
    
    # Oldest first
    items_date_asc = sorted(items, key=lambda x: x["date_modified"])
    assert items_date_asc[0]["file_name"] == "pic1.jpg"
    
    # 2. Test Name sorting
    items_name_asc = sorted(items, key=lambda x: x["file_name"].lower())
    assert items_name_asc[0]["file_name"] == "abc.jpg"
    assert items_name_asc[1]["file_name"] == "pic1.jpg"
    
    # 3. Test Folder grouping logic
    folders = {}
    for item in items:
        parent = os.path.dirname(item["absolute_path"])
        if parent not in folders:
            folders[parent] = []
        folders[parent].append(item)
        
    assert "C:/photos/vacation" in folders
    assert "C:/photos/wedding" in folders
    assert len(folders["C:/photos/vacation"]) == 2
    assert len(folders["C:/photos/wedding"]) == 1


def test_category_reinforcement(temp_db):
    import json
    # 1. Start with empty overrides
    from limo_project.engine.vision_core import CategoryClassifier
    CategoryClassifier.train_model(temp_db)
    assert len(CategoryClassifier._manual_overrides) == 0
    
    # 2. Create mock features
    # Assume image 1 is auto-classified as "Landscape" by heuristics
    # Features (9-d): [face_count, skin_percent, center_skin_percent, edge_density, num_colors, line_count, text_lines_count, sky_columns, nature_percent]
    # For a green-ground blue-sky landscape:
    # sky_columns = 3.0, nature_percent = 0.5, face_count = 0, others low/0
    feat_landscape = [0.0, 0.0, 0.0, 0.02, 30.0, 2.0, 0.0, 3.0, 0.5]
    
    # Add to DB
    path1 = "C:/photos/landscape_override.jpg"
    file_id = temp_db.add_media_file(
        absolute_path=path1,
        file_name="landscape_override.jpg",
        date_modified=123456.0,
        category_label="Landscape",
        thumbnail_blob=b"thumb",
        features=json.dumps(feat_landscape),
        is_manual_category=0
    )
    
    # User manually changes category of path1 to "Uncategorized"
    temp_db.update_media_category(path1, "Uncategorized")
    
    # Verify DB has updated: original_category = "Landscape", category_label = "Uncategorized", is_manual_category = 1
    rec = temp_db.get_media_file_by_path(path1)
    assert rec["category_label"] == "Uncategorized"
    assert rec["original_category"] == "Landscape"
    assert rec["is_manual_category"] == 1
    
    # Train classifier
    CategoryClassifier.train_model(temp_db)
    assert len(CategoryClassifier._manual_overrides) == 1
    assert CategoryClassifier._manual_overrides[0]["category_label"] == "Uncategorized"
    assert CategoryClassifier._manual_overrides[0]["original_category"] == "Landscape"
    
    # 3. Classify a similar image that would normally be classified as "Landscape" by heuristics
    img_similar = np.zeros((600, 800, 3), dtype=np.uint8)
    img_similar[:180, :] = (255, 120, 0) # sky
    img_similar[180:, :] = (0, 150, 0)   # ground
    
    # If we classify without reinforcement (with empty overrides), it should be "Landscape"
    CategoryClassifier._manual_overrides = []
    assert classify_image(img_similar, face_count=0) == "Landscape"
    
    # Now, with reinforcement enabled
    CategoryClassifier.train_model(temp_db)
    # It should dynamically override to "Uncategorized"!
    assert classify_image(img_similar, face_count=0) == "Uncategorized"


def test_group_reinforcement(temp_db):
    import json
    from limo_project.engine.vision_core import CategoryClassifier
    
    # Reset classifier overrides
    CategoryClassifier._manual_overrides = []
    
    # Image 1 features: 2 faces, skin percent high, etc.
    # [face_count, skin_percent, center_skin_percent, edge_density, num_colors, line_count, text_lines_count, sky_columns, nature_percent]
    feat_group = [2.0, 0.25, 0.20, 0.08, 60.0, 5.0, 0.0, 0.0, 0.1]
    
    path1 = "C:/photos/group_photo_1.jpg"
    file_id = temp_db.add_media_file(
        absolute_path=path1,
        file_name="group_photo_1.jpg",
        date_modified=123456.0,
        category_label="Couple", # Auto-classified as Couple because of 2 faces
        thumbnail_blob=b"thumb",
        features=json.dumps(feat_group),
        is_manual_category=0
    )
    
    # User manually changes category of path1 to "Group"
    temp_db.update_media_category(path1, "Group")
    
    # Verify DB has updated
    rec = temp_db.get_media_file_by_path(path1)
    assert rec["category_label"] == "Group"
    assert rec["original_category"] == "Couple"
    
    # Train classifier
    CategoryClassifier.train_model(temp_db)
    assert len(CategoryClassifier._manual_overrides) == 1
    
    # Now, classify another image that has 2 faces and very similar features
    # Without manual override learning, it would be classified as "Couple"
    CategoryClassifier._manual_overrides = []
    img_dummy = np.zeros((400, 400, 3), dtype=np.uint8)
    assert classify_image(img_dummy, face_count=2, features_list=feat_group) == "Couple"
    
    # With manual override learning enabled
    CategoryClassifier.train_model(temp_db)
    # The system should correct it to "Group"!
    assert classify_image(img_dummy, face_count=2, features_list=feat_group) == "Group"
