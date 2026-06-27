import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from limo_project.engine.vision_core import classify_image, extract_classification_features

# Create mockup image
img_text = np.zeros((600, 800, 3), dtype=np.uint8)
img_text.fill(255) # white background
for row in range(50, 550, 50): # 10 lines of text
    for col in range(50, 750, 80): # words along the line
        cv2.rectangle(img_text, (col, row), (col + 60, row + 10), (0, 0, 0), -1)

# Downscale
h, w = img_text.shape[:2]
scale = 256 / max(h, w)
small_w = int(w * scale)
small_h = int(h * scale)
img_small = cv2.resize(img_text, (small_w, small_h), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

# Match extract_classification_features: Gaussian blur, then Canny.
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)

# Dilate horizontally with a wider kernel to bridge gaps between words
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
text_lines_count = 0
print(f"Total contours: {len(contours)}")
for i, cnt in enumerate(contours):
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect = w_box / h_box
    passed = aspect > 2.0 and 3 <= h_box <= 20 and w_box >= 30
    print(f"Contour {i}: x={x}, y={y}, w={w_box}, h={h_box}, aspect={aspect:.2f}, passed={passed}")
    if passed:
        text_lines_count += 1
print(f"text_lines_count: {text_lines_count}")

features = extract_classification_features(img_text, face_count=0)
print(f"extract_classification_features text_lines_count: {features[6]}")
print(f"classify_image: {classify_image(img_text, face_count=0, features_list=features)}")
