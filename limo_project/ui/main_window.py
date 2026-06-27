import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QProgressBar, QSlider, QComboBox, QScrollArea,
    QGridLayout, QFrame, QSplitter, QMessageBox, QMenu, QSystemTrayIcon, QButtonGroup,
    QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize, QTimer, QEvent, QPoint, QUrl
from PyQt6.QtGui import QPixmap, QIcon, QAction, QPainter, QColor, QBrush, QPen, QPolygon, QDesktopServices
from limo_project.engine.database import DatabaseManager
from limo_project.engine.vision_core import (
    scan_directory_generator,
    load_and_downscale_image,
    classify_image,
    generate_thumbnail,
    extract_faces_and_embeddings,
    search_similar_faces,
    extract_classification_features,
    CategoryClassifier
)


def get_logo_path():
    return get_resource_path("installer", "logo.png")


def get_icon_path():
    icon_path = get_resource_path("installer", "logo.ico")
    if os.path.exists(icon_path):
        return icon_path
    return get_logo_path()


def get_resource_path(*parts):
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(base_dir, *parts)


class IndexThread(QThread):
    progress_max = pyqtSignal(int)
    progress_step = pyqtSignal(int, str)
    finished = pyqtSignal(int, int)

    def __init__(self, directory, db_manager):
        super().__init__()
        self.directory = directory
        self.db = db_manager
        self._is_running = True
        self._is_paused = False

    def stop(self):
        self._is_running = False
        self._is_paused = False

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def run(self):
        all_files = []
        for file_path in scan_directory_generator(self.directory):
            if not self._is_running:
                return
            all_files.append(file_path)

        total = len(all_files)
        self.progress_max.emit(total)

        added = 0
        skipped = 0

        for idx, file_path in enumerate(all_files):
            while self._is_paused:
                if not self._is_running:
                    return
                self.msleep(100)

            if not self._is_running:
                break

            self.progress_step.emit(idx + 1, file_path)

            try:
                mtime = os.path.getmtime(file_path)
                filename = os.path.basename(file_path)

                cached = self.db.get_media_file_by_path(file_path)
                if cached and cached["date_modified"] == mtime:
                    skipped += 1
                    continue

                img_bgr, w, h = load_and_downscale_image(file_path, max_edge=1024)
                if img_bgr is None:
                    skipped += 1
                    continue

                face_locations, face_encodings = extract_faces_and_embeddings(img_bgr)
                features_vec = extract_classification_features(img_bgr, len(face_locations))
                category = classify_image(img_bgr, len(face_locations), features_list=features_vec)
                thumbnail_blob = generate_thumbnail(img_bgr)

                features_str = json.dumps(features_vec)
                file_id = self.db.add_media_file(
                    absolute_path=file_path,
                    file_name=filename,
                    date_modified=mtime,
                    category_label=category,
                    thumbnail_blob=thumbnail_blob,
                    features=features_str,
                    is_manual_category=0
                )

                self.db.remove_embeddings_for_file(file_id)
                for encoding in face_encodings:
                    self.db.add_face_embedding(file_id, encoding)

                added += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                skipped += 1

        self.finished.emit(added, skipped)


class ImageDropZone(QLabel):
    fileDropped = pyqtSignal(str)
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop Reference Face Image Here\nor Click to Browse")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #4f4f5e;
                border-radius: 6px;
                background-color: #16161b;
                color: #a0a0b0;
                padding: 10px;
                font-weight: bold;
            }
            QLabel:hover {
                border-color: #6c5ce7;
                color: #ffffff;
            }
        """)
        self.setAcceptDrops(True)
        self.setWordWrap(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #6c5ce7;
                    border-radius: 6px;
                    background-color: #242430;
                    color: #ffffff;
                    padding: 10px;
                    font-weight: bold;
                }
            """)

    def dragLeaveEvent(self, event):
        self.reset_style()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.fileDropped.emit(file_path)
        self.reset_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def reset_style(self):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #4f4f5e;
                border-radius: 6px;
                background-color: #16161b;
                color: #a0a0b0;
                padding: 10px;
                font-weight: bold;
            }
        """)


class CategoryPill(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                background-color: #1a1a22;
                border: 1px solid #3a3a4c;
                border-radius: 14px;
                padding: 6px 14px;
                color: #a0a0b0;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #242432;
                border-color: #8a7df0;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #6c5ce7;
                border-color: #6c5ce7;
                color: #ffffff;
            }
        """)


class ViewModePill(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                background-color: #1a1a22;
                border: 1px solid #3a3a4c;
                border-radius: 4px;
                padding: 6px 12px;
                color: #a0a0b0;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #242432;
                border-color: #8a7df0;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #6c5ce7;
                border-color: #6c5ce7;
                color: #ffffff;
            }
        """)


class FolderTile(QWidget):
    doubleClicked = pyqtSignal(str)

    def __init__(self, absolute_path, name, file_count, parent=None):
        super().__init__(parent)
        self.absolute_path = absolute_path
        self.setFixedWidth(160)
        self.setFixedHeight(180)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Folder Icon Draw
        self.icon_label = QLabel(self)
        self.icon_label.setFixedSize(140, 95)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pixmap = QPixmap(90, 70)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#e1b12c"))
        painter.setPen(Qt.PenStyle.NoPen)
        # Folder back tab
        painter.drawRect(5, 5, 30, 12)
        # Folder main body
        painter.drawRect(5, 14, 80, 48)
        painter.end()

        self.icon_label.setPixmap(pixmap)

        self.name_label = QLabel(name, self)
        self.name_label.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 12px;")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics = self.name_label.fontMetrics()
        self.name_label.setText(metrics.elidedText(name, Qt.TextElideMode.ElideRight, 140))

        self.count_label = QLabel(f"{file_count} items", self)
        self.count_label.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.count_label)

        self.setObjectName("FolderTile")
        self.setStyleSheet("""
            QWidget#FolderTile {
                background-color: #1e1e24;
                border: 1px solid #2d2d38;
                border-radius: 8px;
            }
            QWidget#FolderTile:hover {
                background-color: #262632;
                border: 1px solid #e1b12c;
            }
        """)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit(self.absolute_path)


class UpFolderTile(QWidget):
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setFixedHeight(180)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.icon_label = QLabel(self)
        self.icon_label.setFixedSize(140, 95)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pixmap = QPixmap(90, 70)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#7f8c8d"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(5, 5, 30, 12)
        painter.drawRect(5, 14, 80, 48)
        
        # Up Arrow Overlay using native QPoint imports
        painter.setBrush(QColor("#ffffff"))
        points = [
            QPoint(45, 20),
            QPoint(35, 32),
            QPoint(41, 32),
            QPoint(41, 48),
            QPoint(49, 48),
            QPoint(49, 32),
            QPoint(55, 32)
        ]
        painter.drawPolygon(QPolygon(points))
        painter.end()

        self.icon_label.setPixmap(pixmap)

        self.name_label = QLabel(".. (Up One Level)", self)
        self.name_label.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 12px;")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.name_label)
        layout.addStretch()

        self.setObjectName("UpFolderTile")
        self.setStyleSheet("""
            QWidget#UpFolderTile {
                background-color: #1e1e24;
                border: 1px solid #2d2d38;
                border-radius: 8px;
            }
            QWidget#UpFolderTile:hover {
                background-color: #262632;
                border: 1px solid #95a5a6;
            }
        """)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()





class UserGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LIMO User Guide")
        self.setFixedSize(500, 480)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowCloseButtonHint)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #121216;
                color: #e0e0e6;
            }
            QLabel {
                color: #e0e0e6;
            }
            QPushButton {
                background-color: #6c5ce7;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5b4cc4;
            }
            QScrollArea {
                border: 1px solid #272733;
                background-color: #1a1a20;
                border-radius: 6px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Title
        title_label = QLabel("LIMO User Guide & Instructions", self)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Scroll Area for guide content
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        
        content_widget = QWidget(scroll)
        content_widget.setStyleSheet("background-color: transparent;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(12)
        
        guide_text = """
<h3>🔑 Core Operations</h3>
<p><b>1. Select Folder:</b> Click <i>Setup -> Select Media Folder...</i> to choose a directory. LIMO starts indexing and auto-categorizing photos in the background thread.</p>
<p><b>2. Live Sync Dashboard:</b> View scan progress, <b>Pause</b>, or <b>Cancel</b> via the compact bar at the top of the interface.</p>

<h3>📂 Grid Navigation</h3>
<p><b>1. View Modes:</b> Toggle between the flat <b>Library</b> list and yellow <b>Folders</b> directory group view using the header pills.</p>
<p><b>2. Subfolder Navigation:</b> Double-click yellow folder cards to open subdirectories. Double-click the <i>.. (Up One Level)</i> card to go back.</p>
<p><b>3. Sorting:</b> Use the sort dropdown to arrange media by Date (Newest/Oldest) or File Name (A-Z/Z-A).</p>

<h3>🧠 Multi-Photo Selection</h3>
<p><b>1. Selection:</b> Left-click image cards to toggle selection. Selected photos display a glowing purple outline. The left panel shows the active selection count.</p>
<p><b>2. Bulk Categorize:</b> Right-click any selected card, select <i>Change Category</i>, and select a target. This updates all selected items simultaneously and retrains the model.</p>
<p><b>3. Quick Select:</b> Right-clicking an unselected card clears other choices and focuses only on that item.</p>

<h3>🔎 Facial Search & Feedback</h3>
<p><b>1. Drag & Drop Search:</b> Drag a face image into the <i>Facial Search</i> box (or click to browse). Use the slider to adjust Strictness Tolerance.</p>
<p><b>2. Training the Model (👍/👎):</b> Tiles with lower match confidence show thumbs up/down icons:
<ul>
  <li>Click 👍 to add the face to the reference cluster, helping the model learn different angles and lighting.</li>
  <li>Click 👎 to blacklist the face, removing it from results.</li>
</ul>
</p>

<h3>🔄 Reinforcement Overrides</h3>
<p>Changing an auto-classified category of an image teaches the classifier. Normalizing key properties (face counts, textures, text lines, sky percentages, and color distributions), the model dynamically adjusts candidate boundaries, correcting similar photos instantly!</p>
"""
        
        lbl_content = QLabel(guide_text, self)
        lbl_content.setWordWrap(True)
        lbl_content.setTextFormat(Qt.TextFormat.RichText)
        lbl_content.setStyleSheet("font-size: 12px; line-height: 18px; color: #b2bec3;")
        content_layout.addWidget(lbl_content)
        
        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Close Button
        btn_close = QPushButton("Close Guide", self)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About LIMO")
        self.setFixedSize(420, 340)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowCloseButtonHint)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #121216;
                color: #e0e0e6;
            }
            QLabel {
                color: #e0e0e6;
            }
            QPushButton {
                background-color: #6c5ce7;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5b4cc4;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 25)
        layout.setSpacing(15)
        
        # Logo widget
        self.logo_label = QLabel(self)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(get_logo_path()).scaled(
            64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.logo_label.setPixmap(pixmap)
        layout.addWidget(self.logo_label)
        
        # Title
        self.title_label = QLabel("Local Intelligent Media Organizer", self)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Version
        self.version_label = QLabel("v1.0.0 (Offline & Local Edition)", self)
        self.version_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version_label)
        
        # Description
        self.desc_label = QLabel(
            "LIMO is a premium local photo organization utility. It features "
            "completely offline face recognition, dynamic categorization indexing, "
            "and on-the-fly reinforcement model refinement via feedback loops.",
            self
        )
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_label.setStyleSheet("color: #b2bec3; font-size: 12px; line-height: 16px;")
        layout.addWidget(self.desc_label)
        
        # Creator Link
        self.creator_label = QLabel(self)
        self.creator_label.setTextFormat(Qt.TextFormat.RichText)
        self.creator_label.setText("Created by <a href='https://avijitroy.com' style='color: #6c5ce7; text-decoration: none;'>Avijit Roy</a>")
        self.creator_label.setOpenExternalLinks(True)
        self.creator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.creator_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.creator_label)
        
        layout.addStretch()
        
        # Button
        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.accept)
        layout.addWidget(self.btn_close, alignment=Qt.AlignmentFlag.AlignCenter)


class ImageTile(QWidget):
    doubleClicked = pyqtSignal(str)
    feedbackClicked = pyqtSignal(str, int, int) # file_path, embedding_id, is_match
    categoryChanged = pyqtSignal(str, str)      # file_path, new_category
    selectionChanged = pyqtSignal(str, bool)     # file_path, is_selected
    rightClicked = pyqtSignal(str)              # file_path

    def __init__(self, file_path, name, category, thumbnail_blob, distance=None, embedding_id=None, is_selected=False, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.embedding_id = embedding_id
        self.is_selected = is_selected
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        self.has_feedback = (distance is not None and distance > 0.40 and embedding_id is not None)

        self.setFixedWidth(160)
        if self.has_feedback:
            self.setFixedHeight(235)
        else:
            self.setFixedHeight(210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.img_label = QLabel(self)
        self.img_label.setFixedSize(148, 120)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background-color: #0c0c0e; border-radius: 4px; border: 1px solid #1a1a20;")

        if thumbnail_blob:
            pixmap = QPixmap()
            pixmap.loadFromData(thumbnail_blob)
            scaled_pixmap = pixmap.scaled(
                148, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.img_label.setPixmap(scaled_pixmap)
        else:
            self.img_label.setText("No Thumbnail")
            self.img_label.setStyleSheet("background-color: #0c0c0e; color: #505060; border-radius: 4px;")

        self.name_label = QLabel(name, self)
        self.name_label.setToolTip(file_path)
        self.name_label.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 12px;")
        metrics = self.name_label.fontMetrics()
        elided_name = metrics.elidedText(name, Qt.TextElideMode.ElideRight, 140)
        self.name_label.setText(elided_name)

        self.cat_label = QLabel(category, self)
        self.cat_label.setStyleSheet("color: #8a8a9e; font-size: 11px;")

        layout.addWidget(self.img_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.cat_label)

        if distance is not None:
            confidence = int((1.0 - distance) * 100)
            self.dist_label = QLabel(f"Match: {confidence}%", self)
            self.dist_label.setStyleSheet("color: #00b894; font-size: 11px; font-weight: bold;")
            layout.addWidget(self.dist_label)
            
            if self.has_feedback:
                feedback_container = QWidget(self)
                feedback_layout = QHBoxLayout(feedback_container)
                feedback_layout.setContentsMargins(0, 2, 0, 2)
                feedback_layout.setSpacing(4)
                
                lbl_ask = QLabel("Match?", self)
                lbl_ask.setStyleSheet("font-size: 10px; color: #a0a0b0;")
                feedback_layout.addWidget(lbl_ask)
                
                btn_yes = QPushButton("👍", self)
                btn_yes.setToolTip("Accept as correct match")
                btn_yes.setStyleSheet("""
                    QPushButton {
                        background-color: #1b3a24;
                        border: 1px solid #00b894;
                        border-radius: 9px;
                        padding: 0px;
                        font-size: 10px;
                        min-height: 18px;
                        max-height: 18px;
                        min-width: 26px;
                        max-width: 26px;
                    }
                    QPushButton:hover {
                        background-color: #00b894;
                    }
                """)
                btn_yes.clicked.connect(lambda: self.feedbackClicked.emit(self.file_path, self.embedding_id, 1))
                feedback_layout.addWidget(btn_yes)
                
                btn_no = QPushButton("👎", self)
                btn_no.setToolTip("Reject match")
                btn_no.setStyleSheet("""
                    QPushButton {
                        background-color: #3d1b1b;
                        border: 1px solid #d63031;
                        border-radius: 9px;
                        padding: 0px;
                        font-size: 10px;
                        min-height: 18px;
                        max-height: 18px;
                        min-width: 26px;
                        max-width: 26px;
                    }
                    QPushButton:hover {
                        background-color: #d63031;
                    }
                """)
                btn_no.clicked.connect(lambda: self.feedbackClicked.emit(self.file_path, self.embedding_id, 0))
                feedback_layout.addWidget(btn_no)
                
                layout.addWidget(feedback_container)
        else:
            layout.addStretch()

        self.setObjectName("ImageTile")
        self.update_style()

        # Context Menu setup
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet("""
                QWidget#ImageTile {
                    background-color: #242435;
                    border: 2px solid #8a7df0;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget#ImageTile {
                    background-color: #1e1e24;
                    border: 1px solid #2d2d38;
                    border-radius: 6px;
                }
                QWidget#ImageTile:hover {
                    background-color: #262632;
                    border: 1px solid #8a7df0;
                }
            """)

    def setSelected(self, selected):
        self.is_selected = selected
        self.update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Toggle selection
            self.setSelected(not self.is_selected)
            self.selectionChanged.emit(self.file_path, self.is_selected)
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit(self.file_path)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit(self.file_path)

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a20;
                border: 1px solid #272733;
                color: #e0e0e6;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background-color: #6c5ce7;
                color: #ffffff;
            }
        """)
        
        open_action = QAction("Open Photo", self)
        open_action.triggered.connect(lambda: self.doubleClicked.emit(self.file_path))
        menu.addAction(open_action)
        
        cat_menu = QMenu("Change Category", menu)
        cat_menu.setStyleSheet(menu.styleSheet())
        
        categories = ["Portrait", "Couple", "Group", "Landscape", "Documents", "Uncategorized"]
        for cat in categories:
            act = QAction(cat, self)
            act.triggered.connect(lambda checked, c=cat: self.categoryChanged.emit(self.file_path, c))
            cat_menu.addAction(act)
            
        menu.addMenu(cat_menu)
        menu.exec(self.mapToGlobal(pos))


class MainWindow(QMainWindow):
    def __init__(self, db_path="face_index.db"):
        super().__init__()
        self.db = DatabaseManager(db_path)
        CategoryClassifier.train_model(self.db)
        self.index_thread = None
        self.ref_embedding = None
        self.ref_file_path = None
        self.selected_directory = ""
        self.is_background_sync = False
        
        self.view_mode = "library"
        self.current_folder = None
        
        self.categories_list = ["All", "Portrait", "Couple", "Group", "Landscape", "Documents", "Uncategorized"]
        self.selected_category = "All"
        self.selected_files = set()  # Track multi-selected paths
        
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.load_next_batch)
        
        self.loading_timer_folders = QTimer(self)
        self.loading_timer_folders.timeout.connect(self.load_next_folder_batch)
        
        self.live_refresh_timer = QTimer(self)
        self.live_refresh_timer.setSingleShot(True)
        self.live_refresh_timer.timeout.connect(self.trigger_search)

        self.setWindowTitle("Local Intelligent Media Organizer (LIMO)")
        self.setWindowIcon(QIcon(get_icon_path()))
        self.resize(1100, 750)
        
        self.setup_styles()
        self.init_menu()
        self.init_ui()
        self.init_system_tray()
        
        self.load_all_media_from_db()

        self.auto_sync_timer = QTimer(self)
        self.auto_sync_timer.timeout.connect(self.trigger_auto_sync)
        self.auto_sync_timer.start(180000)

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121216;
            }
            QWidget {
                font-family: 'Segoe UI', -apple-system, sans-serif;
                color: #e0e0e6;
                font-size: 13px;
            }
            QFrame#ControlPanel, QFrame#ResultsPanel {
                background-color: #1a1a20;
                border: 1px solid #272733;
                border-radius: 8px;
            }
            QLineEdit {
                background-color: #111115;
                border: 1px solid #3a3a4c;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid #6c5ce7;
            }
            QPushButton {
                background-color: #3b3b4a;
                border: none;
                border-radius: 4px;
                color: #ffffff;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4b4b5c;
            }
            QPushButton:pressed {
                background-color: #272733;
            }
            QProgressBar {
                border: 1px solid #2d2d38;
                border-radius: 4px;
                background-color: #111115;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6c5ce7, stop:1 #9c8eff);
                border-radius: 3px;
            }
            QComboBox {
                background-color: #111115;
                border: 1px solid #3a3a4c;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #121216;
                width: 10px;
                margin: 0px 0 0px 0;
            }
            QScrollBar::handle:vertical {
                background: #2a2a35;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3f3f50;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QMenuBar {
                background-color: #1a1a20;
                border-bottom: 1px solid #272733;
                color: #e0e0e6;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #2a2a35;
                border-radius: 4px;
            }
            QMenu {
                background-color: #1a1a20;
                border: 1px solid #272733;
                color: #e0e0e6;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background-color: #6c5ce7;
                color: #ffffff;
            }
        """)

    def init_menu(self):
        menubar = self.menuBar()
        setup_menu = menubar.addMenu("Setup")

        select_folder_action = QAction("Select Media Folder...", self)
        select_folder_action.triggered.connect(self.browse_directory)
        setup_menu.addAction(select_folder_action)

        clear_cache_action = QAction("Clear Index Cache...", self)
        clear_cache_action.triggered.connect(self.clear_index_data)
        setup_menu.addAction(clear_cache_action)

        setup_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        setup_menu.addAction(exit_action)

        # Help & User Guide Menu
        help_menu = menubar.addMenu("Help")
        
        guide_action = QAction("User Guide...", self)
        guide_action.triggered.connect(self.show_user_guide_dialog)
        help_menu.addAction(guide_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("About LIMO...", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # Sync Dashboard layout (Progress + Pause + Cancel)
        self.sync_dashboard = QWidget(self)
        self.sync_dashboard.setVisible(False)
        sync_layout = QHBoxLayout(self.sync_dashboard)
        sync_layout.setContentsMargins(0, 0, 0, 0)
        sync_layout.setSpacing(8)

        self.progress_bar = QProgressBar(self)
        sync_layout.addWidget(self.progress_bar, stretch=1)

        self.btn_pause_sync = QPushButton("Pause", self)
        self.btn_pause_sync.setStyleSheet("""
            QPushButton {
                background-color: #d35400;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.btn_pause_sync.clicked.connect(self.toggle_pause_sync)
        sync_layout.addWidget(self.btn_pause_sync)

        self.btn_cancel_sync = QPushButton("Cancel", self)
        self.btn_cancel_sync.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        self.btn_cancel_sync.clicked.connect(self.cancel_sync)
        sync_layout.addWidget(self.btn_cancel_sync)

        self.sync_dashboard.setFixedHeight(45)
        main_layout.addWidget(self.sync_dashboard, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        
        # Left Panel - Search Control Core
        left_panel = QFrame(self)
        left_panel.setObjectName("ControlPanel")
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)

        lbl_search_header = QLabel("FACIAL SEARCH", self)
        lbl_search_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #6c5ce7;")
        left_layout.addWidget(lbl_search_header)

        left_layout.addWidget(QLabel("Reference Face Image:", self))
        self.drop_zone = ImageDropZone(self)
        self.drop_zone.setFixedHeight(180)
        self.drop_zone.fileDropped.connect(self.load_reference_image)
        self.drop_zone.clicked.connect(self.select_reference_image)
        left_layout.addWidget(self.drop_zone)

        self.btn_clear_face = QPushButton("Clear Face Search", self)
        self.btn_clear_face.setEnabled(False)
        self.btn_clear_face.clicked.connect(self.clear_reference_image)
        left_layout.addWidget(self.btn_clear_face)

        self.lbl_tolerance = QLabel("Strictness Tolerance: 0.55", self)
        left_layout.addWidget(self.lbl_tolerance)
        
        self.slider_tolerance = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_tolerance.setRange(5, 100)
        self.slider_tolerance.setValue(55)
        self.slider_tolerance.valueChanged.connect(self.on_tolerance_changed)
        left_layout.addWidget(self.slider_tolerance)

        left_layout.addStretch()
        
        self.status_label = QLabel("Setup a folder using 'Setup -> Select Media Folder...' to begin.", self)
        self.status_label.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left_panel)

        # Right Panel - Category Pills + Dynamic Grid View
        right_panel = QFrame(self)
        right_panel.setObjectName("ResultsPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        # Controls Header Layout (Pills + Sorting + View Toggles)
        header_controls_layout = QHBoxLayout()
        header_controls_layout.setContentsMargins(0, 0, 0, 0)
        header_controls_layout.setSpacing(10)

        self.category_container = QWidget(self)
        self.category_layout = QHBoxLayout(self.category_container)
        self.category_layout.setContentsMargins(0, 0, 0, 0)
        self.category_layout.setSpacing(8)
        self.setup_category_filters()
        header_controls_layout.addWidget(self.category_container)

        self.lbl_grid_loading = QLabel(self)
        self.lbl_grid_loading.setStyleSheet("color: #8a7df0; font-weight: bold; font-size: 12px; margin-left: 10px;")
        self.lbl_grid_loading.setVisible(False)
        header_controls_layout.addWidget(self.lbl_grid_loading)

        header_controls_layout.addStretch()

        # Sorting ComboBox Selector
        self.combo_sort = QComboBox(self)
        self.combo_sort.addItems([
            "Sort: Date (Newest)",
            "Sort: Date (Oldest)",
            "Sort: Name (A-Z)",
            "Sort: Name (Z-A)"
        ])
        self.combo_sort.currentTextChanged.connect(self.trigger_search)
        self.combo_sort.setFixedWidth(170)
        header_controls_layout.addWidget(self.combo_sort)

        # View Mode toggle button group
        self.view_mode_group = QButtonGroup(self)
        self.view_mode_group.setExclusive(True)

        self.btn_view_library = ViewModePill("Library", self)
        self.btn_view_library.setChecked(True)
        self.btn_view_library.clicked.connect(self.set_library_view_mode)
        self.view_mode_group.addButton(self.btn_view_library)
        header_controls_layout.addWidget(self.btn_view_library)

        self.btn_view_folder = ViewModePill("Folders", self)
        self.btn_view_folder.clicked.connect(self.set_folder_view_mode)
        self.view_mode_group.addButton(self.btn_view_folder)
        header_controls_layout.addWidget(self.btn_view_folder)

        right_layout.addLayout(header_controls_layout)

        # Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        
        self.grid_widget = QWidget(self)
        self.grid_widget.setStyleSheet("background-color: transparent;")
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scroll_area.setWidget(self.grid_widget)
        right_layout.addWidget(self.scroll_area)

        splitter.addWidget(right_panel)
        
        splitter.setSizes([280, 820])
        main_layout.addWidget(splitter, 1)

    def setup_category_filters(self):
        while self.category_layout.count():
            item = self.category_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if len(self.categories_list) <= 10:
            self.pill_group = QButtonGroup(self)
            self.pill_group.setExclusive(True)
            
            for cat in self.categories_list:
                pill = CategoryPill(cat, self)
                if cat == self.selected_category:
                    pill.setChecked(True)
                
                pill.clicked.connect(self.on_category_pill_clicked)
                self.pill_group.addButton(pill)
                self.category_layout.addWidget(pill)
        else:
            combo = QComboBox(self)
            combo.addItems(self.categories_list)
            combo.setCurrentText(self.selected_category)
            combo.currentTextChanged.connect(self.on_category_combo_changed)
            self.category_layout.addWidget(combo)

    def on_category_pill_clicked(self):
        sender = self.sender()
        if sender:
            self.selected_category = sender.text()
            self.trigger_search()

    def on_category_combo_changed(self, text):
        self.selected_category = text
        self.trigger_search()

    def set_library_view_mode(self):
        self.view_mode = "library"
        self.current_folder = None
        self.trigger_search()

    def set_folder_view_mode(self):
        self.view_mode = "folder"
        self.current_folder = None
        self.trigger_search()

    def enter_folder(self, path):
        self.current_folder = path
        self.trigger_search()

    def exit_folder(self):
        self.current_folder = None
        self.trigger_search()

    def init_system_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(get_icon_path()))
        self.tray_icon.setToolTip("Local Intelligent Media Organizer (LIMO)")

        tray_menu = QMenu()
        restore_action = QAction("Restore LIMO", self)
        restore_action.triggered.connect(self.restore_from_tray)
        tray_menu.addAction(restore_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.exit_application)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.restore_from_tray()

    def restore_from_tray(self):
        self.showNormal()
        self.activateWindow()

    def exit_application(self):
        self.tray_icon.hide()
        QApplication.quit()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMinimized():
                self.hide()
                self.tray_icon.showMessage(
                    "LIMO Minimized",
                    "LIMO is indexing and searching in the background.",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
                event.accept()
                return
        super().changeEvent(event)

    def closeEvent(self, event):
        if self.index_thread and self.index_thread.isRunning():
            self.index_thread.stop()
            self.index_thread.wait()
        self.tray_icon.hide()
        event.accept()

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Media Folder to Scan")
        if directory:
            self.selected_directory = directory
            self.status_label.setText(f"Folder selected: {os.path.basename(directory)}")
            self.start_indexing(is_background=False)

    def trigger_auto_sync(self):
        if self.selected_directory and os.path.exists(self.selected_directory):
            if not self.index_thread or not self.index_thread.isRunning():
                self.start_indexing(is_background=True)

    def start_indexing(self, is_background=False):
        directory = self.selected_directory
        if not directory or not os.path.exists(directory):
            return

        self.is_background_sync = is_background

        if not is_background:
            self.progress_bar.setValue(0)
            self.sync_dashboard.setVisible(True)
            self.btn_pause_sync.setText("Pause")
            self.btn_pause_sync.setEnabled(True)
            self.btn_cancel_sync.setEnabled(True)
            self.status_label.setText("Scanning media directory...")
        else:
            self.status_label.setText("Background checking for updates...")

        self.index_thread = IndexThread(directory, self.db)
        self.index_thread.progress_max.connect(self.progress_bar.setMaximum)
        self.index_thread.progress_step.connect(self.on_index_step)
        self.index_thread.finished.connect(self.on_index_finished)
        self.index_thread.start()

    def toggle_pause_sync(self):
        if not self.index_thread or not self.index_thread.isRunning():
            return

        if not self.index_thread._is_paused:
            self.index_thread.pause()
            self.btn_pause_sync.setText("Resume")
            self.status_label.setText("Sync paused.")
        else:
            self.index_thread.resume()
            self.btn_pause_sync.setText("Pause")
            self.status_label.setText("Scanning media directory...")

    def cancel_sync(self):
        if not self.index_thread or not self.index_thread.isRunning():
            return

        self.btn_pause_sync.setEnabled(False)
        self.btn_cancel_sync.setEnabled(False)
        self.index_thread.stop()
        self.status_label.setText("Sync cancelled. Cleaning up thread...")

    def show_user_guide_dialog(self):
        dialog = UserGuideDialog(self)
        dialog.exec()

    def show_about_dialog(self):
        dialog = AboutDialog(self)
        dialog.exec()

    def on_index_step(self, step, filepath):
        if not self.is_background_sync:
            self.progress_bar.setValue(step)
            self.status_label.setText(f"Scanning: {os.path.basename(filepath)}")
            
        # Throttled live update of the grid while scanning
        if not self.live_refresh_timer.isActive():
            self.live_refresh_timer.start(1200)

    def on_index_finished(self, added, skipped):
        self.sync_dashboard.setVisible(False)
        self.live_refresh_timer.stop()
        
        if not self.is_background_sync:
            self.status_label.setText("Sync complete.")
            QMessageBox.information(
                self,
                "Sync Completed",
                f"Directory scan completed.\nAdded/Updated: {added} files.\nSkipped (Up-to-date): {skipped} files."
            )
        else:
            if added > 0:
                self.status_label.setText(f"Background check complete. Added {added} new file(s).")
                self.tray_icon.showMessage(
                    "LIMO Index Updated",
                    f"Background sync added {added} new photo(s).",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
            else:
                self.status_label.setText("Background check complete. No changes.")

        self.trigger_search()

    def clear_index_data(self):
        reply = QMessageBox.question(
            self,
            "Clear Cache Data",
            "Are you sure you want to completely clear the local index and cache database?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.clear_cache()
            self.clear_reference_image()
            self.clear_grid()
            self.status_label.setText("Cache database cleared.")
            QMessageBox.information(self, "Cache Cleared", "The database has been reset successfully.")

    def select_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Face Photo", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if file_path:
            self.load_reference_image(file_path)

    def load_reference_image(self, file_path):
        if not os.path.exists(file_path):
            return

        self.status_label.setText("Extracting face encoding...")
        img_bgr, _, _ = load_and_downscale_image(file_path, max_edge=1024)
        if img_bgr is None:
            self.status_label.setText("Could not load reference image.")
            return

        face_locations, encodings = extract_faces_and_embeddings(img_bgr)

        if not encodings:
            self.status_label.setText("No face detected in reference photo.")
            QMessageBox.warning(self, "No Face Found", "Could not detect any face in the reference photo.")
            return

        self.ref_file_path = file_path
        self.ref_embedding = encodings[0]
        self.btn_clear_face.setEnabled(True)

        pixmap = QPixmap(file_path)
        scaled = pixmap.scaled(
            self.drop_zone.width() - 8,
            self.drop_zone.height() - 8,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.drop_zone.setPixmap(scaled)
        
        self.status_label.setText("Reference face loaded.")
        self.trigger_search()

    def clear_reference_image(self):
        self.ref_file_path = None
        self.ref_embedding = None
        self.btn_clear_face.setEnabled(False)
        self.drop_zone.setPixmap(QPixmap())
        self.drop_zone.setText("Drag & Drop Reference Face Image Here\nor Click to Browse")
        self.status_label.setText("Face search cleared.")
        self.trigger_search()

    def on_tolerance_changed(self, value):
        tolerance = value / 100.0
        self.lbl_tolerance.setText(f"Strictness Tolerance: {tolerance:.2f}")
        if self.ref_embedding is not None:
            self.trigger_search()

    def trigger_search(self):
        # Stop any active incremental loading timers
        if hasattr(self, 'loading_timer') and self.loading_timer.isActive():
            self.loading_timer.stop()
        if hasattr(self, 'loading_timer_folders') and self.loading_timer_folders.isActive():
            self.loading_timer_folders.stop()
            
        self.selected_files.clear()  # Clear selection on search
        self.lbl_grid_loading.setText("Searching...")
        self.lbl_grid_loading.setVisible(True)
        
        # Schedule the search & sorting computation 10ms later to keep UI fluid
        QTimer.singleShot(10, self.perform_search_and_populate)

    def perform_search_and_populate(self):
        try:
            category = self.selected_category
            
            # Retrieve items
            if self.ref_embedding is not None:
                db_embeddings = self.db.get_all_embeddings()
                feedbacks = self.db.get_feedback_for_reference(self.ref_file_path)
                tolerance = self.slider_tolerance.value() / 100.0
                items = search_similar_faces(self.ref_embedding, db_embeddings, feedbacks=feedbacks, tolerance=tolerance)
                
                if category.lower() != 'all':
                    items = [m for m in items if m["category_label"] == category]
            else:
                items = self.db.get_media_files(category=category)

            # Sort
            sort_type = self.combo_sort.currentText()
            if sort_type == "Sort: Date (Newest)":
                items.sort(key=lambda x: x.get("date_modified", 0), reverse=True)
            elif sort_type == "Sort: Date (Oldest)":
                items.sort(key=lambda x: x.get("date_modified", 0))
            elif sort_type == "Sort: Name (A-Z)":
                items.sort(key=lambda x: x.get("file_name", "").lower())
            elif sort_type == "Sort: Name (Z-A)":
                items.sort(key=lambda x: x.get("file_name", "").lower(), reverse=True)

            # Populate dynamically (starts the incremental timers)
            if self.view_mode == "library":
                self.populate_grid(items)
            else:
                if self.current_folder is None:
                    folders_dict = {}
                    for item in items:
                        parent_path = os.path.dirname(item["absolute_path"])
                        if parent_path not in folders_dict:
                            folders_dict[parent_path] = []
                        folders_dict[parent_path].append(item)
                    
                    folder_paths = list(folders_dict.keys())
                    is_reverse = "Z-A" in sort_type or "Newest" in sort_type
                    folder_paths.sort(key=lambda x: os.path.basename(x).lower(), reverse=is_reverse)
                    
                    self.populate_folder_grid(folders_dict, folder_paths)
                else:
                    folder_items = [item for item in items if os.path.dirname(item["absolute_path"]) == self.current_folder]
                    self.populate_grid(folder_items, show_up_level=True)
        except Exception as e:
            print(f"Error performing search: {e}")
            self.lbl_grid_loading.setVisible(False)

    def on_feedback_received(self, file_path, embedding_id, is_match):
        if not self.ref_file_path:
            return
            
        self.db.add_face_feedback(self.ref_file_path, embedding_id, is_match)
        action = "confirmed as match 👍" if is_match == 1 else "rejected as match 👎"
        self.status_label.setText(f"Feedback recorded: Face {action}. Reinforcing search model...")
        self.trigger_search()

    def on_category_changed_manually(self, file_path, new_category):
        targets = list(self.selected_files)
        if not targets or file_path not in self.selected_files:
            targets = [file_path]
            
        for path in targets:
            self.db.update_media_category(path, new_category)
            
        CategoryClassifier.train_model(self.db)
        self.selected_files.clear()
        self.status_label.setText(f"Category of {len(targets)} photo(s) modified to '{new_category}'. ML model reinforced.")
        self.trigger_search()

    def on_tile_selection_changed(self, file_path, is_selected):
        if is_selected:
            self.selected_files.add(file_path)
        else:
            self.selected_files.discard(file_path)
            
        if self.selected_files:
            self.status_label.setText(f"Selected {len(self.selected_files)} photo(s) to categorize.")
        else:
            self.status_label.setText("Ready.")

    def on_tile_right_clicked(self, file_path):
        if file_path not in self.selected_files:
            self.select_single_file_for_menu(file_path)

    def select_single_file_for_menu(self, file_path):
        self.selected_files.clear()
        self.selected_files.add(file_path)
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, ImageTile):
                widget.setSelected(widget.file_path == file_path)

    def load_all_media_from_db(self):
        self.trigger_search()

    def clear_grid(self):
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def populate_grid(self, items, show_up_level=False):
        self.clear_grid()
        self.loading_items = items
        self.loading_index = 0
        self.loading_show_up_level = show_up_level
        self.selected_files.clear()
        
        self.lbl_grid_loading.setText("Loading...")
        self.lbl_grid_loading.setVisible(True)
        self.loading_timer.start(15) # Render batch every 15ms

    def load_next_batch(self):
        if not self.loading_items or self.loading_index >= len(self.loading_items):
            self.loading_timer.stop()
            self.lbl_grid_loading.setVisible(False)
            total = len(self.loading_items)
            if self.view_mode == "library":
                if self.ref_embedding is not None:
                    self.status_label.setText(f"Search results: {total} matching face(s) found.")
                else:
                    self.status_label.setText(f"Showing {total} file(s).")
            else:
                self.status_label.setText(f"Showing {total} file(s) in '{os.path.basename(self.current_folder)}'.")
            return

        batch_size = 15
        end_idx = min(self.loading_index + batch_size, len(self.loading_items))
        
        width = self.scroll_area.viewport().width()
        if width <= 0:
            width = 820
        tile_width = 175
        columns = max(1, width // tile_width)
        
        start_pos = 1 if self.loading_show_up_level else 0
        if self.loading_show_up_level and self.loading_index == 0:
            up_tile = UpFolderTile(self)
            up_tile.doubleClicked.connect(self.exit_folder)
            self.grid_layout.addWidget(up_tile, 0, 0)
            
        for idx in range(self.loading_index, end_idx):
            item = self.loading_items[idx]
            path = item["absolute_path"]
            is_sel = path in self.selected_files
            tile = ImageTile(
                file_path=path,
                name=item["file_name"],
                category=item["category_label"],
                thumbnail_blob=item["thumbnail_blob"],
                distance=item.get("distance", None),
                embedding_id=item.get("embedding_id", None),
                is_selected=is_sel,
                parent=self
            )
            tile.doubleClicked.connect(self.open_file_in_explorer)
            tile.feedbackClicked.connect(self.on_feedback_received)
            tile.categoryChanged.connect(self.on_category_changed_manually)
            tile.selectionChanged.connect(self.on_tile_selection_changed)
            tile.rightClicked.connect(self.on_tile_right_clicked)
            
            grid_pos = start_pos + idx
            row = grid_pos // columns
            col = grid_pos % columns
            self.grid_layout.addWidget(tile, row, col)
            
        self.loading_index = end_idx
        self.lbl_grid_loading.setText(f"Loading {self.loading_index}/{len(self.loading_items)}...")

    def populate_folder_grid(self, folders_dict, sorted_paths):
        self.clear_grid()
        if not sorted_paths:
            return
            
        self.loading_items = sorted_paths
        self.loading_index = 0
        self.loading_folders_dict = folders_dict
        self.selected_files.clear()
        
        self.lbl_grid_loading.setText("Loading folders...")
        self.lbl_grid_loading.setVisible(True)
        self.loading_timer_folders.start(15)

    def load_next_folder_batch(self):
        if not self.loading_items or self.loading_index >= len(self.loading_items):
            self.loading_timer_folders.stop()
            self.lbl_grid_loading.setVisible(False)
            self.status_label.setText(f"Showing {len(self.loading_items)} folders.")
            return

        batch_size = 15
        end_idx = min(self.loading_index + batch_size, len(self.loading_items))
        
        width = self.scroll_area.viewport().width()
        if width <= 0:
            width = 820
        tile_width = 175
        columns = max(1, width // tile_width)
        
        for idx in range(self.loading_index, end_idx):
            path = self.loading_items[idx]
            name = os.path.basename(path)
            if not name:
                name = path
            count = len(self.loading_folders_dict[path])
            
            tile = FolderTile(absolute_path=path, name=name, file_count=count, parent=self)
            tile.doubleClicked.connect(self.enter_folder)
            
            row = idx // columns
            col = idx % columns
            self.grid_layout.addWidget(tile, row, col)
            
        self.loading_index = end_idx
        self.lbl_grid_loading.setText(f"Loading {self.loading_index}/{len(self.loading_items)}...")

    def rearrange_grid(self):
        width = self.scroll_area.viewport().width()
        if width <= 0:
            return
        tile_width = 175
        columns = max(1, width // tile_width)

        tiles = []
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            widget = item.widget()
            if widget:
                tiles.append(widget)

        for idx, tile in enumerate(tiles):
            self.grid_layout.removeWidget(tile)
            row = idx // columns
            col = idx % columns
            self.grid_layout.addWidget(tile, row, col)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rearrange_grid()

    def open_file_in_explorer(self, file_path):
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", "The media file could not be located on the storage drive.")
            return
        try:
            os.startfile(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Execution Failure", f"Failed to open file: {e}")
