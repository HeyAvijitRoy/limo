import os
import sys
import numpy as np
import cv2
from PIL import Image
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QProgressBar, QSlider, QComboBox, QScrollArea,
    QGridLayout, QFrame, QSplitter, QMessageBox, QMenu, QSystemTrayIcon, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize, QTimer, QEvent
from PyQt6.QtGui import QPixmap, QIcon, QAction, QPainter, QColor, QBrush, QPen
from limo_project.engine.database import DatabaseManager
from limo_project.engine.vision_core import (
    scan_directory_generator,
    load_and_downscale_image,
    classify_image,
    generate_thumbnail,
    extract_faces_and_embeddings,
    search_similar_faces
)

class IndexThread(QThread):
    progress_max = pyqtSignal(int)
    progress_step = pyqtSignal(int, str)
    finished = pyqtSignal(int, int)

    def __init__(self, directory, db_manager):
        super().__init__()
        self.directory = directory
        self.db = db_manager
        self._is_running = True

    def stop(self):
        self._is_running = False

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
                category = classify_image(img_bgr, len(face_locations))
                thumbnail_blob = generate_thumbnail(img_bgr)

                file_id = self.db.add_media_file(
                    absolute_path=file_path,
                    file_name=filename,
                    date_modified=mtime,
                    category_label=category,
                    thumbnail_blob=thumbnail_blob
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
                border-color: #6c5ce7;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #6c5ce7;
                border-color: #6c5ce7;
                color: #ffffff;
            }
        """)


class ImageTile(QWidget):
    doubleClicked = pyqtSignal(str)
    feedbackClicked = pyqtSignal(str, int, int) # file_path, embedding_id, is_match

    def __init__(self, file_path, name, category, thumbnail_blob, distance=None, embedding_id=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.embedding_id = embedding_id
        
        # If low confidence match (< 60%, i.e. distance > 0.40), render feedback controls
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
            
            # Interactive reinforced learning feedback layout
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

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e24;
                border: 1px solid #2d2d38;
                border-radius: 6px;
            }
            QWidget:hover {
                background-color: #262630;
                border: 1px solid #6c5ce7;
            }
        """)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit(self.file_path)


class MainWindow(QMainWindow):
    def __init__(self, db_path="face_index.db"):
        super().__init__()
        self.db = DatabaseManager(db_path)
        self.index_thread = None
        self.ref_embedding = None
        self.ref_file_path = None
        self.selected_directory = ""
        self.is_background_sync = False
        
        self.categories_list = ["All", "Portraits", "Landscapes", "Screenshots/Documents", "Uncategorized"]
        self.selected_category = "All"

        self.setWindowTitle("Local Intelligent Media Organizer (LIMO)")
        self.resize(1100, 750)
        
        self.setup_styles()
        self.init_menu()
        self.init_ui()
        self.init_system_tray()
        
        self.load_all_media_from_db()

        self.auto_sync_timer = QTimer(self)
        self.auto_sync_timer.timeout.connect(self.trigger_auto_sync)
        self.auto_sync_timer.start(180000) # 3 minutes

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

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

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

        self.category_container = QWidget(self)
        self.category_layout = QHBoxLayout(self.category_container)
        self.category_layout.setContentsMargins(0, 0, 0, 0)
        self.category_layout.setSpacing(8)
        
        self.setup_category_filters()
        right_layout.addWidget(self.category_container)

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
        main_layout.addWidget(splitter)

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
            
            self.category_layout.addStretch()
        else:
            lbl_filter = QLabel("Category Filter:", self)
            lbl_filter.setStyleSheet("font-weight: bold; color: #a0a0b0;")
            self.category_layout.addWidget(lbl_filter)

            combo = QComboBox(self)
            combo.addItems(self.categories_list)
            combo.setCurrentText(self.selected_category)
            combo.currentTextChanged.connect(self.on_category_combo_changed)
            self.category_layout.addWidget(combo)
            self.category_layout.addStretch()

    def on_category_pill_clicked(self):
        sender = self.sender()
        if sender:
            self.selected_category = sender.text()
            self.trigger_search()

    def on_category_combo_changed(self, text):
        self.selected_category = text
        self.trigger_search()

    def init_system_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#6c5ce7"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 28, 28)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(10, 10, 12, 12)
        painter.end()
        
        self.tray_icon.setIcon(QIcon(pixmap))
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
            self.progress_bar.setVisible(True)
            self.status_label.setText("Scanning media directory...")
        else:
            self.status_label.setText("Background checking for updates...")

        self.index_thread = IndexThread(directory, self.db)
        self.index_thread.progress_max.connect(self.progress_bar.setMaximum)
        self.index_thread.progress_step.connect(self.on_index_step)
        self.index_thread.finished.connect(self.on_index_finished)
        self.index_thread.start()

    def on_index_step(self, step, filepath):
        if not self.is_background_sync:
            self.progress_bar.setValue(step)
            self.status_label.setText(f"Scanning: {os.path.basename(filepath)}")

    def on_index_finished(self, added, skipped):
        self.progress_bar.setVisible(False)
        
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
        category = self.selected_category
        
        if self.ref_embedding is not None:
            db_embeddings = self.db.get_all_embeddings()
            feedbacks = self.db.get_feedback_for_reference(self.ref_file_path)
            tolerance = self.slider_tolerance.value() / 100.0
            matches = search_similar_faces(self.ref_embedding, db_embeddings, feedbacks=feedbacks, tolerance=tolerance)
            
            if category.lower() != 'all':
                matches = [m for m in matches if m["category_label"] == category]
                
            self.populate_grid(matches)
            self.status_label.setText(f"Search results: {len(matches)} matching face(s) found.")
        else:
            items = self.db.get_media_files(category=category)
            self.populate_grid(items)
            self.status_label.setText(f"Showing {len(items)} file(s).")

    def on_feedback_received(self, file_path, embedding_id, is_match):
        if not self.ref_file_path:
            return
            
        # Record feedback in DB
        self.db.add_face_feedback(self.ref_file_path, embedding_id, is_match)
        
        # Display reinforcement status message
        action = "confirmed as match 👍" if is_match == 1 else "rejected as match 👎"
        self.status_label.setText(f"Feedback recorded: Face {action}. Reinforcing search model...")
        
        # Instantly update search results
        self.trigger_search()

    def load_all_media_from_db(self):
        self.trigger_search()

    def clear_grid(self):
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def populate_grid(self, items):
        self.clear_grid()
        if not items:
            return

        width = self.scroll_area.viewport().width()
        if width <= 0:
            width = 820
        
        tile_width = 175
        columns = max(1, width // tile_width)

        for idx, item in enumerate(items):
            tile = ImageTile(
                file_path=item["absolute_path"],
                name=item["file_name"],
                category=item["category_label"],
                thumbnail_blob=item["thumbnail_blob"],
                distance=item.get("distance", None),
                embedding_id=item.get("embedding_id", None),
                parent=self
            )
            tile.doubleClicked.connect(self.open_file_in_explorer)
            tile.feedbackClicked.connect(self.on_feedback_received)
            
            row = idx // columns
            col = idx % columns
            self.grid_layout.addWidget(tile, row, col)

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
