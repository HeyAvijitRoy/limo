import sqlite3
import os
import numpy as np

class DatabaseManager:
    def __init__(self, db_path="face_index.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS media_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                absolute_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                date_modified REAL NOT NULL,
                category_label TEXT DEFAULT 'Uncategorized',
                thumbnail_blob BLOB
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER,
                embedding_vector BLOB NOT NULL, -- 128-float array serialized via numpy.tobytes()
                FOREIGN KEY(file_id) REFERENCES media_files(file_id) ON DELETE CASCADE
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                reference_path TEXT NOT NULL,
                target_embedding_id INTEGER NOT NULL,
                is_match INTEGER NOT NULL, -- 1 for 👍, 0 for 👎
                FOREIGN KEY(target_embedding_id) REFERENCES face_embeddings(embedding_id) ON DELETE CASCADE,
                UNIQUE(reference_path, target_embedding_id)
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON media_files(absolute_path);")
            conn.commit()

    def add_media_file(self, absolute_path, file_name, date_modified, category_label, thumbnail_blob):
        """
        Inserts or replaces a media file record.
        Returns the file_id.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO media_files (absolute_path, file_name, date_modified, category_label, thumbnail_blob)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(absolute_path) DO UPDATE SET
                file_name=excluded.file_name,
                date_modified=excluded.date_modified,
                category_label=excluded.category_label,
                thumbnail_blob=excluded.thumbnail_blob;
            """, (absolute_path, file_name, date_modified, category_label, thumbnail_blob))
            
            cursor.execute("SELECT file_id FROM media_files WHERE absolute_path = ?", (absolute_path,))
            row = cursor.fetchone()
            return row["file_id"]

    def add_face_embedding(self, file_id, embedding_vector):
        """
        Inserts a face embedding vector.
        embedding_vector should be a 128-element numpy float64/float32 array or similar.
        """
        if isinstance(embedding_vector, np.ndarray):
            blob_data = embedding_vector.astype(np.float64).tobytes()
        else:
            blob_data = np.array(embedding_vector, dtype=np.float64).tobytes()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO face_embeddings (file_id, embedding_vector)
            VALUES (?, ?);
            """, (file_id, blob_data))
            conn.commit()

    def remove_embeddings_for_file(self, file_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_embeddings WHERE file_id = ?;", (file_id,))
            conn.commit()

    def get_media_files(self, category=None):
        """
        Retrieves media files.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if category and category.lower() != 'all':
                cursor.execute("""
                SELECT file_id, absolute_path, file_name, date_modified, category_label, thumbnail_blob
                FROM media_files
                WHERE category_label = ?;
                """, (category,))
            else:
                cursor.execute("""
                SELECT file_id, absolute_path, file_name, date_modified, category_label, thumbnail_blob
                FROM media_files;
                """)
            return [dict(row) for row in cursor.fetchall()]

    def get_all_embeddings(self):
        """
        Returns a list of dictionaries with embedding vectors as numpy arrays.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT fe.embedding_id, fe.file_id, fe.embedding_vector, mf.absolute_path, mf.file_name, mf.category_label, mf.thumbnail_blob
            FROM face_embeddings fe
            JOIN media_files mf ON fe.file_id = mf.file_id;
            """)
            results = []
            for row in cursor.fetchall():
                vector_bytes = row["embedding_vector"]
                vector = np.frombuffer(vector_bytes, dtype=np.float64)
                results.append({
                    "embedding_id": row["embedding_id"],
                    "file_id": row["file_id"],
                    "embedding": vector,
                    "absolute_path": row["absolute_path"],
                    "file_name": row["file_name"],
                    "category_label": row["category_label"],
                    "thumbnail_blob": row["thumbnail_blob"]
                })
            return results

    def add_face_feedback(self, reference_path, target_embedding_id, is_match):
        """
        Inserts or replaces user feedback for a target face against a reference face.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO face_feedback (reference_path, target_embedding_id, is_match)
            VALUES (?, ?, ?);
            """, (reference_path, target_embedding_id, int(is_match)))
            conn.commit()

    def get_feedback_for_reference(self, reference_path):
        """
        Returns a dictionary mapping target_embedding_id -> is_match (1 or 0) for a reference.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT target_embedding_id, is_match
            FROM face_feedback
            WHERE reference_path = ?;
            """, (reference_path,))
            return {row["target_embedding_id"]: row["is_match"] for row in cursor.fetchall()}

    def delete_media_file(self, absolute_path):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM media_files WHERE absolute_path = ?;", (absolute_path,))
            conn.commit()

    def get_media_file_by_path(self, absolute_path):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT file_id, absolute_path, file_name, date_modified, category_label
            FROM media_files
            WHERE absolute_path = ?;
            """, (absolute_path,))
            row = cursor.fetchone()
            return dict(row) if row else None
            
    def clear_cache(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_feedback;")
            cursor.execute("DELETE FROM face_embeddings;")
            cursor.execute("DELETE FROM media_files;")
            conn.commit()
