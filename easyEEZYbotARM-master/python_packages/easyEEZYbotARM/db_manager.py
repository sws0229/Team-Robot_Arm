import sqlite3
from datetime import datetime

class DBManager:
    def __init__(self, db_name='robot_arm.db'):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS sorting_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            action TEXT,
            is_processed INTEGER DEFAULT 0
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_log(self, ball_status, action_type):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = "INSERT INTO sorting_logs (timestamp, status, action, is_processed) VALUES (?, ?, ?, 0)"
        self.conn.execute(query, (now, ball_status, action_type))
        self.conn.commit()

    def get_pending_action(self):
        cursor = self.conn.cursor()
        query = "SELECT id, action FROM sorting_logs WHERE is_processed = 0 ORDER BY id ASC LIMIT 1"
        cursor.execute(query)
        return cursor.fetchone()

    def mark_as_processed(self, log_id):
        query = "UPDATE sorting_logs SET is_processed = 1 WHERE id = ?"
        self.conn.execute(query, (log_id,))
        self.conn.commit()

    def get_statistics(self):
        cursor = self.conn.cursor()
        query = "SELECT status, COUNT(*) FROM sorting_logs GROUP BY status"
        cursor.execute(query)
        return dict(cursor.fetchall())

    def close(self):
        self.conn.close()