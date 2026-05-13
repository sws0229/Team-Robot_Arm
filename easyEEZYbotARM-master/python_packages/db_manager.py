import sqlite3
from datetime import datetime

class DBManager:
    def __init__(self, db_name='robot_arm.db'):
        # db_name은 .gitignore에 등록된 확장자를 사용하십시오.
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS sorting_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            action TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_log(self, ball_status, action_type):
        """
        ball_status: 'Normal' or 'Dented'
        action_type: 'A' or 'B'
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = "INSERT INTO sorting_logs (timestamp, status, action) VALUES (?, ?, ?)"
        self.conn.execute(query, (now, ball_status, action_type))
        self.conn.commit()

    def close(self):
        self.conn.close()