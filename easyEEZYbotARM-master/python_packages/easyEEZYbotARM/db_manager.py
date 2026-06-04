"""
robot_arm.db 관리 모듈.

기존 sorting_logs 테이블을 확장해 인식·동작·결과 정보를 모두 저장합니다.
기존 DB 파일이 있어도 ALTER TABLE 로 컬럼만 추가하므로 데이터는 보존됩니다.

새 API (권장):
    log_id = db.insert_detection(status=..., defect_reason=..., circle_ratio=...,
                                  radius_px=..., edge_px=..., grid_row=...,
                                  grid_col=..., pixel_cx=..., pixel_cy=...,
                                  action=...)
    # 동작 끝나면
    db.update_action_result(log_id, target_j1=..., command_text=...,
                            action_started=..., action_finished=...,
                            simulation=..., success=..., error_msg=...)

기존 API (하위 호환):
    db.insert_log(status, action)
"""

import sqlite3
from datetime import datetime


# sorting_logs 확장 컬럼 목록 (기존 DB 마이그레이션용)
_EXTRA_COLUMNS = [
    # 인식 상세
    ("defect_reason",   "TEXT"),
    ("circle_ratio",    "REAL"),
    ("radius_px",       "INTEGER"),
    ("edge_px",         "INTEGER"),
    ("grid_row",        "INTEGER"),
    ("grid_col",        "INTEGER"),
    ("pixel_cx",        "INTEGER"),
    ("pixel_cy",        "INTEGER"),
    # 동작 결과
    ("target_j1",       "INTEGER"),
    ("command_text",    "TEXT"),
    ("action_started",  "TEXT"),
    ("action_finished", "TEXT"),
    ("simulation",      "INTEGER"),
    ("success",         "INTEGER"),
    ("error_msg",       "TEXT"),
]


def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class DBManager:
    def __init__(self, db_name='robot_arm.db'):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # dict처럼 컬럼명 접근 가능
        self._create_table()
        self._migrate()

    # ------------------------------------------------------------------
    # 스키마
    # ------------------------------------------------------------------
    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS sorting_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            action TEXT,
            is_processed INTEGER DEFAULT 0
        )
        """)
        self.conn.commit()

    def _migrate(self):
        """기존 DB 에 새 컬럼이 없으면 추가 (데이터 보존)."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(sorting_logs)")
        existing = {row[1] for row in cur.fetchall()}
        for col, typ in _EXTRA_COLUMNS:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE sorting_logs ADD COLUMN {col} {typ}")
        self.conn.commit()

    # ------------------------------------------------------------------
    # 기존 API (하위 호환)
    # ------------------------------------------------------------------
    def insert_log(self, ball_status, action_type):
        """예전 호출 코드를 위해 유지. 새 코드는 insert_detection 사용 권장."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sorting_logs (timestamp, status, action, is_processed) "
            "VALUES (?, ?, ?, 0)",
            (_now(), ball_status, action_type),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_pending_action(self):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, action FROM sorting_logs "
            "WHERE is_processed = 0 ORDER BY id ASC LIMIT 1"
        )
        return cur.fetchone()

    def mark_as_processed(self, log_id):
        self.conn.execute(
            "UPDATE sorting_logs SET is_processed = 1 WHERE id = ?", (log_id,)
        )
        self.conn.commit()

    def get_statistics(self):
        return self.stats_status_counts()

    # ------------------------------------------------------------------
    # 새 API
    # ------------------------------------------------------------------
    def insert_detection(self, *, status, action=None,
                         defect_reason=None, circle_ratio=None,
                         radius_px=None, edge_px=None,
                         grid_row=None, grid_col=None,
                         pixel_cx=None, pixel_cy=None):
        """검출 시점 정보를 INSERT 하고 row id 반환."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO sorting_logs
                (timestamp, status, action, is_processed,
                 defect_reason, circle_ratio, radius_px, edge_px,
                 grid_row, grid_col, pixel_cx, pixel_cy)
            VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (_now(), status, action,
             defect_reason, circle_ratio, radius_px, edge_px,
             grid_row, grid_col, pixel_cx, pixel_cy),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_action_result(self, log_id, *,
                             target_j1=None, command_text=None,
                             action_started=None, action_finished=None,
                             simulation=None, success=None, error_msg=None):
        """동작 시퀀스 종료 후 결과를 UPDATE."""
        self.conn.execute(
            """
            UPDATE sorting_logs
            SET target_j1       = ?,
                command_text    = ?,
                action_started  = ?,
                action_finished = ?,
                simulation      = ?,
                success         = ?,
                error_msg       = ?,
                is_processed    = 1
            WHERE id = ?
            """,
            (target_j1, command_text, action_started, action_finished,
             simulation, success, error_msg, log_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # 통계 헬퍼
    # ------------------------------------------------------------------
    def fetch_all(self, limit=None):
        cur = self.conn.cursor()
        q = "SELECT * FROM sorting_logs ORDER BY id DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        cur.execute(q)
        return [dict(r) for r in cur.fetchall()]

    def stats_status_counts(self):
        cur = self.conn.cursor()
        cur.execute("SELECT status, COUNT(*) FROM sorting_logs GROUP BY status")
        return dict(cur.fetchall())

    def stats_defect_reasons(self):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT defect_reason, COUNT(*) FROM sorting_logs "
            "WHERE status='Damaged' AND defect_reason IS NOT NULL "
            "GROUP BY defect_reason"
        )
        return dict(cur.fetchall())

    def stats_grid(self):
        """3x3 그리드 셀별 검출 횟수."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        cur = self.conn.cursor()
        cur.execute(
            "SELECT grid_row, grid_col, COUNT(*) FROM sorting_logs "
            "WHERE grid_row IS NOT NULL AND grid_col IS NOT NULL "
            "GROUP BY grid_row, grid_col"
        )
        for r, c, n in cur.fetchall():
            if 0 <= r < 3 and 0 <= c < 3:
                grid[r][c] = n
        return grid

    def stats_action_success(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS ok,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS fail,
                SUM(CASE WHEN simulation = 1 THEN 1 ELSE 0 END) AS sim,
                COUNT(*) AS total
            FROM sorting_logs
            WHERE action IS NOT NULL
            """
        )
        row = cur.fetchone()
        return {
            'ok': row['ok'] or 0,
            'fail': row['fail'] or 0,
            'sim': row['sim'] or 0,
            'total': row['total'] or 0,
        }

    def close(self):
        self.conn.close()
