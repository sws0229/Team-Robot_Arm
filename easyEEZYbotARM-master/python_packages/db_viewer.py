import os
from db_manager import DBManager

# 상위 폴더의 DB 파일 절대 경로 지정
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'robot_arm.db'))
db = DBManager(db_path)

print("--- 통계 데이터 ---")
print(db.get_statistics())

print("\n--- 대기 중인 동작 (미처리) ---")
print(db.get_pending_action())

print("\n--- 최근 로그 (10건) ---")
cursor = db.conn.cursor()
cursor.execute("SELECT * FROM sorting_logs ORDER BY id DESC LIMIT 10")
for row in cursor.fetchall():
    print(row)

db.close()