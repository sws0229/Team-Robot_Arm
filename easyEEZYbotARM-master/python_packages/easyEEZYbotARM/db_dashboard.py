"""
robot_arm.db 시각화 대시보드.

실행:
    python db_dashboard.py             # 기본 DB 경로 자동 탐색
    python db_dashboard.py path/to.db  # 특정 DB 지정

출력:
    1) 콘솔: 텍스트 요약 표
    2) matplotlib 창: 6개 패널 대시보드
       ① 정상 vs 불량 카운트
       ② 불량 사유 분포 (Shape / Dent)
       ③ 3x3 그리드 검출 위치 히트맵
       ④ 시간대(시) 별 검출 건수
       ⑤ 동작 결과 (성공/실패/시뮬레이션)
       ⑥ 최근 검출 로그 10건
"""

import os
import sys
import sqlite3
import platform
from datetime import datetime

import numpy as np
import matplotlib

# 한글 폰트 설정 (OS 별)
_sys = platform.system()
if _sys == 'Darwin':
    matplotlib.rcParams['font.family'] = 'AppleGothic'
elif _sys == 'Windows':
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
else:
    # Linux: 나눔고딕이 설치돼 있으면 사용, 아니면 기본 폰트
    matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# DB 로드
# ----------------------------------------------------------------------
def find_db_path():
    """auto4.py 와 동일한 위치 우선, 없으면 상위(easyEEZYbotARM-master) 검색."""
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, '..', 'robot_arm.db')),         # python_packages/
        os.path.abspath(os.path.join(here, '..', '..', 'robot_arm.db')),   # easyEEZYbotARM-master/
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def load_rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM sorting_logs ORDER BY id DESC")
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


# ----------------------------------------------------------------------
# 집계
# ----------------------------------------------------------------------
def status_counts(rows):
    out = {}
    for r in rows:
        k = r.get('status') or 'Unknown'
        out[k] = out.get(k, 0) + 1
    return out


def defect_reasons(rows):
    out = {}
    for r in rows:
        if r.get('status') == 'Damaged':
            k = r.get('defect_reason') or '(미분류)'
            out[k] = out.get(k, 0) + 1
    return out


def grid_counts(rows):
    grid = np.zeros((3, 3), dtype=int)
    for r in rows:
        gr, gc = r.get('grid_row'), r.get('grid_col')
        if gr is None or gc is None:
            continue
        if 0 <= gr < 3 and 0 <= gc < 3:
            grid[gr][gc] += 1
    return grid


def hourly_counts(rows):
    out = {h: 0 for h in range(24)}
    for r in rows:
        ts = r.get('timestamp')
        if not ts:
            continue
        try:
            dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            out[dt.hour] += 1
        except Exception:
            pass
    return out


def action_success(rows):
    ok = fail = sim = total = 0
    for r in rows:
        if r.get('action') is None:
            continue
        total += 1
        if r.get('simulation') == 1:
            sim += 1
        if r.get('success') == 1:
            ok += 1
        elif r.get('success') == 0:
            fail += 1
    return ok, fail, sim, total


# ----------------------------------------------------------------------
# 텍스트 요약
# ----------------------------------------------------------------------
def print_text_summary(rows):
    line = "=" * 64
    print(line)
    print("           Team-Robot_Arm  검사 데이터 요약")
    print(line)
    print(f"총 검출 건수: {len(rows)}")

    sc = status_counts(rows)
    for k, v in sc.items():
        print(f"  - {k}: {v}")

    dr = defect_reasons(rows)
    if dr:
        print("\n[불량 사유]")
        for k, v in dr.items():
            print(f"  - {k}: {v}")

    grid = grid_counts(rows)
    if grid.sum() > 0:
        print("\n[3x3 그리드 검출 빈도]")
        for r in range(3):
            print("  " + " ".join(f"{int(grid[r][c]):4d}" for c in range(3)))

    ok, fail, sim, total = action_success(rows)
    if total > 0:
        print("\n[동작 결과]")
        print(f"  성공: {ok}   실패: {fail}   시뮬레이션: {sim}   전체: {total}")
        if (ok + fail) > 0:
            rate = ok / (ok + fail) * 100
            print(f"  실제 동작 성공률: {rate:.1f}%")

    print("\n[최근 10건]")
    header = (f"  {'ID':>4}  {'시각':<19}  {'상태':<8}  "
              f"{'사유':<8}  {'위치':<6}  {'성공'}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows[:10]:
        rid = str(r.get('id', '-'))
        ts = r.get('timestamp', '-') or '-'
        st = (r.get('status') or '-')
        rs = (r.get('defect_reason') or '-')
        rc = f"({r.get('grid_row','-')},{r.get('grid_col','-')})"
        sc_ = 'O' if r.get('success') == 1 else ('X' if r.get('success') == 0 else '-')
        print(f"  {rid:>4}  {ts:<19}  {st:<8}  {rs:<8}  {rc:<6}  {sc_}")
    print(line)


# ----------------------------------------------------------------------
# 시각 대시보드
# ----------------------------------------------------------------------
def draw_dashboard(rows):
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle('Team-Robot_Arm  검사 데이터 대시보드',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.30,
                          left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ① 정상 vs 불량
    ax1 = fig.add_subplot(gs[0, 0])
    sc = status_counts(rows)
    if sc:
        names = list(sc.keys())
        vals = list(sc.values())
        colors = ['#2ecc71' if n == 'Normal'
                  else '#e74c3c' if n == 'Damaged'
                  else '#95a5a6' for n in names]
        bars = ax1.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v, str(v),
                     ha='center', va='bottom', fontsize=11)
        ax1.set_ylim(0, max(vals) * 1.2 if vals else 1)
    else:
        ax1.text(0.5, 0.5, '데이터 없음', ha='center', va='center',
                 transform=ax1.transAxes)
    ax1.set_title('① 정상 vs 불량 카운트')
    ax1.set_ylabel('건수')

    # ② 불량 사유
    ax2 = fig.add_subplot(gs[0, 1])
    dr = defect_reasons(rows)
    if dr:
        colors2 = ['#e67e22', '#9b59b6', '#34495e', '#16a085']
        ax2.pie(list(dr.values()), labels=list(dr.keys()),
                autopct='%1.0f%%', colors=colors2[:len(dr)],
                startangle=90)
    else:
        ax2.text(0.5, 0.5, '불량 데이터 없음', ha='center', va='center',
                 transform=ax2.transAxes)
        ax2.axis('off')
    ax2.set_title('② 불량 사유 분포')

    # ③ 그리드 히트맵
    ax3 = fig.add_subplot(gs[1, 0])
    grid = grid_counts(rows)
    im = ax3.imshow(grid, cmap='YlOrRd', aspect='equal',
                    vmin=0, vmax=max(grid.max(), 1))
    for r in range(3):
        for c in range(3):
            ax3.text(c, r, str(int(grid[r][c])),
                     ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='black' if grid[r][c] < grid.max() * 0.6 else 'white')
    ax3.set_title('③ 3×3 그리드 검출 위치')
    ax3.set_xticks(range(3))
    ax3.set_yticks(range(3))
    ax3.set_xlabel('col')
    ax3.set_ylabel('row')
    plt.colorbar(im, ax=ax3, shrink=0.7)

    # ④ 시간대별
    ax4 = fig.add_subplot(gs[1, 1])
    hc = hourly_counts(rows)
    ax4.bar(range(24), [hc[h] for h in range(24)], color='#3498db')
    ax4.set_title('④ 시간대별 검출 건수')
    ax4.set_xlabel('시 (0~23)')
    ax4.set_ylabel('건수')
    ax4.set_xticks(range(0, 24, 2))

    # ⑤ 동작 성공률
    ax5 = fig.add_subplot(gs[2, 0])
    ok, fail, sim, total = action_success(rows)
    if total > 0:
        labels = ['성공', '실패', '시뮬레이션']
        vals = [ok, fail, sim]
        colors5 = ['#27ae60', '#c0392b', '#7f8c8d']
        bars = ax5.barh(labels, vals, color=colors5)
        for b, v in zip(bars, vals):
            ax5.text(v + max(vals) * 0.02 if max(vals) > 0 else 0.1,
                     b.get_y() + b.get_height() / 2,
                     str(v), va='center', fontsize=11)
        ax5.set_xlim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    else:
        ax5.text(0.5, 0.5, '동작 데이터 없음', ha='center', va='center',
                 transform=ax5.transAxes)
        ax5.axis('off')
    ax5.set_title(f'⑤ 동작 결과  (총 {total}건)')

    # ⑥ 최근 로그
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    ax6.set_title('⑥ 최근 검출 10건')
    # 모노스페이스 폰트는 한글이 안 들어가 헤더만 영문으로 고정
    txt_lines = [f"{'ID':>4}  {'Time':<14}  {'Status':<8}  {'Reason':<6}  {'Pos':<5} OK"]
    txt_lines.append('-' * 54)
    for r in rows[:10]:
        rid = str(r.get('id', '-'))
        ts = (r.get('timestamp') or '-')
        ts = ts[5:] if len(ts) >= 16 else ts  # 'MM-DD HH:MM:SS'
        st = (r.get('status') or '-')
        rs = (r.get('defect_reason') or '-')
        rc = f"({r.get('grid_row','-')},{r.get('grid_col','-')})"
        sv = r.get('success')
        sc_ = 'O' if sv == 1 else ('X' if sv == 0 else '-')
        txt_lines.append(f"{rid:>4}  {ts:<14}  {st:<8}  {rs:<6}  {rc:<5} {sc_}")
    ax6.text(0.0, 1.0, '\n'.join(txt_lines),
             family='monospace', fontsize=9,
             va='top', ha='left', transform=ax6.transAxes)

    plt.show()


# ----------------------------------------------------------------------
def main():
    db_path = find_db_path()
    print(f"DB 파일: {db_path}")
    if not os.path.exists(db_path):
        print("DB 파일을 찾을 수 없습니다. auto4.py 등을 먼저 실행해 데이터를 모으세요.")
        return

    rows = load_rows(db_path)
    if not rows:
        print("DB가 비어있습니다.")
        return

    print_text_summary(rows)
    draw_dashboard(rows)


if __name__ == "__main__":
    main()
