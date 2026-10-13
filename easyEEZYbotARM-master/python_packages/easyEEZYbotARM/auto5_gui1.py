# ============================================================
# auto5_gui.py - auto5.py용 Tkinter GUI
#
# 사용법: auto5.py와 같은 폴더에 두고 실행
#   python auto5_gui.py
#   python auto5_gui.py --camera 1 --simulation
#   (auto5.py의 실행 옵션을 그대로 사용합니다)
#
# 추가 설치 없이 표준 라이브러리(tkinter) + 기존 의존성만 사용합니다.
# 로봇 동작은 별도 스레드에서 실행되어 영상이 끊기지 않습니다.
# ============================================================

import base64
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import cv2
import numpy as np

import auto5

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_DISPLAY_WIDTH = 1100

# ----- 디자인 팔레트 -----
BG = "#F4F5F7"
CARD = "#FFFFFF"
BORDER = "#E5E7EB"
TEXT = "#111827"
MUTED = "#6B7280"
ACCENT = "#2563EB"
GOOD = "#16A34A"
BAD = "#DC2626"
VIDEO_BG = "#0B0F19"

# 상태 바 / 칩 색상 (배경, 글자)
TONES = {
    "neutral": ("#F3F4F6", "#374151"),
    "info": ("#E0ECFF", "#1D4ED8"),
    "success": ("#DCFCE7", "#166534"),
    "warning": ("#FEF3C7", "#92400E"),
    "danger": ("#FEE2E2", "#B91C1C"),
}

ARM_TEXT = "검사 시작   Space"
CANCEL_TEXT = "검사 취소   Space"
PLACEHOLDER = "시스템이 꺼져 있습니다\n오른쪽의 [시스템 시작]을 눌러 주세요"


def nice_ceiling(value):
    """그래프 세로축 최댓값과 눈금 간격을 보기 좋게 계산한다."""
    value = max(int(value), 4)
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000):
        if -(-value // step) <= 5:
            return -(-value // step) * step, step
    return value, max(value // 5, 1)


def tone_for(color):
    """기존 BGR 색상 값을 상태 톤 이름으로 바꾼다."""
    blue, green, red = color
    if (blue, green, red) == (0, 0, 0):
        return "neutral"
    if red > 200 and green < 80:
        return "danger"
    if green >= 150 and red < 100:
        return "success"
    if red > 200 and 120 <= green <= 180 and blue < 50:
        return "warning"
    return "info"


class AbortRobot(Exception):
    """비상 정지 요청 시 로봇 동작 스레드를 중단시키는 예외."""


def bgr_to_hex(color):
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"


class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        root.title("Smart Quality Control")
        root.minsize(1120, 720)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.classifier = None
        self.running = False
        self.feedback_mode = False
        self.camera_index = 0
        self.grid_w = 1
        self.grid_h = 1
        self.events = queue.Queue()
        self.abort = threading.Event()
        self.worker = None
        self.busy = False
        self.photo = None
        self.stats = {"normal": 0, "damaged": 0, "error": 0}
        self.history = []  # 분류 완료 순서대로 "normal"/"damaged" 기록 (그래프용)
        self.exposure_applied = None
        self.saved_counts = {"normal": 0, "damaged": 0}
        self.reset_state()

        self.build_ui()

        # 로봇 동작 중 대기 함수를 GUI 친화적으로 교체한다.
        # (원래 함수는 cv2.imshow를 쓰기 때문에 스레드에서 쓸 수 없다.)
        auto5.keep_alive_sleep = self.interruptible_sleep

        root.bind("<space>", lambda e: self.toggle_arm() if self.key_ok() else None)
        root.bind("<Escape>", lambda e: self.emergency_stop())
        for key, label in (("n", "normal"), ("N", "normal"), ("d", "damaged"), ("D", "damaged")):
            root.bind(
                f"<KeyPress-{key}>",
                lambda e, lb=label: self.save_feedback(lb) if self.key_ok() else None,
            )

    # ------------------------------------------------------------
    # 상태 초기화
    # ------------------------------------------------------------
    def reset_state(self):
        self.fail_count = 0
        self.last_action_time = 0.0
        self.stable_ball = None
        self.stable_count = 0
        self.armed = False
        self.armed_at = 0.0
        self.waiting_for_clear = False
        self.clear_count = 0
        self.last_balls = []
        self.last_error = None
        self.last_raw = None
        self.last_mask = None
        self.last_edges = None
        self.last_feedback_save = 0.0
        self.feedback_note = None
        self.feedback_note_until = 0.0

    def clear_tracking(self):
        self.stable_ball = None
        self.stable_count = 0

    # ------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------
    def setup_style(self):
        root = self.root
        available = set(tkfont.families())
        self.font = next(
            (
                name
                for name in (
                    "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo",
                    "Noto Sans CJK KR", "Noto Sans KR", "Segoe UI",
                )
                if name in available
            ),
            "Arial",
        )
        F = self.font
        root.configure(bg=BG)
        root.option_add("*TCombobox*Listbox.font", (F, 10))

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=(F, 10))

        def button(name, bg, active, disabled_bg, fg="white", disabled_fg="white"):
            style.configure(
                name, background=bg, foreground=fg, font=(F, 10, "bold"),
                padding=(12, 9), borderwidth=0, relief="flat", focuscolor=bg,
            )
            style.map(
                name,
                background=[("disabled", disabled_bg), ("pressed", active), ("active", active)],
                foreground=[("disabled", disabled_fg)],
            )

        button("Primary.TButton", ACCENT, "#1D4ED8", "#B7CBF7")
        button("Danger.TButton", BAD, "#B91C1C", "#F2B8B8")
        button("Good.TButton", GOOD, "#15803D", "#A7DDB9")
        button("Dark.TButton", "#1F2937", "#111827", "#C4C8CF")
        button("Soft.TButton", "#EEF0F4", "#E2E5EB", "#F3F4F6", fg=TEXT, disabled_fg="#A0A6B1")

        style.configure(
            "Link.TButton", background=CARD, foreground=ACCENT, font=(F, 9),
            padding=0, borderwidth=0, relief="flat", focuscolor=CARD,
        )
        style.map("Link.TButton", foreground=[("active", "#1D4ED8")], background=[("active", CARD)])

        # 구간 선택(세그먼트) 버튼
        style.configure(
            "Seg.Toolbutton", font=(F, 10), padding=(10, 7), background="#EEF0F4",
            foreground=TEXT, borderwidth=0, relief="flat", anchor="center",
        )
        style.map(
            "Seg.Toolbutton",
            background=[
                ("disabled selected", "#9DB8F2"), ("disabled", "#F3F4F6"),
                ("selected", ACCENT), ("active", "#E2E5EB"),
            ],
            foreground=[
                ("disabled selected", "white"), ("disabled", "#A0A6B1"), ("selected", "white"),
            ],
        )

        style.configure("Card.TCheckbutton", background=CARD, foreground=TEXT, font=(F, 10))
        style.map("Card.TCheckbutton", background=[("active", CARD)])

        for name in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(
                name, padding=6, fieldbackground="white", bordercolor=BORDER,
                lightcolor=BORDER, darkcolor=BORDER, arrowsize=14,
            )
            style.map(
                name,
                fieldbackground=[("disabled", "#F3F4F6")],
                foreground=[("disabled", "#9CA3AF")],
            )

        style.configure(
            "Card.Horizontal.TScale", background=ACCENT, troughcolor="#E5E7EB",
            bordercolor=CARD, lightcolor=ACCENT, darkcolor=ACCENT,
        )

    def make_card(self, parent, title, expand=False):
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(
            outer, text=title, bg=CARD, fg=MUTED, font=(self.font, 9, "bold")
        ).pack(anchor="w", padx=14, pady=(10, 4))
        body = tk.Frame(outer, bg=CARD)
        body.pack(fill="both" if expand else "x", expand=expand, padx=14, pady=(0, 12))
        return outer, body

    def add_slider(self, parent, title, variable, text_var, low, high):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text=title, bg=CARD, fg=TEXT, font=(self.font, 10)).pack(side="left")
        tk.Label(
            row, textvariable=text_var, bg=CARD, fg=ACCENT, font=(self.font, 10, "bold")
        ).pack(side="right")
        ttk.Scale(
            parent, from_=low, to=high, variable=variable, command=self.on_hsv,
            style="Card.Horizontal.TScale", takefocus=0,
        ).pack(fill="x", pady=(2, 4))

    def build_ui(self):
        self.mode_var = tk.StringVar(value="feedback" if self.args.feedback else "auto")
        self.camera_var = tk.StringVar(
            value=str(self.args.camera if self.args.camera is not None else auto5.DEFAULT_CAMERA_INDEX)
        )
        self.port_var = tk.StringVar(value=self.args.serial_port)
        self.sim_var = tk.BooleanVar(value=self.args.simulation)
        self.flip_var = tk.StringVar(value=self.args.flip)
        self.view_var = tk.StringVar(value="video")
        self.chart_mode = tk.StringVar(value="bar")
        self.rate_var = tk.StringVar()
        self.auto_exp_var = tk.BooleanVar(value=bool(self.args.auto_exposure))
        self.exp_var = tk.DoubleVar(value=max(-13.0, min(0.0, float(self.args.exposure))))
        self.exp_text = tk.StringVar()
        self.exp_note = tk.StringVar(value="낮을수록 어두워집니다  ·  시작하면 적용됩니다")

        self.setup_style()
        root = self.root

        # ---------- 헤더 ----------
        header = tk.Frame(root, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        header.pack(fill="x")
        title_box = tk.Frame(header, bg=CARD)
        title_box.pack(side="left", padx=20, pady=12)
        tk.Label(
            title_box, text="Smart Quality Control", bg=CARD, fg=TEXT,
            font=(self.font, 15, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box, text="탁구공 품질 검사 · 자동 분류 시스템", bg=CARD, fg=MUTED,
            font=(self.font, 9),
        ).pack(anchor="w")
        self.chip = tk.Label(header, text="", font=(self.font, 10, "bold"), padx=14, pady=5)
        self.chip.pack(side="right", padx=20)

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=16)

        # ---------- 왼쪽: 영상 + 상태 ----------
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        video_card = tk.Frame(left, bg=VIDEO_BG, highlightbackground=BORDER, highlightthickness=1)
        video_card.pack(fill="both", expand=True)
        self.holder = tk.Frame(video_card, bg=VIDEO_BG, width=640, height=400)
        self.holder.pack_propagate(False)
        self.holder.pack(fill="both", expand=True)
        self.video = tk.Label(
            self.holder, bg=VIDEO_BG, fg="#9CA3AF", font=(self.font, 11),
            text=PLACEHOLDER, justify="center",
        )
        self.video.pack(expand=True)
        self.video.bind("<Button-1>", lambda e: self.root.focus_set())
        self.status_label = tk.Label(
            left, text="", anchor="w", font=(self.font, 11, "bold"), padx=16, pady=11
        )
        self.status_label.pack(fill="x", pady=(10, 0))

        # ---------- 왼쪽 아래: 누적 현황 (숫자 + 그래프) ----------
        stat_card, stat_body = self.make_card(left, "누적 현황")
        stat_card.pack(fill="x", pady=(10, 0))
        stat_body.columnconfigure(1, weight=1)

        info = tk.Frame(stat_body, bg=CARD)
        info.grid(row=0, column=0, sticky="n", padx=(0, 16))
        tiles = tk.Frame(info, bg=CARD)
        tiles.pack(anchor="w")
        self.stat_vars = {}
        for index, (key, caption, color) in enumerate(
            (("normal", "정상", GOOD), ("damaged", "불량", BAD), ("error", "오류", MUTED))
        ):
            tile = tk.Frame(tiles, bg="#F9FAFB", highlightbackground=BORDER, highlightthickness=1)
            tile.grid(row=0, column=index, padx=(0 if index == 0 else 6, 0))
            var = tk.StringVar(value="0")
            tk.Label(
                tile, textvariable=var, width=4, bg="#F9FAFB", fg=color,
                font=(self.font, 20, "bold"),
            ).pack(pady=(8, 0))
            tk.Label(
                tile, text=caption, bg="#F9FAFB", fg=MUTED, font=(self.font, 9)
            ).pack(pady=(0, 8))
            self.stat_vars[key] = var
        tk.Label(
            info, textvariable=self.rate_var, bg=CARD, fg=TEXT,
            font=(self.font, 10, "bold"), anchor="w",
        ).pack(fill="x", pady=(10, 0))
        self.fb_count_var = tk.StringVar()
        tk.Label(
            info, textvariable=self.fb_count_var, bg=CARD, fg=MUTED,
            font=(self.font, 9), anchor="w",
        ).pack(fill="x")
        seg3 = tk.Frame(info, bg=CARD)
        seg3.pack(fill="x", pady=(10, 0))
        seg3.columnconfigure((0, 1), weight=1, uniform="chart")
        for index, (label, value) in enumerate((("막대", "bar"), ("추이", "line"))):
            ttk.Radiobutton(
                seg3, text=label, value=value, variable=self.chart_mode,
                style="Seg.Toolbutton", takefocus=0, command=self.draw_chart,
            ).grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0))

        self.chart = tk.Canvas(stat_body, height=160, bg=CARD, highlightthickness=0)
        self.chart.grid(row=0, column=1, sticky="nsew")
        self.chart.bind("<Configure>", self.draw_chart)

        # ---------- 오른쪽: 패널 ----------
        right = tk.Frame(body, bg=BG, width=340)
        right.pack_propagate(False)
        right.pack(side="right", fill="y", padx=(16, 0))

        # 연결 설정
        card, box = self.make_card(right, "연결 설정")
        card.pack(fill="x")
        box.columnconfigure(1, weight=1)
        seg = tk.Frame(box, bg=CARD)
        seg.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        seg.columnconfigure((0, 1), weight=1, uniform="mode")
        r_auto = ttk.Radiobutton(
            seg, text="자동 분류", value="auto", variable=self.mode_var,
            style="Seg.Toolbutton", takefocus=0,
        )
        r_fb = ttk.Radiobutton(
            seg, text="피드백 수집", value="feedback", variable=self.mode_var,
            style="Seg.Toolbutton", takefocus=0,
        )
        r_auto.grid(row=0, column=0, sticky="ew")
        r_fb.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        def form_label(text, row):
            tk.Label(
                box, text=text, bg=CARD, fg=MUTED, font=(self.font, 9)
            ).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 12))

        form_label("카메라 번호", 1)
        cam = ttk.Spinbox(box, from_=0, to=9, width=8, textvariable=self.camera_var)
        cam.grid(row=1, column=1, sticky="ew", pady=3)
        form_label("시리얼 포트", 2)
        port = ttk.Entry(box, textvariable=self.port_var, width=10)
        port.grid(row=2, column=1, sticky="ew", pady=3)
        form_label("화면 뒤집기", 3)
        flip = ttk.Combobox(
            box, width=10, state="readonly", textvariable=self.flip_var,
            values=("both", "horizontal", "vertical", "none"),
        )
        flip.grid(row=3, column=1, sticky="ew", pady=3)
        flip.bind("<<ComboboxSelected>>", lambda e: self.root.focus_set())
        sim = ttk.Checkbutton(
            box, text="시뮬레이션 (로봇 없이 실행)", variable=self.sim_var,
            style="Card.TCheckbutton", takefocus=0,
        )
        sim.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 4))
        self.start_btn = ttk.Button(
            box, text="시스템 시작", style="Primary.TButton", takefocus=0,
            command=self.toggle_system,
        )
        self.start_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.setting_widgets = [r_auto, r_fb, cam, port, sim]

        # 조작 (모드에 따라 해당 버튼만 표시)
        card, box = self.make_card(right, "조작")
        card.pack(fill="x", pady=(12, 0))

        self.auto_ctl = tk.Frame(box, bg=CARD)
        self.arm_btn = ttk.Button(
            self.auto_ctl, text=ARM_TEXT, style="Primary.TButton", takefocus=0,
            command=self.toggle_arm,
        )
        self.arm_btn.pack(fill="x")
        row = tk.Frame(self.auto_ctl, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        row.columnconfigure((0, 1), weight=1, uniform="ctl")
        self.estop_btn = ttk.Button(
            row, text="비상 정지   Esc", style="Danger.TButton", takefocus=0,
            command=self.emergency_stop,
        )
        self.estop_btn.grid(row=0, column=0, sticky="ew")
        self.home_btn = ttk.Button(
            row, text="홈 위치로", style="Soft.TButton", takefocus=0, command=self.launch_home
        )
        self.home_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.fb_ctl = tk.Frame(box, bg=CARD)
        self.fb_ctl.columnconfigure((0, 1), weight=1, uniform="fb")
        self.ok_btn = ttk.Button(
            self.fb_ctl, text="정상 저장   N", style="Good.TButton", takefocus=0,
            command=lambda: self.save_feedback("normal"),
        )
        self.ok_btn.grid(row=0, column=0, sticky="ew")
        self.bad_btn = ttk.Button(
            self.fb_ctl, text="불량 저장   D", style="Danger.TButton", takefocus=0,
            command=lambda: self.save_feedback("damaged"),
        )
        self.bad_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # 카메라 노출
        card, box = self.make_card(right, "카메라 노출")
        card.pack(fill="x", pady=(12, 0))
        exp_row = tk.Frame(box, bg=CARD)
        exp_row.pack(fill="x")
        ttk.Checkbutton(
            exp_row, text="자동 노출", variable=self.auto_exp_var,
            style="Card.TCheckbutton", takefocus=0, command=self.on_auto_exposure,
        ).pack(side="left")
        tk.Label(
            exp_row, textvariable=self.exp_text, bg=CARD, fg=ACCENT,
            font=(self.font, 10, "bold"),
        ).pack(side="right")
        self.exp_scale = ttk.Scale(
            box, from_=-13, to=0, variable=self.exp_var, command=self.on_exposure,
            style="Card.Horizontal.TScale", takefocus=0,
        )
        self.exp_scale.pack(fill="x", pady=(4, 2))
        tk.Label(
            box, textvariable=self.exp_note, bg=CARD, fg=MUTED,
            font=(self.font, 8), anchor="w",
        ).pack(fill="x")

        # 영상 설정
        card, box = self.make_card(right, "영상 설정")
        card.pack(fill="x", pady=(12, 0))
        seg2 = tk.Frame(box, bg=CARD)
        seg2.pack(fill="x", pady=(0, 8))
        seg2.columnconfigure((0, 1, 2), weight=1, uniform="view")
        for index, (label, value) in enumerate(
            (("영상", "video"), ("흰색 마스크", "mask"), ("엣지", "edges"))
        ):
            ttk.Radiobutton(
                seg2, text=label, value=value, variable=self.view_var,
                style="Seg.Toolbutton", takefocus=0,
            ).grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0))

        self.hsv_default = (float(auto5.LOWER_WHITE[2]), float(auto5.UPPER_WHITE[1]))
        self.v_var = tk.DoubleVar(value=self.hsv_default[0])
        self.s_var = tk.DoubleVar(value=self.hsv_default[1])
        self.v_text = tk.StringVar()
        self.s_text = tk.StringVar()
        self.add_slider(box, "밝기 하한 (V)", self.v_var, self.v_text, 100, 255)
        self.add_slider(box, "채도 상한 (S)", self.s_var, self.s_text, 0, 255)
        foot = tk.Frame(box, bg=CARD)
        foot.pack(fill="x")
        tk.Label(
            foot, text="밝기↑ · 채도↓ 일수록 엄격하게 인식", bg=CARD, fg=MUTED,
            font=(self.font, 8),
        ).pack(side="left")
        ttk.Button(
            foot, text="초기화", style="Link.TButton", takefocus=0, command=self.reset_hsv
        ).pack(side="right")

        # 로그
        card, box = self.make_card(right, "로그", expand=True)
        card.pack(fill="both", expand=True, pady=(12, 0))
        self.log_box = tk.Text(
            box, height=5, width=10, state="disabled", wrap="word", bg="#F9FAFB", fg=TEXT,
            relief="flat", font=(self.font, 9), padx=8, pady=6, spacing1=2,
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.log_box.tag_configure("time", foreground=MUTED)
        self.log_box.pack(fill="both", expand=True)

        self.mode_var.trace_add("write", self.show_mode_controls)
        self.show_mode_controls()
        self.on_hsv()
        self.on_auto_exposure()
        self.set_status("대기 중  ·  오른쪽의 [시스템 시작]을 눌러 주세요")
        self.refresh_stats()
        self.update_buttons()

    def show_mode_controls(self, *_):
        if self.mode_var.get() == "feedback":
            self.auto_ctl.pack_forget()
            self.fb_ctl.pack(fill="x")
        else:
            self.fb_ctl.pack_forget()
            self.auto_ctl.pack(fill="x")

    # ------------------------------------------------------------
    # 공통 유틸
    # ------------------------------------------------------------
    def key_ok(self):
        focused = self.root.focus_get()
        return not isinstance(focused, (tk.Entry, ttk.Entry, tk.Spinbox))

    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{stamp}  ", "time")
        self.log_box.insert("end", f"{text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, text, color=(0, 0, 0)):
        bg, fg = TONES[tone_for(color)]
        self.status_label.configure(text=text, bg=bg, fg=fg)

    def refresh_stats(self):
        for key, var in self.stat_vars.items():
            var.set(str(self.stats[key]))
        if self.feedback_mode:
            self.fb_count_var.set(
                f"피드백 저장   정상 {self.saved_counts['normal']}  ·  "
                f"불량 {self.saved_counts['damaged']}"
            )
        else:
            self.fb_count_var.set("")
        total = self.stats["normal"] + self.stats["damaged"]
        if total:
            rate = self.stats["damaged"] / total * 100
            self.rate_var.set(f"불량률 {rate:.1f}%   (총 {total}개)")
        else:
            self.rate_var.set("불량률 -   (총 0개)")
        self.draw_chart()

    def update_buttons(self):
        auto_on = self.running and not self.feedback_mode
        fb_on = self.running and self.feedback_mode
        self.arm_btn.configure(
            state="normal" if auto_on else "disabled",
            text=CANCEL_TEXT if self.armed else ARM_TEXT,
            style="Soft.TButton" if self.armed else "Primary.TButton",
        )
        self.estop_btn.configure(state="normal" if auto_on else "disabled")
        self.home_btn.configure(state="normal" if auto_on and not self.busy else "disabled")
        self.ok_btn.configure(state="normal" if fb_on else "disabled")
        self.bad_btn.configure(state="normal" if fb_on else "disabled")
        for widget in self.setting_widgets:
            widget.configure(state="disabled" if self.running else "normal")
        self.start_btn.configure(
            text="시스템 종료" if self.running else "시스템 시작",
            style="Dark.TButton" if self.running else "Primary.TButton",
        )

        if not self.running:
            chip, tone = "●  꺼짐", "neutral"
        elif self.busy:
            chip, tone = "●  로봇 동작 중", "warning"
        elif self.feedback_mode:
            chip, tone = "●  피드백 수집 중", "info"
        else:
            chip, tone = "●  자동 분류 실행 중", "success"
        bg, fg = TONES[tone]
        self.chip.configure(text=chip, bg=bg, fg=fg)

    def on_hsv(self, _=None):
        # make_white_mask()가 호출될 때마다 전역값을 읽으므로 즉시 반영된다.
        v = int(round(self.v_var.get()))
        s = int(round(self.s_var.get()))
        self.v_text.set(str(v))
        self.s_text.set(str(s))
        auto5.LOWER_WHITE = np.array([0, 0, v], dtype=np.uint8)
        auto5.UPPER_WHITE = np.array([179, s, 255], dtype=np.uint8)

    def reset_hsv(self):
        self.v_var.set(self.hsv_default[0])
        self.s_var.set(self.hsv_default[1])
        self.on_hsv()

    # ------------------------------------------------------------
    # 누적 그래프
    # ------------------------------------------------------------
    def draw_chart(self, *_):
        cv = self.chart
        cv.delete("all")
        width, height = cv.winfo_width(), cv.winfo_height()
        if width < 140 or height < 90:
            return
        font = self.font
        normal, damaged = self.stats["normal"], self.stats["damaged"]

        if normal + damaged == 0:
            cv.create_text(
                width / 2, height / 2, justify="center", fill=MUTED, font=(font, 10),
                text="아직 분류된 공이 없습니다\n분류가 끝날 때마다 그래프가 갱신됩니다",
            )
            return

        left, right, top, bottom = 34, 34, 12, 26
        plot_w, plot_h = width - left - right, height - top - bottom
        y_max, step = nice_ceiling(max(normal, damaged))

        def y_of(value):
            return top + plot_h * (1 - value / y_max)

        for tick in range(0, y_max + 1, step):
            y = y_of(tick)
            cv.create_line(left, y, width - right, y, fill="#EEF0F3")
            cv.create_text(left - 8, y, text=str(tick), anchor="e", fill=MUTED, font=(font, 8))
        cv.create_line(left, y_of(0), width - right, y_of(0), fill="#D1D5DB")

        if self.chart_mode.get() == "bar":
            bar_w = min(plot_w * 0.2, 80)
            for index, (label, value, color) in enumerate(
                (("정상", normal, GOOD), ("불량", damaged, BAD))
            ):
                cx = left + plot_w * (0.3 + 0.4 * index)
                y = y_of(value)
                if value > 0:
                    cv.create_rectangle(
                        cx - bar_w / 2, y, cx + bar_w / 2, y_of(0), fill=color, outline=color
                    )
                cv.create_text(cx, y - 9, text=str(value), fill=color, font=(font, 11, "bold"))
                cv.create_text(cx, height - bottom + 13, text=label, fill=TEXT, font=(font, 9))
            return

        # 추이(누적 꺾은선)
        counts = {"normal": 0, "damaged": 0}
        points = {"normal": [(0, 0)], "damaged": [(0, 0)]}
        for index, key in enumerate(self.history, 1):
            counts[key] += 1
            points["normal"].append((index, counts["normal"]))
            points["damaged"].append((index, counts["damaged"]))
        total_events = max(len(self.history), 1)

        def x_of(index):
            return left + plot_w * index / total_events

        label_shift = {"normal": -7, "damaged": 7} if normal == damaged else {"normal": 0, "damaged": 0}
        for key, color in (("damaged", BAD), ("normal", GOOD)):
            coords = []
            for index, value in points[key]:
                coords += [x_of(index), y_of(value)]
            cv.create_line(*coords, fill=color, width=2, joinstyle="round")
            end_x, end_y = coords[-2], coords[-1]
            cv.create_oval(end_x - 4, end_y - 4, end_x + 4, end_y + 4, fill=color, outline=CARD, width=2)
            cv.create_text(
                end_x + 9, end_y + label_shift[key], text=str(counts[key]), anchor="w",
                fill=color, font=(font, 10, "bold"),
            )

        for offset, (label, color) in enumerate((("정상", GOOD), ("불량", BAD))):
            lx = left + 10 + offset * 52
            cv.create_rectangle(lx, top + 5, lx + 10, top + 15, fill=color, outline=color)
            cv.create_text(lx + 15, top + 10, text=label, anchor="w", fill=TEXT, font=(font, 9))
        cv.create_text(left, height - bottom + 13, text="0", anchor="w", fill=MUTED, font=(font, 8))
        cv.create_text(
            width - right, height - bottom + 13, text=f"{len(self.history)}회",
            anchor="e", fill=MUTED, font=(font, 8),
        )
        cv.create_text(
            left + plot_w / 2, height - bottom + 13, text="분류 횟수 →",
            fill=MUTED, font=(font, 8),
        )

    # ------------------------------------------------------------
    # 카메라 노출
    # ------------------------------------------------------------
    def on_auto_exposure(self):
        auto = self.auto_exp_var.get()
        self.exp_scale.configure(state="disabled" if auto else "normal")
        self.apply_exposure(force=True)

    def on_exposure(self, _=None):
        self.apply_exposure()

    def apply_exposure(self, force=False):
        auto = self.auto_exp_var.get()
        value = int(round(self.exp_var.get()))
        self.exp_text.set("자동" if auto else str(value))
        if self.cap is None:
            return

        key = "auto" if auto else value
        if not force and key == self.exposure_applied:
            return
        self.exposure_applied = key

        # 지원하지 않는 카메라는 아래 설정을 자동으로 무시한다.
        if auto:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.exp_note.set("카메라 자동 노출을 사용 중입니다")
            return
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        actual = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        if abs(actual - value) > 0.5:
            self.exp_note.set(f"카메라 응답 {actual:.1f}  ·  이 값을 지원하지 않을 수 있습니다")
        else:
            self.exp_note.set(f"카메라 응답 {actual:.1f}  ·  낮을수록 어둡게")

    # ------------------------------------------------------------
    # 시작 / 종료
    # ------------------------------------------------------------
    def toggle_system(self):
        if self.running:
            self.stop_system()
        else:
            self.start_system()

    def start_system(self):
        self.feedback_mode = self.mode_var.get() == "feedback"
        try:
            self.camera_index = int(self.camera_var.get())
        except ValueError:
            messagebox.showerror("입력 오류", "카메라 번호는 숫자여야 합니다.")
            return

        try:
            self.classifier = auto5.BallQualityClassifier(self.args.model)
        except (FileNotFoundError, RuntimeError, cv2.error) as error:
            messagebox.showerror("ML 모델 오류", f"{error}\n\n안전을 위해 시작하지 않습니다.")
            return

        auto5.db = None
        auto5.ser = None
        try:
            if self.feedback_mode:
                auto5.prepare_feedback_dirs(self.args.feedback_dir)
                self.saved_counts = auto5.feedback_counts(self.args.feedback_dir)
            else:
                auto5.db = auto5.DBManager(os.path.join(auto5.PARENT_DIR, "robot_arm.db"))
                simulation = self.sim_var.get()
                auto5.ser = auto5.open_serial(self.port_var.get(), simulation)
                if auto5.ser is None and not simulation:
                    raise RuntimeError(
                        f"아두이노({self.port_var.get()}) 연결에 실패했습니다.\n"
                        "로봇 없이 시험하려면 '시뮬레이션'을 체크하세요."
                    )
        except Exception as error:
            self.release_devices()
            messagebox.showerror("장치 연결 오류", str(error))
            return

        self.cap = auto5.open_camera(
            self.camera_index, self.auto_exp_var.get(), float(round(self.exp_var.get()))
        )
        if self.cap is None:
            self.release_devices()
            messagebox.showerror("카메라 오류", f"{self.camera_index}번 카메라를 열 수 없습니다.")
            return

        ok, probe = self.cap.read()
        if not ok:
            self.release_devices()
            messagebox.showerror("카메라 오류", "카메라에서 영상을 읽을 수 없습니다.")
            return

        probe = auto5.flip_frame(probe, self.flip_var.get())
        self.apply_exposure(force=True)  # 실제 카메라 응답값을 읽어 표시
        height, width = probe.shape[:2]
        self.grid_w = max(width // 3, 1)
        self.grid_h = max(height // 3, 1)

        self.reset_state()
        self.running = True
        self.root.focus_set()  # 입력칸에 포커스가 남아 Space/N/D가 막히는 것 방지
        self.update_buttons()
        self.refresh_stats()
        mode_text = "피드백 수집" if self.feedback_mode else "자동 분류"
        self.log(f"시작: {mode_text} | 카메라 {self.camera_index} | {width}x{height}")
        if not self.feedback_mode:
            self.launch_home()
        self.tick()

    def stop_system(self, reason=None):
        if self.busy:
            self.abort.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=3)
        self.running = False
        self.busy = False
        self.release_devices()
        self.armed = False
        self.update_buttons()
        self.video.configure(image="", text=PLACEHOLDER)
        self.exposure_applied = None
        self.exp_note.set("낮을수록 어두워집니다  ·  시작하면 적용됩니다")
        self.photo = None
        self.set_status(reason or "대기 중")
        self.log(reason or "시스템을 종료했습니다.")

    def release_devices(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if auto5.db is not None:
            try:
                auto5.db.close()
            except Exception:
                pass
            auto5.db = None
        if auto5.ser is not None:
            try:
                auto5.ser.close()
            except Exception:
                pass
            auto5.ser = None

    def on_close(self):
        if self.running:
            self.stop_system()
        self.root.destroy()

    # ------------------------------------------------------------
    # 로봇 동작 (별도 스레드)
    # ------------------------------------------------------------
    def interruptible_sleep(self, seconds, cap=None, flip_mode="both"):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.abort.is_set():
                raise AbortRobot("비상 정지")
            time.sleep(0.05)

    def launch_home(self):
        if self.busy or not self.running or self.feedback_mode:
            return
        self.busy = True
        self.abort.clear()
        self.update_buttons()
        self.worker = threading.Thread(target=self.home_job, daemon=True)
        self.worker.start()

    def home_job(self):
        try:
            auto5.reset_to_home()
        finally:
            self.events.put(("idle",))

    def launch_action(self, ball):
        auto5.print_detection(ball)
        quality = "불량" if ball["is_defective"] else "정상"
        self.log(
            f"{quality} 확정 ({ball['row']},{ball['col']}) - 1초 뒤 시작 "
            "(취소: 비상 정지)"
        )
        self.busy = True
        self.abort.clear()
        self.armed = False
        self.clear_tracking()
        self.update_buttons()
        self.worker = threading.Thread(target=self.action_job, args=(ball,), daemon=True)
        self.worker.start()

    def action_job(self, ball):
        try:
            self.interruptible_sleep(1.0)  # 취소 가능한 1초 대기
            log_id = auto5.save_detection_to_db(ball)
            result = auto5.pick_and_place(
                ball["row"], ball["col"], ball["is_defective"], None, "both"
            )
            auto5.update_action_in_db(log_id, result)
            self.events.put(("action_done", ball, result))
        except AbortRobot:
            self.events.put(("aborted", ball, None))
        except Exception as error:
            self.events.put(("error", ball, str(error)))

    def emergency_stop(self):
        if not self.running or self.feedback_mode:
            return
        if self.busy:
            self.abort.set()
            self.log("비상 정지 요청 - 남은 동작을 중단합니다.")
        if self.armed:
            self.armed = False
            self.clear_tracking()
            self.update_buttons()

    def process_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return

            kind = event[0]
            if kind == "idle":
                self.busy = False
            elif kind == "action_done":
                _, ball, result = event
                self.busy = False
                self.last_action_time = time.time()
                self.waiting_for_clear = True
                self.clear_count = 0
                self.armed = False
                if result["success"]:
                    key = "damaged" if ball["is_defective"] else "normal"
                    self.stats[key] += 1
                    self.history.append(key)
                    self.log(f"분류 완료: {'불량' if ball['is_defective'] else '정상'}")
                else:
                    self.stats["error"] += 1
                    self.log(f"동작 실패: {result['error_msg']}")
                self.refresh_stats()
            elif kind in ("aborted", "error"):
                self.busy = False
                self.waiting_for_clear = True
                self.clear_count = 0
                self.armed = False
                self.stats["error"] += 1
                detail = "비상 정지로 중단됨" if kind == "aborted" else f"오류: {event[2]}"
                self.log(f"{detail} - 로봇 위치와 공을 직접 확인하세요.")
                self.refresh_stats()
            self.update_buttons()

    # ------------------------------------------------------------
    # 검사 승인 (SPACE)
    # ------------------------------------------------------------
    def toggle_arm(self):
        if not self.running or self.feedback_mode or self.busy or self.waiting_for_clear:
            return
        if auto5.COOLDOWN_SEC - (time.time() - self.last_action_time) > 0:
            return
        self.armed = not self.armed
        self.clear_tracking()
        if self.armed:
            self.armed_at = time.time()
            self.log("검사 승인 - 손을 화면 밖에 두세요.")
        else:
            self.log("검사를 취소했습니다.")
        self.update_buttons()

    # ------------------------------------------------------------
    # 자동 분류 상태 머신 (run_system의 로직을 그대로 옮김)
    # ------------------------------------------------------------
    def auto_step(self, balls, error, now):
        cooldown = auto5.COOLDOWN_SEC - (now - self.last_action_time)
        action = None
        color = (0, 200, 255)

        if cooldown > 0:
            self.clear_tracking()
            message = f"Cooldown {cooldown:.1f}s"
        elif error is not None:
            self.clear_tracking()
            self.clear_count = 0
            message, color = error, (0, 0, 255)
        elif self.waiting_for_clear:
            self.clear_tracking()
            self.armed = False
            self.clear_count = self.clear_count + 1 if not balls else 0
            if self.clear_count >= auto5.CLEAR_FRAMES_REQUIRED:
                self.waiting_for_clear = False
                self.clear_count = 0
                message, color = "Ready for next ball", (0, 255, 0)
            else:
                message = f"BLOCKED: remove ball {self.clear_count}/{auto5.CLEAR_FRAMES_REQUIRED}"
                color = (0, 0, 255)
        elif not self.armed:
            self.clear_tracking()
            message = "PAUSED: 공을 놓고 손을 치운 뒤 [검사 시작]"
        elif now - self.armed_at < auto5.MANUAL_ARM_SETTLE_SEC:
            self.clear_tracking()
            remain = auto5.MANUAL_ARM_SETTLE_SEC - (now - self.armed_at)
            message = f"ARMED: settling {remain:.1f}s"
        elif len(balls) == 0:
            self.clear_tracking()
            message = "Waiting for one white ball"
        elif len(balls) > 1:
            self.clear_tracking()
            message, color = f"BLOCKED: {len(balls)} candidates", (0, 0, 255)
        else:
            current = balls[0]
            if auto5.is_same_stable_ball(self.stable_ball, current):
                self.stable_count += 1
            else:
                self.stable_count = 1
            self.stable_ball = current
            need = auto5.STABLE_FRAMES_REQUIRED
            if self.stable_count >= need:
                action = current
                message, color = f"Confirmed {need}/{need}", (0, 160, 0)
            else:
                message = f"Checking {self.stable_count}/{need}"
        return action, message, color

    # ------------------------------------------------------------
    # 피드백 저장 (N / D)
    # ------------------------------------------------------------
    def save_feedback(self, label):
        if not self.running or not self.feedback_mode:
            return
        now = time.time()
        if self.last_error is not None or len(self.last_balls) != 1 or self.last_raw is None:
            self.feedback_note = ("NOT SAVED: 공이 정확히 1개 필요합니다", (0, 0, 255))
            self.feedback_note_until = now + 1.5
            return
        if now - self.last_feedback_save < 0.35:
            return
        try:
            path = auto5.save_feedback_sample(
                self.last_raw, self.last_balls[0], label, self.args.feedback_dir, self.camera_index
            )
        except (OSError, cv2.error) as error:
            self.feedback_note = ("SAVE FAILED", (0, 0, 255))
            self.feedback_note_until = now + 1.5
            self.log(f"저장 실패: {error}")
            return
        self.saved_counts[label] += 1
        self.last_feedback_save = now
        ball = self.last_balls[0]
        self.feedback_note = (f"Saved {label}", (0, 160, 0))
        self.feedback_note_until = now + 1.2
        self.log(
            f"저장 {label} (ML={ball['debug']['ml_label']}, "
            f"margin={ball['debug']['ml_margin']:.2f}) {os.path.basename(path)}"
        )
        self.refresh_stats()

    # ------------------------------------------------------------
    # 메인 루프 (Tk after)
    # ------------------------------------------------------------
    def tick(self):
        if not self.running:
            return
        self.process_events()

        ok, frame = self.cap.read()
        if not ok:
            self.fail_count += 1
            if self.fail_count >= auto5.FAIL_LIMIT:
                self.stop_system("카메라 신호가 없어 종료했습니다.")
                return
            self.root.after(100, self.tick)
            return
        self.fail_count = 0
        frame = auto5.flip_frame(frame, self.flip_var.get())
        now = time.time()

        if self.busy:
            display = frame.copy()
            cv2.putText(display, "Robot Moving...", (10, 50), FONT, 1.2, (0, 165, 255), 3)
            self.set_status("로봇 동작 중...", (0, 165, 255))
        else:
            balls, mask, edges, error = auto5.analyze_frame(
                frame, self.grid_w, self.grid_h, self.classifier
            )
            self.last_raw = frame
            self.last_balls = balls
            self.last_error = error
            self.last_mask = mask
            self.last_edges = edges
            display = auto5.draw_detection_screen(frame, balls, self.grid_w, self.grid_h)

            if self.feedback_mode:
                if self.feedback_note is not None and now < self.feedback_note_until:
                    message, color = self.feedback_note
                elif error is not None:
                    message, color = error, (0, 0, 255)
                elif len(balls) != 1:
                    message, color = "공이 정확히 1개 감지되어야 저장할 수 있습니다", (0, 160, 200)
                else:
                    message = (
                        f"ML={balls[0]['debug']['ml_label']} "
                        f"margin={balls[0]['debug']['ml_margin']:.2f}  →  N 또는 D로 정답 저장"
                    )
                    color = (0, 0, 0)
                self.set_status(message, color)
            else:
                action, message, color = self.auto_step(balls, error, now)
                self.set_status(message, color)
                label = CANCEL_TEXT if self.armed else ARM_TEXT
                if self.arm_btn.cget("text") != label:
                    self.update_buttons()
                if action is not None:
                    self.launch_action(action)

        self.show(display)
        self.root.after(10, self.tick)

    def show(self, display):
        view = self.view_var.get()
        if view == "mask" and self.last_mask is not None:
            image = cv2.cvtColor(self.last_mask, cv2.COLOR_GRAY2BGR)
        elif view == "edges" and self.last_edges is not None:
            image = cv2.cvtColor(self.last_edges, cv2.COLOR_GRAY2BGR)
        else:
            image = display

        # 영상 영역 크기에 맞춰 비율을 유지하며 확대/축소
        height, width = image.shape[:2]
        box_w = max(self.holder.winfo_width(), 320)
        box_h = max(self.holder.winfo_height(), 240)
        scale = min(box_w / width, box_h / height, MAX_DISPLAY_WIDTH / width)
        if abs(scale - 1.0) > 0.02:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            image = cv2.resize(
                image, (max(int(width * scale), 1), max(int(height * scale), 1)),
                interpolation=interp,
            )

        ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            self.photo = tk.PhotoImage(data=base64.b64encode(buffer.tobytes()))
            self.video.configure(image=self.photo, text="")


def main():
    args = auto5.parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()