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
from tkinter import messagebox, ttk

import cv2
import numpy as np

import auto5

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_DISPLAY_WIDTH = 800


class AbortRobot(Exception):
    """비상 정지 요청 시 로봇 동작 스레드를 중단시키는 예외."""


def bgr_to_hex(color):
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"


class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        root.title("Smart Quality Control System - GUI")
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
    def build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main)
        left.pack(side="left", padx=6, pady=6)
        holder = tk.Frame(left, width=MAX_DISPLAY_WIDTH, height=480, bg="black")
        holder.pack_propagate(False)
        holder.pack()
        self.video = tk.Label(
            holder, bg="black", fg="white", text="시스템이 꺼져 있습니다.\n[시스템 시작]을 누르세요."
        )
        self.video.pack(expand=True)
        self.status_label = tk.Label(
            left, text="대기 중", anchor="w", font=("TkDefaultFont", 12, "bold")
        )
        self.status_label.pack(fill="x", pady=(4, 0))

        right = tk.Frame(main)
        right.pack(side="right", fill="y", padx=6, pady=6)

        # --- 설정 ---
        box = tk.LabelFrame(right, text="설정")
        box.pack(fill="x", pady=3)
        self.mode_var = tk.StringVar(value="feedback" if self.args.feedback else "auto")
        self.camera_var = tk.StringVar(
            value=str(self.args.camera if self.args.camera is not None else auto5.DEFAULT_CAMERA_INDEX)
        )
        self.port_var = tk.StringVar(value=self.args.serial_port)
        self.sim_var = tk.BooleanVar(value=self.args.simulation)
        self.flip_var = tk.StringVar(value=self.args.flip)

        self.setting_widgets = []
        r1 = tk.Radiobutton(box, text="자동 분류", variable=self.mode_var, value="auto")
        r2 = tk.Radiobutton(box, text="피드백 수집", variable=self.mode_var, value="feedback")
        r1.grid(row=0, column=0, sticky="w")
        r2.grid(row=0, column=1, sticky="w")
        tk.Label(box, text="카메라 번호").grid(row=1, column=0, sticky="w")
        cam = tk.Spinbox(box, from_=0, to=9, width=6, textvariable=self.camera_var)
        cam.grid(row=1, column=1, sticky="w")
        tk.Label(box, text="시리얼 포트").grid(row=2, column=0, sticky="w")
        port = tk.Entry(box, width=10, textvariable=self.port_var)
        port.grid(row=2, column=1, sticky="w")
        sim = tk.Checkbutton(box, text="시뮬레이션 (로봇 미연결)", variable=self.sim_var)
        sim.grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Label(box, text="화면 뒤집기").grid(row=4, column=0, sticky="w")
        flip = ttk.Combobox(
            box,
            width=10,
            state="readonly",
            textvariable=self.flip_var,
            values=("both", "horizontal", "vertical", "none"),
        )
        flip.grid(row=4, column=1, sticky="w")
        self.setting_widgets = [r1, r2, cam, port, sim]

        self.start_btn = tk.Button(
            box, text="시스템 시작", bg="#2e7d32", fg="white", takefocus=0, command=self.toggle_system
        )
        self.start_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)

        # --- 조작 ---
        ctl = tk.LabelFrame(right, text="조작")
        ctl.pack(fill="x", pady=3)
        self.arm_btn = tk.Button(
            ctl, text="검사 시작 (Space)", takefocus=0, command=self.toggle_arm
        )
        self.arm_btn.pack(fill="x", pady=1)
        self.estop_btn = tk.Button(
            ctl, text="비상 정지 (Esc)", bg="#c62828", fg="white", takefocus=0,
            command=self.emergency_stop,
        )
        self.estop_btn.pack(fill="x", pady=1)
        self.home_btn = tk.Button(ctl, text="홈 위치로 이동", takefocus=0, command=self.launch_home)
        self.home_btn.pack(fill="x", pady=1)
        fb = tk.Frame(ctl)
        fb.pack(fill="x", pady=1)
        self.ok_btn = tk.Button(
            fb, text="정상 저장 (N)", takefocus=0, command=lambda: self.save_feedback("normal")
        )
        self.ok_btn.pack(side="left", expand=True, fill="x")
        self.bad_btn = tk.Button(
            fb, text="불량 저장 (D)", takefocus=0, command=lambda: self.save_feedback("damaged")
        )
        self.bad_btn.pack(side="left", expand=True, fill="x")

        # --- 흰색 HSV ---
        hsv = tk.LabelFrame(right, text="흰색 인식 (HSV)")
        hsv.pack(fill="x", pady=3)
        self.v_var = tk.IntVar(value=int(auto5.LOWER_WHITE[2]))
        self.s_var = tk.IntVar(value=int(auto5.UPPER_WHITE[1]))
        tk.Scale(
            hsv, from_=100, to=255, orient="horizontal", variable=self.v_var,
            label="밝기 V 하한 (높을수록 엄격)", command=self.on_hsv,
        ).pack(fill="x")
        tk.Scale(
            hsv, from_=0, to=255, orient="horizontal", variable=self.s_var,
            label="채도 S 상한 (낮을수록 엄격)", command=self.on_hsv,
        ).pack(fill="x")

        # --- 보기 ---
        view = tk.LabelFrame(right, text="화면 보기")
        view.pack(fill="x", pady=3)
        self.view_var = tk.StringVar(value="video")
        for text, value in (("영상", "video"), ("흰색 마스크", "mask"), ("엣지", "edges")):
            tk.Radiobutton(view, text=text, variable=self.view_var, value=value).pack(side="left")

        # --- 통계 / 로그 ---
        info = tk.LabelFrame(right, text="통계")
        info.pack(fill="x", pady=3)
        self.stats_var = tk.StringVar()
        tk.Label(info, textvariable=self.stats_var, justify="left", anchor="w").pack(fill="x")

        logf = tk.LabelFrame(right, text="로그")
        logf.pack(fill="both", expand=True, pady=3)
        self.log_box = tk.Text(logf, width=38, height=10, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True)

        self.refresh_stats()
        self.update_buttons()

    # ------------------------------------------------------------
    # 공통 유틸
    # ------------------------------------------------------------
    def key_ok(self):
        focused = self.root.focus_get()
        return not isinstance(focused, (tk.Entry, ttk.Entry, tk.Spinbox))

    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, text, color=(0, 0, 0)):
        self.status_label.configure(text=text, fg=bgr_to_hex(color))

    def refresh_stats(self):
        text = (
            f"정상 {self.stats['normal']}  /  불량 {self.stats['damaged']}  /  "
            f"오류 {self.stats['error']}"
        )
        if self.feedback_mode:
            text += (
                f"\n피드백 저장: 정상 {self.saved_counts['normal']}, "
                f"불량 {self.saved_counts['damaged']}"
            )
        self.stats_var.set(text)

    def update_buttons(self):
        auto_on = self.running and not self.feedback_mode
        fb_on = self.running and self.feedback_mode
        self.arm_btn.configure(
            state="normal" if auto_on else "disabled",
            text="검사 취소 (Space)" if self.armed else "검사 시작 (Space)",
        )
        self.estop_btn.configure(state="normal" if auto_on else "disabled")
        self.home_btn.configure(state="normal" if auto_on and not self.busy else "disabled")
        self.ok_btn.configure(state="normal" if fb_on else "disabled")
        self.bad_btn.configure(state="normal" if fb_on else "disabled")
        for widget in self.setting_widgets:
            widget.configure(state="disabled" if self.running else "normal")
        self.start_btn.configure(
            text="시스템 종료" if self.running else "시스템 시작",
            bg="#c62828" if self.running else "#2e7d32",
        )

    def on_hsv(self, _=None):
        # make_white_mask()가 호출될 때마다 전역값을 읽으므로 즉시 반영된다.
        auto5.LOWER_WHITE = np.array([0, 0, int(self.v_var.get())], dtype=np.uint8)
        auto5.UPPER_WHITE = np.array([179, int(self.s_var.get()), 255], dtype=np.uint8)

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
            self.camera_index, self.args.auto_exposure, self.args.exposure
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
        height, width = probe.shape[:2]
        self.grid_w = max(width // 3, 1)
        self.grid_h = max(height // 3, 1)

        self.reset_state()
        self.running = True
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
        self.video.configure(image="", text="시스템이 꺼져 있습니다.\n[시스템 시작]을 누르세요.")
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
                label = "검사 취소 (Space)" if self.armed else "검사 시작 (Space)"
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

        height, width = image.shape[:2]
        if width > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / width
            image = cv2.resize(image, (MAX_DISPLAY_WIDTH, int(height * scale)))

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