"""
========================================
ARTME 3D MK2 — Data Monitor 
========================================
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import re
import csv
import math
from collections import deque
from datetime import datetime

try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

try:
    from prediction import DEFAULT_SEQUENCE_LENGTH, predict_diameter
    PREDICTION_OK = True
except Exception as e:
    DEFAULT_SEQUENCE_LENGTH = 40
    predict_diameter = None
    PREDICTION_OK = False
    print(f"Prediction not available: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit these to match your setup
# ══════════════════════════════════════════════════════════════════════════════
SERIAL_PORT   = "COM4"      # ARTME extruder port
BAUD_RATE     = 250000
MESSUHR_PORT  = "COM6"      # Dial gauge (diameter sensor) port
READ_INTERVAL = 1           # seconds between ARTME polls

# Diameter target / tolerance (defines the green band on the graph)
DIA_TARGET    = 1.75
DIA_TOL       = 0.06
TEMP_TOL      = 3.0
SEN_HI        = 8.0
SEN_LO        = 5.0

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG      = "#f7f8fa"     # main window background (soft off-white)
CARD    = "#ffffff"     # card / panel surfaces
LINE    = "#e3e6 ea".replace(" ", "")  # hairline borders
INK     = "#1f2933"     # primary text
MUTED   = "#7b8794"     # secondary text
FAINT   = "#cbd2d9"     # very light text / grid
ACCENT  = "#2563eb"     # blue accent
GOOD    = "#1f9d57"     # in-tolerance green
BAD     = "#e0392b"     # out-of-tolerance red
BAND    = "#eef4ff"     # graph tolerance band fill
BANDEDGE= "#9bb8e8"     # graph band edge
GRID    = "#eef0f3"     # graph grid lines

FONT    = "Segoe UI"    # clean modern sans (falls back gracefully)
FONT_MONO = "Consolas"  # for numeric readouts


# ══════════════════════════════════════════════════════════════════════════════
#  MESSUHR  (diameter sensor reader) 
# ══════════════════════════════════════════════════════════════════════════════
class Messuhr2:
    def __init__(self, port, baud=115200):
        self.port      = port
        self.baud      = baud
        self.diameter  = None
        self.connected = False
        self._lock     = threading.Lock()
        self._history  = []
        self._last_raw = ""
        self._try_connect()

    def _try_connect(self):
        if not SERIAL_OK:
            self.connected = False
            return
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(1.5)
            self.connected = True
            threading.Thread(target=self._read_loop, daemon=True).start()
        except Exception as e:
            self.connected = False
            print(f"Messuhr not connected: {e}")

    def _parse_line(self, line):
        try:
            m = re.search(r'MM:([-\d\.]+)', line)
            if not m:
                return None
            mm_value = float(m.group(1))
            if mm_value < -100 or mm_value > 100:
                return None
            return round(mm_value, 3)
        except:
            return None

    def _read_loop(self):
        while True:
            try:
                raw  = self.ser.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if line == "":
                    continue
                self._last_raw = line
                val = self._parse_line(line)
                ts  = time.time()
                with self._lock:
                    if val is not None:
                        self.diameter = val
                        self._history.append((ts, val))
                        if len(self._history) > 6000:
                            self._history.pop(0)
            except Exception:
                self.connected = False
                break

    def get(self):
        with self._lock:
            return self.diameter

    def disconnect(self):
        try:
            self.connected = False
            self.ser.close()
        except:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  SERIAL HELPERS  (ARTME) 
# ══════════════════════════════════════════════════════════════════════════════
def send_command(ser, cmd):
    try:
        ser.reset_input_buffer()
        time.sleep(0.1)
        ser.write((cmd + '\n').encode())
        lines = []
        deadline = time.time() + 0.8
        while time.time() < deadline:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        lines.append(line)
                except:
                    pass
            else:
                time.sleep(0.02)
        return '\n'.join(lines)
    except:
        return ""

def parse_m105(text):
    actual, target = 0.0, 0.0
    try:
        for line in text.split('\n'):
            if 'ok T:' in line:
                a = re.search(r'T:([\d.]+)', line)
                t = re.search(r'T:[\d.]+ /([\d.]+)', line)
                if a: actual = float(a.group(1))
                if t: target = float(t.group(1))
                break
    except:
        pass
    return actual, target

def parse_m700(text):
    data = {}
    try:
        for line in text.split('\n'):
            if 'ok D:' in line:
                for key, pat in {
                    'length_mm':        r'L:([\d.]+)',
                    'fil_width':        r'Fil:([\d.]+)',
                    'extruder_rpm':     r'E:([\d.]+)',
                    'puller_speed_raw': r'P:([\d.]+)',
                    'actual_temp':      r'T0:([\d.]+)',
                    'target_temp':      r'T0:[\d.]+ /([\d.]+)',
                }.items():
                    m = re.search(pat, line)
                    if m: data[key] = float(m.group(1))
                p_raw = data.get('puller_speed_raw', 0)
                data['puller_speed'] = round(p_raw * 1.5898, 2)
                w = re.search(r'\bW:(\d+)', line)
                data['fan_pwm'] = int(w.group(1)) if w else 0
                data['fan_pct'] = round(data['fan_pwm'] / 2.55)
                break
    except:
        pass
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  CSV RECORDING  
# ══════════════════════════════════════════════════════════════════════════════
CSV_HEADERS = [
    "Timestamp", "Diameter (mm)", "Temp Act (C)", "Temp Tgt (C)",
    "Extruder RPM", "Puller RPM", "Sensor Pos", "Fan %",
]

def csv_start_session():
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"recording_{ts}.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["ARTME 3D MK2 — Recording Session"])
    w.writerow([f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    w.writerow([f"Target Diameter: {DIA_TARGET:.2f} mm   Tolerance: +/-{DIA_TOL:.2f} mm"])
    w.writerow([])
    w.writerow(CSV_HEADERS)
    f.flush()
    return f, w, path

def csv_write_row(writer, diameter, artme_data):
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{diameter:.3f}" if diameter is not None else "",
        f"{artme_data.get('actual_temp', 0):.1f}",
        f"{artme_data.get('target_temp', 0):.1f}",
        f"{artme_data.get('extruder_rpm', 0):.1f}",
        f"{artme_data.get('puller_speed', 0):.2f}",
        f"{artme_data.get('fil_width', 0):.2f}",
        f"{artme_data.get('fan_pct', 0)}",
    ])

def csv_close_session(f, writer, diameter_readings):
    if not diameter_readings:
        f.close()
        return
    lo = DIA_TARGET - DIA_TOL
    hi = DIA_TARGET + DIA_TOL
    valid  = [d for d in diameter_readings if d is not None]
    exceed = [d for d in valid if d < lo or d > hi]
    pct_ok = 100.0 * (len(valid) - len(exceed)) / len(valid) if valid else 0
    writer.writerow([])
    writer.writerow(["--- SESSION SUMMARY ---"])
    writer.writerow([f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    writer.writerow([f"Total samples: {len(valid)}"])
    writer.writerow([f"Within tolerance: {len(valid)-len(exceed)} ({pct_ok:.1f}%)"])
    writer.writerow([f"Exceeded tolerance: {len(exceed)}"])
    writer.writerow(["STATUS: " + ("DIAMETER EXCEEDED TOLERANCE" if exceed
                                   else "FILAMENT WITHIN LIMITS — ALL READINGS OK")])
    f.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ARTME 3D MK2 — Data Monitor")
        self.root.geometry("1180x820")
        self.root.minsize(960, 700)
        self.root.configure(bg=BG)

        # runtime state
        self.ser            = None
        self.running        = False
        self.reading_no     = 0
        self.start_time     = None
        self.current_data   = {}
        self.messuhr_obj    = None
        self.artme_online   = False
        self.messuhr_online = False
        self._destroyed     = False
        self._prediction_error_reported = False
        self._prediction_feature_history = deque(maxlen=DEFAULT_SEQUENCE_LENGTH)

        # recording state
        self._rec_active    = False
        self._rec_file      = None
        self._rec_writer    = None
        self._rec_path      = ""
        self._rec_diameters = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_metric_cards()
        self._build_record_bar()
        self._build_graph()
        self.root.after(300, self._start_graph_update)

    # ── HEADER ────────────────────────────────────────────────────────────────
    def _build_header(self):
        bar = tk.Frame(self.root, bg=CARD, height=58)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Title block
        title_wrap = tk.Frame(bar, bg=CARD)
        title_wrap.pack(side="left", padx=20)
        tk.Label(title_wrap, text="ARTME 3D MK2", bg=CARD, fg=INK,
                 font=(FONT, 15, "bold")).pack(anchor="w", pady=(9, 0))
        tk.Label(title_wrap, text="Data monitor", bg=CARD, fg=MUTED,
                 font=(FONT, 9)).pack(anchor="w")

        # Connection chips (right)
        right = tk.Frame(bar, bg=CARD)
        right.pack(side="right", padx=18)

        self.dot_artme, _ = self._conn_chip(
            right, "ARTME", self._toggle_artme)
        tk.Frame(right, bg=LINE, width=1, height=28).pack(side="left", padx=10, pady=14)
        self.dot_messuhr, _ = self._conn_chip(
            right, "MESSUHR", self._toggle_messuhr)

        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x")

    def _conn_chip(self, parent, label, command):
        chip = tk.Frame(parent, bg=CARD)
        chip.pack(side="left")
        dot = tk.Label(chip, text="●", bg=CARD, fg=BAD, font=(FONT, 12))
        dot.pack(side="left", padx=(0, 5))
        tk.Label(chip, text=label, bg=CARD, fg=INK,
                 font=(FONT, 10, "bold")).pack(side="left", padx=(0, 8))
        btn = tk.Button(chip, text="Connect", command=command,
                        bg=CARD, fg=ACCENT, font=(FONT, 9),
                        relief="flat", bd=0, cursor="hand2",
                        activebackground="#eef4ff", activeforeground=ACCENT,
                        highlightthickness=1, highlightbackground=BANDEDGE,
                        padx=12, pady=3)
        btn.pack(side="left")
        return dot, btn

    # ── METRIC CARDS ──────────────────────────────────────────────────────────
    def _build_metric_cards(self):
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="x", padx=16, pady=14)

        self._metric_defs = [
            ("Diameter",      "mm",    "fil_width"),
            ("Diameter predicted", "mm", "fil_width_predicted"),
            ("Temp act / tgt","°C",    "actual_temp"),
            ("Extruder",      "rpm",   "extruder_rpm"),
            ("Puller",        "mm/s",  "puller_speed"),
            ("Sensor pos",    "",      "fil_width_sen"),
            ("Fan speed",     "%",     "fan_pct"),
        ]
        self._metric_labels = {}

        for i, (lbl, unit, key) in enumerate(self._metric_defs):
            card = tk.Frame(wrap, bg=CARD, highlightthickness=1,
                            highlightbackground=LINE)
            card.grid(row=0, column=i, sticky="nsew", padx=5)
            wrap.columnconfigure(i, weight=1)

            head = tk.Frame(card, bg=CARD)
            head.pack(fill="x", padx=12, pady=(10, 0))
            tk.Label(head, text=lbl.upper(), bg=CARD, fg=MUTED,
                     font=(FONT, 8, "bold")).pack(side="left")
            if unit:
                tk.Label(head, text=unit, bg=CARD, fg=FAINT,
                         font=(FONT, 8)).pack(side="right")

            val = tk.Label(card, text="--", bg=CARD, fg=INK,
                           font=(FONT_MONO, 22, "bold"), anchor="w")
            val.pack(anchor="w", padx=12, pady=(2, 12))
            self._metric_labels[key] = val

    # ── RECORD BAR ────────────────────────────────────────────────────────────
    def _build_record_bar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=21, pady=(0, 6))

        self.rec_btn = tk.Button(
            bar, text="●  Start recording",
            command=self._toggle_recording,
            bg=CARD, fg=BAD, font=(FONT, 10, "bold"),
            relief="flat", bd=0, cursor="hand2",
            activebackground="#fdeceb", activeforeground=BAD,
            highlightthickness=1, highlightbackground="#f3b6b1",
            padx=16, pady=6)
        self.rec_btn.pack(side="left")

        self.rec_info = tk.Label(bar, text="Not recording",
                                 bg=BG, fg=MUTED, font=(FONT, 9))
        self.rec_info.pack(side="left", padx=14)

        self.reading_lbl = tk.Label(bar, text="R:0  |  0s",
                                    bg=BG, fg=MUTED, font=(FONT_MONO, 9))
        self.reading_lbl.pack(side="right")

    # ── GRAPH ─────────────────────────────────────────────────────────────────
    def _build_graph(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=21, pady=(6, 18))

        card = tk.Frame(outer, bg=CARD, highlightthickness=1,
                        highlightbackground=LINE)
        card.pack(fill="both", expand=True)

        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(head, text="LIVE DIAMETER", bg=CARD, fg=INK,
                 font=(FONT, 10, "bold")).pack(side="left")
        tk.Label(head,
                 text=f"target {DIA_TARGET:.2f} mm   ±{DIA_TOL:.2f}",
                 bg=CARD, fg=MUTED, font=(FONT, 9)).pack(side="left", padx=12)

        self.graph_cv = tk.Canvas(card, bg=CARD, highlightthickness=0, bd=0)
        self.graph_cv.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Nonlinear Y transform (expands the in-tolerance band)
        tgt, tol = DIA_TARGET, DIA_TOL
        dia_lo, dia_hi = tgt - tol, tgt + tol
        EXPAND, COMPRESS = 10.0, 0.5

        def _tf(v):
            if dia_lo <= v <= dia_hi:
                return (v - tgt) * EXPAND
            elif v > dia_hi:
                return (dia_hi - tgt) * EXPAND + (v - dia_hi) * COMPRESS
            else:
                return (dia_lo - tgt) * EXPAND + (v - dia_lo) * COMPRESS

        self._y_transform = _tf
        self._g_tgt, self._g_tol = tgt, tol
        self._g_dia_lo, self._g_dia_hi = dia_lo, dia_hi
        self._g_static_ok = False

        self.graph_cv.bind("<Configure>", self._graph_on_resize)

    def _graph_on_resize(self, event=None):
        if hasattr(self, '_resize_job'):
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self._graph_draw_static)

    def _graph_draw_static(self):
        cv = self.graph_cv
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            self.root.after(100, self._graph_draw_static)
            return

        self._g_w, self._g_h = w, h
        tgt    = self._g_tgt
        dia_lo = self._g_dia_lo
        dia_hi = self._g_dia_hi
        tf     = self._y_transform

        def ty(tv):
            return (1.0 - (tv + 1.5) / 3.0) * h

        y_band_top = ty(tf(dia_hi))
        y_band_bot = ty(tf(dia_lo))
        y_centre   = ty(tf(tgt))

        cv.delete("static")

        # Horizontal grid lines
        for yf in [0.15, 0.30, 0.50, 0.70, 0.85]:
            yp = yf * h
            cv.create_line(0, yp, w, yp, fill=GRID, width=1, tags="static")

        # Tolerance band — light blue fill
        cv.create_rectangle(0, y_band_top, w, y_band_bot,
                            fill=BAND, outline="", tags="static")
        cv.create_line(0, y_band_top, w, y_band_top,
                       fill=BANDEDGE, width=1, tags="static")
        cv.create_line(0, y_band_bot, w, y_band_bot,
                       fill=BANDEDGE, width=1, tags="static")

        # Centre target line — dashed accent
        dash, gap, x = 7, 11, 0
        while x < w:
            x2 = min(x + dash, w)
            cv.create_line(x, y_centre, x2, y_centre,
                           fill=ACCENT, width=1, tags="static")
            x += dash + gap

        # Y-axis labels
        label_defs = [
            (tgt + 0.5, FAINT, 8),
            (dia_hi,    ACCENT, 9),
            (tgt,       MUTED, 8),
            (dia_lo,    ACCENT, 9),
            (tgt - 0.5, FAINT, 8),
        ]
        for rv, col, sz in label_defs:
            tv = tf(rv)
            if -1.5 <= tv <= 1.5:
                yp = ty(tv)
                cv.create_text(8, yp, text=f"{rv:.2f}", fill=col,
                               font=(FONT_MONO, sz), anchor="w", tags="static")

        self._g_static_ok = True

    def _start_graph_update(self):
        from collections import deque
        self._col_buf = None
        self._col_w   = 0
        self._col_h   = 0
        self.root.after(33, self._update_graph)

    def _update_graph(self):
        if self._destroyed:
            return
        self.root.after(33, self._update_graph)

        if not self.messuhr_obj or not self._g_static_ok:
            return

        from collections import deque
        cv = self.graph_cv
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            return

        if self._col_buf is None or self._col_w != w or self._col_h != h:
            self._col_buf = deque([None] * w, maxlen=w)
            self._col_w, self._col_h = w, h

        d = self.messuhr_obj.get()
        if d is None:
            self._col_buf.append(None)
        else:
            tgt, tol = self.cfg_tgt(), self.cfg_tol()
            lo, hi = tgt - tol, tgt + tol
            tf = self._y_transform
            in_b = lo <= d <= hi
            tv = tf(d)
            py = int((1.0 - (tv + 1.5) / 3.0) * (h - 1))
            py = max(0, min(h - 1, py))
            self._col_buf.append((py, in_b))

        cv.delete("data")
        buf = self._col_buf
        if len(buf) < 2:
            return
        x0 = w - len(buf)

        green_segs, red_segs = [], []
        g_cur, r_cur, prev = [], [], None

        for i, entry in enumerate(buf):
            x = x0 + i
            if entry is None:
                if len(g_cur) >= 4: green_segs.append(g_cur)
                if len(r_cur) >= 4: red_segs.append(r_cur)
                g_cur, r_cur, prev = [], [], None
                continue
            py, in_b = entry
            if prev is not None and prev[1] != in_b:
                if in_b:
                    r_cur.extend([x, py])
                    if len(r_cur) >= 4: red_segs.append(r_cur)
                    r_cur = []
                    g_cur = [x, py]
                else:
                    g_cur.extend([x, py])
                    if len(g_cur) >= 4: green_segs.append(g_cur)
                    g_cur = []
                    r_cur = [x, py]
            else:
                if in_b: g_cur.extend([x, py])
                else:    r_cur.extend([x, py])
            prev = (py, in_b)

        if len(g_cur) >= 4: green_segs.append(g_cur)
        if len(r_cur) >= 4: red_segs.append(r_cur)

        for seg in red_segs:
            cv.create_line(seg, fill=BAD, width=2,
                           capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="data")
        for seg in green_segs:
            cv.create_line(seg, fill=GOOD, width=2,
                           capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="data")

    def cfg_tgt(self): return DIA_TARGET
    def cfg_tol(self): return DIA_TOL

    # ──────────────────────────────────────────────────────────────────────────
    #  METRIC UPDATES
    # ──────────────────────────────────────────────────────────────────────────
    def _current_measured_diameter(self):
        if not self.messuhr_online or not self.messuhr_obj:
            return None

        diameter = self.messuhr_obj.get()
        if diameter is None or not math.isfinite(diameter):
            return None

        return diameter

    def _format_prediction_with_error(self, prediction):
        if prediction is None or not math.isfinite(prediction):
            return "--", INK

        measured_diameter = self._current_measured_diameter()
        if measured_diameter is None:
            return (
                f"{prediction:.3f}",
                GOOD if abs(prediction - DIA_TARGET) <= DIA_TOL else BAD,
            )

        prediction_error = abs(prediction - measured_diameter)
        return (
            f"{prediction:.3f} +/- {prediction_error:.3f}",
            GOOD if prediction_error <= DIA_TOL else BAD,
        )

    def _update_metrics(self, data):
        self.current_data = data
        act    = data.get("actual_temp", 0)
        tgt    = data.get("target_temp", 0)
        rpm    = data.get("extruder_rpm", 0)
        fan    = data.get("fan_pct", 0)
        sen    = data.get("fil_width", 0)
        puller = data.get("puller_speed", 0)
        pred   = data.get("fil_width_predicted")

        if not self.messuhr_online:
            self._metric_labels["fil_width"].config(text="--", fg=INK)

        self._metric_labels["actual_temp"].config(
            text=f"{act:.1f}/{tgt:.0f}" if act > 0 else "--",
            fg=(GOOD if abs(act - tgt) <= TEMP_TOL else BAD) if act > 0 else INK)
        prediction_text, prediction_color = self._format_prediction_with_error(pred)
        self._metric_labels["fil_width_predicted"].config(
            text=prediction_text, fg=prediction_color)
        self._metric_labels["extruder_rpm"].config(
            text=f"{rpm:.1f}" if rpm > 0 else "--", fg=INK)
        self._metric_labels["puller_speed"].config(
            text=f"{puller:.2f}" if puller > 0 else "--", fg=INK)
        self._metric_labels["fan_pct"].config(
            text=str(int(fan)) if fan else "--", fg=INK)
        self._metric_labels["fil_width_sen"].config(
            text=f"{sen:.2f}" if sen > 0 else "--",
            fg=(GOOD if SEN_LO <= sen <= SEN_HI else BAD) if sen > 0 else INK)

        self.reading_lbl.config(
            text=f"R:{data.get('reading_no',0)}  |  {data.get('elapsed',0):.0f}s")

    # ──────────────────────────────────────────────────────────────────────────
    #  CONNECTION — ARTME
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_artme(self):
        if not self.artme_online:
            self._connect_artme()
        else:
            self._disconnect_artme()

    def _connect_artme(self):
        if not SERIAL_OK:
            print("pyserial not installed")
            return
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
            time.sleep(2)
            send_command(self.ser, "M115")
            self.artme_online = True
            self.running      = True
            self.start_time   = time.time()
            self.reading_no   = 0
            self._prediction_feature_history.clear()
            self.dot_artme.config(fg=GOOD)
            threading.Thread(target=self._data_loop, daemon=True).start()
        except Exception as e:
            print(f"ARTME connection failed: {e}")

    def _disconnect_artme(self):
        self.running      = False
        self.artme_online = False
        if self.ser:
            try: self.ser.close()
            except: pass
        self.dot_artme.config(fg=BAD)

    # ──────────────────────────────────────────────────────────────────────────
    #  CONNECTION — MESSUHR
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_messuhr(self):
        if not self.messuhr_online:
            self._connect_messuhr()
        else:
            self._disconnect_messuhr()

    def _connect_messuhr(self):
        if self.messuhr_obj and self.messuhr_obj.connected:
            self.messuhr_obj.disconnect()
        self.messuhr_obj = Messuhr2(MESSUHR_PORT)
        if self.messuhr_obj.connected:
            self.messuhr_online = True
            self.dot_messuhr.config(fg=GOOD)
            self._update_messuhr_display()
        else:
            print(f"Messuhr failed to connect on {MESSUHR_PORT}")

    def _disconnect_messuhr(self):
        if self.messuhr_obj:
            self.messuhr_obj.disconnect()
        self.messuhr_online = False
        self.dot_messuhr.config(fg=BAD)

    def _update_messuhr_display(self):
        if self._destroyed or not self.messuhr_obj:
            return
        if self.messuhr_online and not self.messuhr_obj.connected:
            self.messuhr_online = False
            self.dot_messuhr.config(fg=BAD)

        d = self.messuhr_obj.get()
        if d is not None:
            ok = abs(d - DIA_TARGET) <= DIA_TOL
            self._metric_labels["fil_width"].config(
                text=f"{d:.3f}", fg=GOOD if ok else BAD)
        else:
            self._metric_labels["fil_width"].config(text="--", fg=INK)

        self.root.after(100, self._update_messuhr_display)

    def _predict_current_diameter(self, data):
        if not PREDICTION_OK or predict_diameter is None:
            return None

        temp_act = data.get("actual_temp", 0)
        extruder_rpm = data.get("extruder_rpm", 0)
        puller_rpm = data.get("puller_speed", 0)
        sensor_pos = data.get("fil_width", 0)
        if min(temp_act, extruder_rpm, puller_rpm, sensor_pos) <= 0:
            self._prediction_feature_history.clear()
            return None

        feature_row = [temp_act, extruder_rpm, puller_rpm, sensor_pos]
        self._prediction_feature_history.append(feature_row)
        if len(self._prediction_feature_history) < DEFAULT_SEQUENCE_LENGTH:
            return None

        try:
            prediction_mm = predict_diameter(list(self._prediction_feature_history))
            self._prediction_error_reported = False
            return prediction_mm
        except Exception as e:
            if not self._prediction_error_reported:
                print(f"Prediction failed: {e}")
                self._prediction_error_reported = True
            return None

    # ──────────────────────────────────────────────────────────────────────────
    #  DATA LOOP  (ARTME polling)
    # ──────────────────────────────────────────────────────────────────────────
    def _data_loop(self):
        while self.running:
            try:
                loop_start = time.time()
                self.reading_no += 1
                elapsed = round(time.time() - self.start_time, 1)

                m700_resp = send_command(self.ser, "M700")
                m700_data = parse_m700(m700_resp)

                m105_resp = send_command(self.ser, "M105")
                actual_temp, target_temp = parse_m105(m105_resp)
                if actual_temp == 0:
                    actual_temp, target_temp = parse_m105(m700_resp)

                data = {
                    'reading_no':   self.reading_no,
                    'elapsed':      elapsed,
                    'actual_temp':  actual_temp,
                    'target_temp':  target_temp,
                    'extruder_rpm': 0.0,
                    'puller_speed': 0.0,
                    'fil_width':    0.0,
                    'fil_width_predicted': None,
                    'length_mm':    0.0,
                    'fan_pct':      0,
                }
                if m700_data:
                    data.update(m700_data)
                    if actual_temp > 0: data['actual_temp'] = actual_temp
                    if target_temp > 0: data['target_temp'] = target_temp

                data['fil_width_predicted'] = self._predict_current_diameter(data)

                self.root.after(0, self._update_metrics, data)

                elapsed_loop = time.time() - loop_start
                time.sleep(max(0, READ_INTERVAL - elapsed_loop))

            except Exception as e:
                print(f"Data loop error: {e}")
                if self.ser and not self.ser.is_open:
                    self.running = False
                    self.root.after(0, self._artme_lost)
                time.sleep(READ_INTERVAL)

    def _artme_lost(self):
        self.artme_online = False
        self.running      = False
        self.dot_artme.config(fg=BAD)

    # ──────────────────────────────────────────────────────────────────────────
    #  CSV RECORDING
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_recording(self):
        if not self._rec_active:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        try:
            f, w, path = csv_start_session()
            self._rec_file      = f
            self._rec_writer    = w
            self._rec_path      = path
            self._rec_active    = True
            self._rec_diameters = []
            self.rec_btn.config(text="■  Stop recording",
                                fg="#ffffff", bg=BAD,
                                activebackground="#c4302a",
                                activeforeground="#ffffff",
                                highlightbackground=BAD)
            self.rec_info.config(text=f"Recording → {path}", fg=GOOD)
            self._csv_tick()
        except Exception as e:
            self.rec_info.config(text=f"Failed to start: {e}", fg=BAD)

    def _stop_recording(self):
        self._rec_active = False
        dia_values = [v for _, v in self._rec_diameters if v is not None]
        try:
            csv_close_session(self._rec_file, self._rec_writer, dia_values)
        except Exception as e:
            print(f"Error closing CSV: {e}")
        self._rec_file = None
        self._rec_writer = None
        self.rec_btn.config(text="●  Start recording",
                            fg=BAD, bg=CARD,
                            activebackground="#fdeceb",
                            activeforeground=BAD,
                            highlightbackground="#f3b6b1")
        n = len(dia_values)
        self.rec_info.config(text=f"Saved {n} samples → {self._rec_path}", fg=MUTED)

    def _csv_tick(self):
        if not self._rec_active:
            return
        try:
            dia = self.messuhr_obj.get() if self.messuhr_obj else None
            ts  = datetime.now().strftime("%H:%M:%S")
            self._rec_diameters.append((ts, dia))
            csv_write_row(self._rec_writer, dia, self.current_data)
            self._rec_file.flush()
        except Exception as e:
            print(f"CSV write error: {e}")
        self.root.after(1000, self._csv_tick)

    # ──────────────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._destroyed = True
        if self._rec_active:
            try: self._stop_recording()
            except: pass
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = MonitorApp(root)
    root.mainloop()
