import sys
import json
import time
import os
import threading
from typing import Optional

import numpy as np
import serial
import serial.tools.list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QSlider, QComboBox,
    QSpinBox, QCheckBox, QPlainTextEdit, QFileDialog, QMessageBox, QLineEdit,
    QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────
HOME         = [90, 90, 90, 90, 70]
GRIPPER_MIN  = 20
GRIPPER_MAX  = 110
JOINT_NAMES  = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper"]
JOINT_RANGES = [(0, 180), (0, 180), (0, 180), (0, 180), (20, 110)]
LINK_LEN     = [1.4, 1.2, 0.8, 0.5]
BASE_HEIGHT  = 0.45
BAUD_RATE    = 115200
SAFE_LIFT    = 28
SAVE_FILE    = "saved_commands.json"

def clamp_gripper(v: float) -> int:
    return int(max(GRIPPER_MIN, min(GRIPPER_MAX, v)))

def clamp_joint(v: float, idx: int) -> int:
    lo, hi = JOINT_RANGES[idx]
    return int(max(lo, min(hi, v)))


# ─── Serial Worker ─────────────────────────────────────────────────────────────
class SerialWorker(QThread):
    log_signal    = pyqtSignal(str)
    status_signal = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.ser: Optional[serial.Serial] = None
        self.running = True
        self._queue: list[str] = []
        self._lock  = threading.Lock()

    @property
    def connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def connect(self, port: str) -> bool:
        try:
            self.ser = serial.Serial(
                port, BAUD_RATE,
                timeout=0.1,
                write_timeout=0.5,
            )
            time.sleep(0.3)
            self.ser.reset_input_buffer()
            self.status_signal.emit(True)
            self.log_signal.emit(f"[CONN] Connected → {port} @ {BAUD_RATE}")
            return True
        except Exception as e:
            self.log_signal.emit(f"[ERR] {e}")
            self.status_signal.emit(False)
            return False

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.status_signal.emit(False)
        self.log_signal.emit("[CONN] Disconnected")

    def send_raw(self, angles: list) -> None:
        safe = angles[:]
        safe[4] = clamp_gripper(safe[4])
        cmd = ",".join(str(int(a)) for a in safe) + "\n"
        with self._lock:
            self._queue = [cmd]

    def run(self) -> None:
        while self.running:
            with self._lock:
                cmds, self._queue = self._queue[:], []
            for cmd in cmds:
                if self.connected:
                    try:
                        self.ser.write(cmd.encode())
                        self.log_signal.emit(f"[TX] {cmd.strip()}")
                    except serial.SerialTimeoutException:
                        self.log_signal.emit("[WARN] Serial write timeout")
                    except Exception as e:
                        self.log_signal.emit(f"[ERR TX] {e}")

            if self.connected:
                try:
                    while self.ser.in_waiting:
                        line = self.ser.readline().decode(errors="ignore").strip()
                        if line:
                            self.log_signal.emit(f"[RX] {line}")
                except Exception:
                    pass
            self.msleep(15)


# ─── Motion Player ─────────────────────────────────────────────────────────────
class MotionPlayer(QThread):
    angles_signal   = pyqtSignal(list)
    finished_signal = pyqtSignal()
    log_signal      = pyqtSignal(str)
    step_signal     = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.sequence: list[list] = []
        self.speed    = 1.0
        self.loop     = False
        self._paused  = False
        self._stopped = False
        self.current  = HOME[:]

    def load(self, seq: list[list]) -> None:
        self.sequence = [f[:] for f in seq]

    def pause(self) -> None:
        self._paused = not self._paused

    def stop(self) -> None:
        self._stopped = True
        self._paused  = False

    def _interp(self, frm: list, to: list, steps: int):
        a = np.array(frm, float)
        b = np.array(to,  float)
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3.0 - 2.0 * t)
            frame = list(a + (b - a) * t)
            frame[4] = clamp_gripper(frame[4])
            yield frame

    def run(self) -> None:
        self._stopped = False
        total = len(self.sequence)
        while True:
            for idx, target in enumerate(self.sequence):
                if self._stopped:
                    break
                while self._paused:
                    self.msleep(50)
                    if self._stopped:
                        break
                if self._stopped:
                    break
                steps = max(8, int(28 / max(0.1, self.speed)))
                for frame in self._interp(self.current, target, steps):
                    if self._stopped:
                        break
                    self.angles_signal.emit(frame)
                    self.current = frame
                    self.msleep(int(18 / max(0.1, self.speed)))
                self.step_signal.emit(idx + 1, total)
            if not self.loop or self._stopped:
                break
        self.finished_signal.emit()


# ─── 3D OpenGL Arm Viewport ────────────────────────────────────────────────────
class ArmViewport(QOpenGLWidget):
    joint_drag_signal = pyqtSignal(int, float)

    def __init__(self) -> None:
        super().__init__()
        self._angles  = HOME[:]
        self._target  = HOME[:]
        self._rot_x   = 18.0
        self._rot_y   = -32.0
        self._zoom    = 1.0
        self._last_pt = None
        self._dark_mode = True
        self.setMinimumSize(450, 490)
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(16)

    def set_theme(self, dark_mode: bool):
        self._dark_mode = dark_mode
        self.update()

    def set_angles(self, angles: list) -> None:
        self._target = angles[:]

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        self._zoom += delta * 0.001
        self._zoom = max(0.5, min(2.0, self._zoom))
        self.update()

    def _tick(self) -> None:
        changed = False
        for i in range(5):
            d = self._target[i] - self._angles[i]
            if abs(d) > 0.22:
                self._angles[i] += d * 0.22
                changed = True
            else:
                self._angles[i] = self._target[i]
        if changed:
            self.update()

    def initializeGL(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHT2)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(44.0, w / max(h, 1), 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        if self._dark_mode:
            glClearColor(0.12, 0.12, 0.12, 1.0)
            glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 7.0, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1.0])
            glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1.0])
            glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, 4.0, 2.0, 1.0])
            glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.3, 0.3, 0.4, 1.0])
            grid_c1, grid_c2 = (0.2, 0.2, 0.2), (0.15, 0.15, 0.15)
        else:
            glClearColor(0.96, 0.92, 0.90, 1.0)
            glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 7.0, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.7, 0.7, 0.7, 1.0])
            glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.4, 0.4, 0.4, 1.0])
            glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, 4.0, 2.0, 1.0])
            glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.5, 0.5, 0.5, 1.0])
            grid_c1, grid_c2 = (0.85, 0.82, 0.80), (0.9, 0.87, 0.85)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 3, 10,   0, 2.4, 0,   0, 1, 0)
        glScalef(self._zoom, self._zoom, self._zoom)
        glRotatef(self._rot_x, 1, 0, 0)
        glRotatef(self._rot_y, 0, 1, 0)
        
        self._draw_grid(grid_c1, grid_c2)
        self._draw_arm(self._angles)

    def _box(self, w: float, h: float, d: float) -> None:
        hx, hy, hz = w * 0.5, h * 0.5, d * 0.5
        faces = [
            ((0, 0, 1),  [(-hx,-hy,hz),(hx,-hy,hz),(hx,hy,hz),(-hx,hy,hz)]),
            ((0, 0,-1),  [(hx,-hy,-hz),(-hx,-hy,-hz),(-hx,hy,-hz),(hx,hy,-hz)]),
            ((0, 1, 0),  [(-hx,hy,hz),(hx,hy,hz),(hx,hy,-hz),(-hx,hy,-hz)]),
            ((0,-1, 0),  [(-hx,-hy,-hz),(hx,-hy,-hz),(hx,-hy,hz),(-hx,-hy,hz)]),
            ((1, 0, 0),  [(hx,-hy,hz),(hx,-hy,-hz),(hx,hy,-hz),(hx,hy,hz)]),
            ((-1,0, 0),  [(-hx,-hy,-hz),(-hx,-hy,hz),(-hx,hy,hz),(-hx,hy,-hz)]),
        ]
        glBegin(GL_QUADS)
        for norm, verts in faces:
            glNormal3f(*norm)
            for v in verts:
                glVertex3f(*v)
        glEnd()

    def _box_link(self, width: float, depth: float, length: float) -> None:
        glPushMatrix()
        glTranslatef(0, length * 0.5, 0)
        self._box(width, length, depth)
        glPopMatrix()

    def _disc(self, radius: float, thickness: float, slices: int = 32) -> None:
        glPushMatrix()
        glTranslatef(-thickness * 0.5, 0, 0)
        glRotatef(90, 0, 1, 0)
        q = gluNewQuadric()
        gluQuadricNormals(q, GLU_SMOOTH)
        gluCylinder(q, radius, radius, thickness, slices, 2)
        gluDisk(q, 0.04, radius, slices, 1)
        glTranslatef(0, 0, thickness)
        gluDisk(q, 0.04, radius, slices, 1)
        gluDeleteQuadric(q)
        glPopMatrix()

    def _draw_axes(self) -> None:
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        sz = 0.55
        glBegin(GL_LINES)
        glColor3f(0.9, 0.2, 0.2); glVertex3f(0,0,0); glVertex3f(sz,0,0)
        glColor3f(0.2, 0.9, 0.2); glVertex3f(0,0,0); glVertex3f(0,sz,0)
        glColor3f(0.2, 0.4, 0.9); glVertex3f(0,0,0); glVertex3f(0,0,sz)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _draw_grid(self, c1, c2) -> None:
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        for i in range(-7, 8):
            glColor3f(*(c1 if i==0 else c2))
            glVertex3f(i, 0, -7); glVertex3f(i, 0, 7)
            glVertex3f(-7, 0, i); glVertex3f(7, 0, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_arm(self, a: list) -> None:
        if self._dark_mode:
            BASE_C, HUB_C, LINK_C, DISC_C, GRIP_C = (0.7,0.7,0.7), (0.6,0.6,0.6), (0.8,0.8,0.8), (0.5,0.5,0.5), (0.1,0.1,0.1)
        else:
            BASE_C, HUB_C, LINK_C, DISC_C, GRIP_C = (0.4,0.4,0.4), (0.5,0.5,0.5), (0.3,0.3,0.3), (0.6,0.6,0.6), (0.2,0.2,0.2)

        glPushMatrix()
        glColor3f(*BASE_C)
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        q = gluNewQuadric(); gluQuadricNormals(q, GLU_SMOOTH)
        gluCylinder(q, 1.08, 0.94, 0.56, 44, 3)
        gluDisk(q, 0, 1.08, 44, 1)
        glTranslatef(0, 0, 0.56)
        gluDisk(q, 0, 0.94, 44, 1)
        gluDeleteQuadric(q)
        glPopMatrix()

        glColor3f(*HUB_C)
        glTranslatef(0, 0.56, 0)
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        q = gluNewQuadric(); gluQuadricNormals(q, GLU_SMOOTH)
        gluCylinder(q, 0.32, 0.24, 0.22, 36, 2)
        gluDisk(q, 0, 0.32, 36, 1)
        glTranslatef(0, 0, 0.22)
        gluDisk(q, 0, 0.24, 36, 1)
        gluDeleteQuadric(q)
        glPopMatrix()

        glTranslatef(0, 0.22, 0)
        glRotatef(a[0] - 90, 0, 1, 0)

        glColor3f(*DISC_C)
        self._disc(0.28, 0.20)
        glRotatef(-(a[1] - 90), 1, 0, 0)
        glColor3f(*LINK_C)
        self._box_link(0.25, 0.16, LINK_LEN[0])

        glTranslatef(0, LINK_LEN[0], 0)
        glColor3f(*DISC_C)
        self._disc(0.22, 0.17)
        glRotatef(+(a[2] - 90), 1, 0, 0)
        glColor3f(*LINK_C)
        self._box_link(0.21, 0.14, LINK_LEN[1])

        glTranslatef(0, LINK_LEN[1], 0)
        glColor3f(*DISC_C)
        self._disc(0.16, 0.14)
        glColor3f(0.8, 0.1, 0.1)
        glPushMatrix()
        glTranslatef(0.14, 0.04, 0)
        self._box(0.10, 0.09, 0.07)
        glPopMatrix()
        glRotatef(-(a[3] - 90), 1, 0, 0)
        glColor3f(*LINK_C)
        self._box_link(0.16, 0.11, LINK_LEN[2])

        glTranslatef(0, LINK_LEN[2], 0)
        glColor3f(*DISC_C)
        self._disc(0.11, 0.10)

        t = (a[4] - GRIPPER_MIN) / max(1, GRIPPER_MAX - GRIPPER_MIN)
        spread = 0.044 + t * 0.135

        glColor3f(*GRIP_C)
        for side in (-1.0, 1.0):
            glPushMatrix()
            glTranslatef(side * spread, 0, 0)

            glPushMatrix()
            glTranslatef(0, 0.20, 0)
            self._box(0.060, 0.34, 0.092)
            glPopMatrix()

            glPushMatrix()
            glTranslatef(-side * 0.048, 0.37, 0)
            self._box(0.108, 0.055, 0.092)
            glPopMatrix()

            glPushMatrix()
            glTranslatef(-side * 0.048, 0.055, 0)
            self._box(0.108, 0.055, 0.092)
            glPopMatrix()
            glPopMatrix()

        glPopMatrix()
        self._draw_axes()

    def mousePressEvent(self, e) -> None:
        self._last_pt = e.position()

    def mouseMoveEvent(self, e) -> None:
        if self._last_pt is not None:
            dx = e.position().x() - self._last_pt.x()
            dy = e.position().y() - self._last_pt.y()
            
            if e.buttons() & Qt.MouseButton.LeftButton:
                self._rot_y += dx * 0.50
                self._rot_x  = max(-88, min(88, self._rot_x + dy * 0.50))
            elif e.buttons() & Qt.MouseButton.RightButton:
                self.joint_drag_signal.emit(0, dx * 0.3)
                self.joint_drag_signal.emit(1, -dy * 0.3)
            elif e.buttons() & Qt.MouseButton.MiddleButton:
                self.joint_drag_signal.emit(2, -dy * 0.3)
                self.joint_drag_signal.emit(3, dx * 0.3)

            self._last_pt = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._last_pt = None


# ─── Main Window ──────────────────────────────────────────────────────────────
class RoboArmPro(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RoboArm Pro  —  Professional Control System")
        self.resize(1400, 900)
        
        self.dark_mode = True
        self.setStyleSheet(DARK_STYLE)

        self.angles    = HOME[:]
        self.recording = False
        self.sequence: list[list] = []
        
        self.saved_slots = {}
        self._load_slots_from_disk()

        self._send_timer = QTimer(self)
        self._send_timer.setSingleShot(True)
        self._send_timer.setInterval(40)
        self._send_timer.timeout.connect(self._flush_serial)

        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(150)
        self._rec_timer.timeout.connect(self._record_frame)

        self.serial_w = SerialWorker()
        self.serial_w.log_signal.connect(self._log)
        self.serial_w.status_signal.connect(self._on_connection)
        self.serial_w.start()

        self.player = MotionPlayer()
        self.player.angles_signal.connect(self._apply_angles)
        self.player.finished_signal.connect(self._on_play_done)
        self.player.log_signal.connect(self._log)
        self.player.step_signal.connect(
            lambda s, t: self.statusBar().showMessage(f"Playing  step {s}/{t}")
        )
        
        self.joystick = None
        if PYGAME_AVAILABLE:
            pygame.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
        
        self._gamepad_timer = QTimer(self)
        self._gamepad_timer.setInterval(50)
        self._gamepad_timer.timeout.connect(self._poll_gamepad)
        self._gamepad_timer.start()

        self._build_ui()
        self.viewport.joint_drag_signal.connect(self._on_joint_drag)

    def _on_joint_drag(self, joint_idx: int, delta: float) -> None:
        new_val = self.angles[joint_idx] + delta
        new_val = clamp_joint(new_val, joint_idx)
        self.angles[joint_idx] = new_val
        self._apply_angles(self.angles)

    def _on_gamepad_toggle(self, checked: bool) -> None:
        if checked:
            self._gamepad_btn.setText("🎮 Xbox Controller [ACTIVE]")
        else:
            self._gamepad_btn.setText("🎮 Connect Xbox Controller")

    def _poll_gamepad(self):
        if not hasattr(self, '_gamepad_btn') or not self._gamepad_btn.isChecked() or not self.joystick:
            return
        pygame.event.pump()
        dz = 0.15
        
        ax_base = self.joystick.get_axis(0)
        ax_shoulder = self.joystick.get_axis(1)
        ax_elbow = self.joystick.get_axis(4) if self.joystick.get_numaxes() > 4 else 0
        ax_wrist = self.joystick.get_axis(3) if self.joystick.get_numaxes() > 3 else 0
        
        hat = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0,0)
        
        deltas = [0]*5
        if abs(ax_base) > dz: deltas[0] = ax_base * 2.0
        if abs(ax_shoulder) > dz: deltas[1] = -ax_shoulder * 2.0
        if abs(ax_elbow) > dz: deltas[2] = -ax_elbow * 2.0
        if abs(ax_wrist) > dz: deltas[3] = ax_wrist * 2.0
        if hat[1] != 0: deltas[4] = hat[1] * 2.0
        
        changed = False
        for i in range(5):
            if deltas[i] != 0:
                self.angles[i] = clamp_joint(self.angles[i] + deltas[i], i)
                if i == 4:
                    self.angles[i] = clamp_gripper(self.angles[i])
                changed = True
        
        if changed:
            self._apply_angles(self.angles)

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.setStyleSheet(DARK_STYLE)
            self._theme_btn.setText("☀ Switch to Light Mode")
        else:
            self.setStyleSheet(PASTEL_STYLE)
            self._theme_btn.setText("☾ Switch to Dark Mode")
        self.viewport.set_theme(self.dark_mode)

    def _corrected_angles(self, angles: list) -> list:
        out = angles[:]
        for i in range(5):
            if self._invert_cb[i].isChecked():
                lo, hi = JOINT_RANGES[i]
                out[i] = lo + hi - out[i]
        if self._swap_cb.isChecked():
            out[3], out[4] = out[4], out[3]
        return out

    def _display_angles(self, angles: list) -> list:
        out = angles[:]
        for i in range(5):
            if self._invert_cb[i].isChecked():
                lo, hi = JOINT_RANGES[i]
                out[i] = lo + hi - out[i]
        return out

    def _flush_serial(self) -> None:
        self.serial_w.send_raw(self._corrected_angles(self.angles))

    def _send_now(self, angles: list) -> None:
        self.serial_w.send_raw(self._corrected_angles(angles))

    # ─── New Dashboard UI Layout ───────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        
        main_layout = QVBoxLayout(root)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header Bar
        header = QWidget()
        header.setObjectName("headerPanel")
        header.setMinimumHeight(60)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(15, 10, 15, 10)
        
        app_title = QLabel("RoboArm Pro Workspace")
        app_title.setObjectName("appTitle")
        h_layout.addWidget(app_title)
        
        self._port_cb = QComboBox()
        self._port_cb.setMinimumWidth(100)
        self._refresh_ports()
        
        btn_refresh = QPushButton("↻")
        btn_refresh.setObjectName("btnSecondary")
        btn_refresh.setFixedWidth(30)
        btn_refresh.clicked.connect(self._refresh_ports)
        
        self._btn_conn = QPushButton("CONNECT")
        self._btn_conn.setObjectName("btnPrimary")
        self._btn_conn.clicked.connect(self._toggle_conn)
        
        self._gamepad_btn = QPushButton("🎮 Connect Xbox Controller")
        self._gamepad_btn.setObjectName("btnSecondary")
        self._gamepad_btn.setCheckable(True)
        self._gamepad_btn.toggled.connect(self._on_gamepad_toggle)
        if not PYGAME_AVAILABLE:
            self._gamepad_btn.setEnabled(False)
            self._gamepad_btn.setToolTip("Install 'pygame' to use Xbox Controller")
            
        self._theme_btn = QPushButton("☀ Switch to Light Mode")
        self._theme_btn.setObjectName("btnSecondary")
        self._theme_btn.clicked.connect(self._toggle_theme)
        
        h_layout.addStretch()
        for w in [self._port_cb, btn_refresh, self._btn_conn, self._gamepad_btn, self._theme_btn]:
            h_layout.addWidget(w)
            
        main_layout.addWidget(header)
        
        # Content Area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left Panel: Viewport + Sequence Recorder / Tabs
        left_panel = QWidget()
        l_layout = QVBoxLayout(left_panel)
        l_layout.setContentsMargins(0,0,0,0)
        
        self.viewport = ArmViewport()
        l_layout.addWidget(self.viewport, 3)
        
        # Sequence Recorder Top Controls
        seq_group = QGroupBox("Sequence Recorder & Saved Tabs")
        seq_v = QVBoxLayout(seq_group)
        
        seq_top = QHBoxLayout()
        self._btn_rec = QPushButton("⏺ Record"); self._btn_rec.setObjectName("btnDanger")
        self._btn_rec.clicked.connect(self._toggle_record)
        btn_add = QPushButton("+ Frame"); btn_add.setObjectName("btnSecondary")
        btn_add.clicked.connect(self._add_frame)
        btn_play = QPushButton("▶ Play Workspace"); btn_play.setObjectName("btnPrimary")
        btn_play.clicked.connect(self._play_seq)
        btn_pause = QPushButton("⏸"); btn_pause.setObjectName("btnSecondary")
        btn_pause.clicked.connect(self.player.pause)
        btn_stop = QPushButton("⏹"); btn_stop.setObjectName("btnSecondary")
        btn_stop.clicked.connect(self._stop_player)
        self._loop_cb = QCheckBox("Loop playback")
        self._loop_cb.stateChanged.connect(lambda s: setattr(self.player, "loop", bool(s)))
        btn_clr = QPushButton("✕ Clear"); btn_clr.setObjectName("btnSecondary")
        btn_clr.clicked.connect(self._clear_seq)
        self._seq_lbl = QLabel("Active Workspace: 0 frames")
        
        for w in [self._btn_rec, btn_add, btn_play, btn_pause, btn_stop, self._loop_cb, btn_clr, self._seq_lbl]:
            seq_top.addWidget(w)
        seq_v.addLayout(seq_top)
        
        # Professional Tabs for Saved Commands
        self.tab_widget = QTabWidget()
        self._tab_labels = {}
        for i in range(1, 11):
            k = str(i)
            slot_data = self.saved_slots.get(k, {"name": f"Slot {i}", "sequence": []})
            
            w = QWidget()
            t_layout = QHBoxLayout(w)
            
            # Left side of tab: Info and Rename
            info_v = QVBoxLayout()
            h_rename = QHBoxLayout()
            name_edit = QLineEdit(slot_data["name"])
            name_edit.setPlaceholderText("Sequence Name...")
            btn_rename = QPushButton("Rename Tab")
            btn_rename.setObjectName("btnSecondary")
            btn_rename.clicked.connect(lambda _, x=k, e=name_edit: self._rename_tab(x, e))
            h_rename.addWidget(name_edit)
            h_rename.addWidget(btn_rename)
            info_v.addLayout(h_rename)
            
            lbl_info = QLabel(f"Length: {len(slot_data['sequence'])} frames")
            info_v.addWidget(lbl_info)
            self._tab_labels[k] = lbl_info
            
            t_layout.addLayout(info_v)
            t_layout.addStretch()
            
            # Right side of tab: Save/Load Actions
            act_v = QVBoxLayout()
            btn_save = QPushButton("Save Workspace ➔ to this Tab")
            btn_save.setObjectName("btnPrimary")
            btn_save.clicked.connect(lambda _, x=k: self._save_to_tab(x))
            
            btn_load = QPushButton("Load from this Tab ➔ Workspace")
            btn_load.setObjectName("btnSecondary")
            btn_load.clicked.connect(lambda _, x=k: self._load_from_tab(x))
            
            act_v.addWidget(btn_save)
            act_v.addWidget(btn_load)
            t_layout.addLayout(act_v)
            
            self.tab_widget.addTab(w, slot_data["name"])
            
        seq_v.addWidget(self.tab_widget)
        l_layout.addWidget(seq_group, 1)
        
        content_layout.addWidget(left_panel, 3)
        
        # Right Panel: Sliders, Automation, Console
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        r_layout.setContentsMargins(0,0,0,0)
        
        r_layout.addWidget(self._sliders_panel())
        r_layout.addWidget(self._pick_panel())
        r_layout.addWidget(self._log_panel())
        
        content_layout.addWidget(right_panel, 2)
        
        main_layout.addLayout(content_layout)
        self.statusBar().showMessage("Ready — professional workspace loaded")

    # ─── Tabs Manager Logic ─────────────────────────────────────────
    def _load_slots_from_disk(self):
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, "r") as f:
                    self.saved_slots = json.load(f)
            except:
                pass
        for i in range(1, 11):
            k = str(i)
            if k not in self.saved_slots:
                self.saved_slots[k] = {"name": f"Slot {i}", "sequence": []}
                
    def _save_slots_to_disk(self):
        with open(SAVE_FILE, "w") as f:
            json.dump(self.saved_slots, f, indent=2)

    def _rename_tab(self, slot_id: str, edit_widget: QLineEdit):
        new_name = edit_widget.text().strip()
        if new_name:
            self.saved_slots[slot_id]["name"] = new_name
            idx = int(slot_id) - 1
            self.tab_widget.setTabText(idx, new_name)
            self._save_slots_to_disk()
            self._log(f"[SEQ] Tab {slot_id} renamed to '{new_name}'")

    def _save_to_tab(self, slot_id: str):
        if not self.sequence:
            QMessageBox.warning(self, "Empty", "Active workspace sequence is empty.")
            return
        self.saved_slots[slot_id]["sequence"] = [f[:] for f in self.sequence]
        self._save_slots_to_disk()
        self._tab_labels[slot_id].setText(f"Length: {len(self.sequence)} frames")
        self._log(f"[SEQ] Saved {len(self.sequence)} frames to Tab {slot_id}")

    def _load_from_tab(self, slot_id: str):
        seq = self.saved_slots[slot_id]["sequence"]
        if not seq:
            QMessageBox.information(self, "Empty Tab", "This tab has no saved frames.")
            return
        self.sequence = [f[:] for f in seq]
        self._seq_lbl.setText(f"Active Workspace: {len(self.sequence)} frames")
        self._log(f"[SEQ] Loaded {len(self.sequence)} frames from Tab {slot_id}")

    # ─── Control Panels ─────────────────────────────────────────────
    def _sliders_panel(self) -> QGroupBox:
        group = QGroupBox("Joint Manual Controls")
        gl = QGridLayout(group)
        gl.setVerticalSpacing(8)
        self._sliders:   list[QSlider]   = []
        self._aval:      list[QLabel]    = []
        self._invert_cb: list[QCheckBox] = []

        # Home & Swap row
        top_h = QHBoxLayout()
        btn_home = QPushButton("⌂ Reset to Home")
        btn_home.setObjectName("btnPrimary")
        btn_home.clicked.connect(self._go_home)
        self._swap_cb = QCheckBox("Swap Wrist/Gripper channels")
        top_h.addWidget(btn_home)
        top_h.addWidget(self._swap_cb)
        gl.addLayout(top_h, 0, 0, 1, 4)

        for i, (name, (lo, hi)) in enumerate(zip(JOINT_NAMES, JOINT_RANGES)):
            lbl = QLabel(name)
            sl  = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi); sl.setValue(HOME[i])
            val = QLabel(f"{HOME[i]}°")
            inv = QCheckBox("Inv")
            sl.valueChanged.connect(lambda v, ix=i: self._on_slider(ix, v))
            inv.stateChanged.connect(lambda _, ix=i: self._refresh_viewport())
            
            row = i + 1
            gl.addWidget(lbl, row, 0)
            gl.addWidget(sl,  row, 1)
            gl.addWidget(val, row, 2)
            gl.addWidget(inv, row, 3)
            self._sliders.append(sl)
            self._aval.append(val)
            self._invert_cb.append(inv)

        spl = QLabel("Playback Speed")
        self._spd_sl  = QSlider(Qt.Orientation.Horizontal)
        self._spd_sl.setRange(1, 10); self._spd_sl.setValue(5)
        self._spd_val = QLabel("1.0×")
        self._spd_sl.valueChanged.connect(self._on_speed)
        
        gl.addWidget(spl, 6, 0)
        gl.addWidget(self._spd_sl, 6, 1)
        gl.addWidget(self._spd_val, 6, 2)
        return group

    def _pick_panel(self) -> QGroupBox:
        group = QGroupBox("Pick & Place Automation")
        layout = QVBoxLayout(group)
        grid = QGridLayout()
        grid.addWidget(QLabel("Joint"), 0, 0)
        grid.addWidget(QLabel("Pick °"), 0, 1)
        grid.addWidget(QLabel("Place °"), 0, 2)

        labels        = ["Base", "Shoulder", "Elbow", "Wrist"]
        pick_def      = [90, 55, 120, 90]
        place_def     = [40, 55, 120, 90]
        self._pick_spins:  list[QSpinBox] = []
        self._place_spins: list[QSpinBox] = []
        for i, name in enumerate(labels):
            grid.addWidget(QLabel(name), i + 1, 0)
            ps = QSpinBox(); ps.setRange(0, 180); ps.setValue(pick_def[i])
            pl = QSpinBox(); pl.setRange(0, 180); pl.setValue(place_def[i])
            grid.addWidget(ps, i + 1, 1); grid.addWidget(pl, i + 1, 2)
            self._pick_spins.append(ps); self._place_spins.append(pl)

        grid.addWidget(QLabel("Gripper"), 5, 0)
        self._pick_grip  = QSpinBox()
        self._pick_grip.setRange(GRIPPER_MIN, GRIPPER_MAX)
        self._pick_grip.setValue(GRIPPER_MAX - 8)
        self._place_grip = QSpinBox()
        self._place_grip.setRange(GRIPPER_MIN, GRIPPER_MAX)
        self._place_grip.setValue(GRIPPER_MIN + 5)
        grid.addWidget(self._pick_grip,  5, 1)
        grid.addWidget(self._place_grip, 5, 2)
        layout.addLayout(grid)

        btns = QHBoxLayout()
        btn_exec = QPushButton("Execute Routine")
        btn_exec.setObjectName("btnPrimary")
        btn_exec.clicked.connect(self._exec_pick_place)
        btn_prev = QPushButton("Preview Path")
        btn_prev.setObjectName("btnSecondary")
        btn_prev.clicked.connect(self._preview_path)
        btns.addWidget(btn_exec); btns.addWidget(btn_prev)
        layout.addLayout(btns)
        return group

    def _log_panel(self) -> QGroupBox:
        group = QGroupBox("System Console")
        v = QVBoxLayout(group)
        self._log_box = QPlainTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumBlockCount(300)
        self._log_box.setFixedHeight(120)
        btn_clr = QPushButton("Clear Console")
        btn_clr.setObjectName("btnSecondary")
        btn_clr.clicked.connect(self._log_box.clear)
        v.addWidget(self._log_box)
        v.addWidget(btn_clr, alignment=Qt.AlignmentFlag.AlignRight)
        return group

    # ─── Sliders / Playback / Actions ───────────────────────────────
    def _on_slider(self, idx: int, val: int) -> None:
        self.angles[idx] = val
        self._aval[idx].setText(f"{val}°")
        self.viewport.set_angles(self._display_angles(self.angles))
        self._send_timer.start()

    def _refresh_viewport(self) -> None:
        self.viewport.set_angles(self._display_angles(self.angles))

    def _on_speed(self, val: int) -> None:
        spd = val / 5.0
        self.player.speed = spd
        self._spd_val.setText(f"{spd:.1f}×")

    def _apply_angles(self, angles: list) -> None:
        self.angles = angles[:]
        for i, sl in enumerate(self._sliders):
            sl.blockSignals(True)
            sl.setValue(clamp_joint(angles[i], i))
            sl.blockSignals(False)
            self._aval[i].setText(f"{int(angles[i])}°")
        self.viewport.set_angles(self._display_angles(angles))
        self._send_now(angles)

    def _refresh_ports(self) -> None:
        self._port_cb.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self._port_cb.addItems(ports or ["No ports found"])

    def _toggle_conn(self) -> None:
        if self.serial_w.connected:
            self.serial_w.disconnect()
            self._btn_conn.setText("CONNECT"); self._btn_conn.setObjectName("btnPrimary")
        else:
            port = self._port_cb.currentText()
            if port and port != "No ports found":
                ok = self.serial_w.connect(port)
                if ok:
                    self._btn_conn.setText("DISCONNECT")
                    self._btn_conn.setObjectName("btnDanger")
        self._btn_conn.setStyle(self._btn_conn.style())

    def _on_connection(self, connected: bool) -> None:
        if connected:
            self.statusBar().showMessage("Arduino connected — Ready")
        else:
            self.statusBar().showMessage("Disconnected")

    def _go_home(self) -> None:
        self._apply_angles(HOME[:])
        self._log("[HOME] Moved to home position")

    def _toggle_record(self) -> None:
        self.recording = not self.recording
        if self.recording:
            self.sequence = []
            self._seq_lbl.setText("Active Workspace: 0 frames")
            self._rec_timer.start()
            self._btn_rec.setText("⏹ Stop Recording"); self._btn_rec.setObjectName("btnDanger")
            self._log("[REC] Recording started")
        else:
            self._rec_timer.stop()
            self._btn_rec.setText("⏺ Record"); self._btn_rec.setObjectName("btnDanger")
            self._log(f"[REC] Stopped — {len(self.sequence)} frames")
        self._btn_rec.setStyle(self._btn_rec.style())

    def _record_frame(self) -> None:
        frame = self.angles[:]
        frame[4] = clamp_gripper(frame[4])
        self.sequence.append(frame)
        self._seq_lbl.setText(f"Active Workspace: {len(self.sequence)} frames")

    def _add_frame(self) -> None:
        frame = self.angles[:]
        frame[4] = clamp_gripper(frame[4])
        self.sequence.append(frame)
        self._seq_lbl.setText(f"Active Workspace: {len(self.sequence)} frames")
        self._log(f"[REC] Manual frame added")

    def _clear_seq(self) -> None:
        self.sequence = []
        self._seq_lbl.setText("Active Workspace: 0 frames")
        self._log("[REC] Workspace sequence cleared")

    def _play_seq(self) -> None:
        if not self.sequence:
            QMessageBox.warning(self, "No Sequence", "Active workspace has no frames to play.")
            return
        self._stop_player()
        self.player.loop = self._loop_cb.isChecked()
        self.player.load(self.sequence)
        self.player.current = self.angles[:]
        self.player.start()
        self._log(f"[SEQ] Playing {len(self.sequence)} frames ×{self.player.speed:.1f}")

    def _stop_player(self) -> None:
        if self.player.isRunning():
            self.player.stop()
            self.player.wait(1200)

    def _on_play_done(self) -> None:
        self._log("[SEQ] Playback complete")
        self.statusBar().showMessage("Playback finished")

    def _raised(self, joints4: list) -> list:
        j = joints4[:]
        j[1] = clamp_joint(joints4[1] - SAFE_LIFT, 1)
        return j

    def _exec_pick_place(self) -> None:
        pick4  = [s.value() for s in self._pick_spins]
        place4 = [s.value() for s in self._place_spins]
        gc     = clamp_gripper(self._pick_grip.value())
        go     = clamp_gripper(self._place_grip.value())

        seq = [
            self._raised(HOME[:4])  + [go],
            self._raised(pick4)     + [go],
            pick4                   + [go],
            pick4                   + [gc],
            self._raised(pick4)     + [gc],
            self._raised(place4)    + [gc],
            place4                  + [gc],
            place4                  + [go],
            self._raised(place4)    + [go],
            self._raised(HOME[:4])  + [go],
        ]
        for step in seq:
            step[4] = clamp_gripper(step[4])

        self._stop_player()
        self.player.load(seq)
        self.player.current = self.angles[:]
        self.player.start()
        self._log("[P&P] Pick & Place started (10 steps)")

    def _preview_path(self) -> None:
        pick  = [s.value() for s in self._pick_spins]
        place = [s.value() for s in self._place_spins]
        msg = (
            f"PICK   B={pick[0]}°  Sh={pick[1]}°  El={pick[2]}°  "
            f"Wr={pick[3]}°  Grip={self._pick_grip.value()}°\n"
            f"PLACE  B={place[0]}°  Sh={place[1]}°  El={place[2]}°  "
            f"Wr={place[3]}°  Grip={self._place_grip.value()}°\n\n"
        )
        QMessageBox.information(self, "Pick & Place — Path Preview", msg)

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_box.appendPlainText(f"[{ts}] {msg}")
        sb = self._log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event) -> None:
        self._rec_timer.stop()
        self._send_timer.stop()
        self._gamepad_timer.stop()
        if PYGAME_AVAILABLE:
            pygame.quit()
        self.serial_w.running = False
        self.serial_w.disconnect()
        self._stop_player()
        event.accept()


DARK_STYLE = """
* {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
    font-size: 13px;
    color: #E0E2E4;
}
QMainWindow, QWidget { background-color: #1E1E1E; }
#headerPanel { background-color: #252526; border-bottom: 1px solid #333333; }
#appTitle { font-size: 18px; font-weight: bold; color: #4EC9B0; padding: 10px; }
#btnPrimary { background-color: #0E639C; border: none; border-radius: 4px; color: #FFFFFF; font-weight: bold; padding: 6px 16px; }
#btnPrimary:hover { background-color: #1177BB; }
#btnPrimary:pressed { background-color: #094771; }
#btnPrimary:checked { background-color: #009966; }
#btnSecondary { background-color: #3A3D41; border: none; border-radius: 4px; color: #CCCCCC; font-weight: bold; padding: 6px 16px; }
#btnSecondary:hover { background-color: #4C4F53; }
#btnDanger { background-color: #C53A32; border: none; border-radius: 4px; color: #FFFFFF; font-weight: bold; padding: 6px 16px; }
#btnDanger:hover { background-color: #E54B41; }
QGroupBox { background-color: #252526; border: 1px solid #333333; border-radius: 6px; margin-top: 15px; padding-top: 15px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 5px; color: #4EC9B0; font-weight: bold; font-size: 12px; }
QSlider::groove:horizontal { height: 6px; background: #333333; border-radius: 3px; }
QSlider::handle:horizontal { background: #0E639C; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #1177BB; }
QSlider::sub-page:horizontal { background: #0E639C; border-radius: 3px; }
QTabWidget::pane { border: 1px solid #333333; background-color: #252526; border-radius: 4px; }
QTabBar::tab { background-color: #2D2D30; color: #969696; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; border-right: 1px solid #1E1E1E; }
QTabBar::tab:selected { background-color: #252526; color: #FFFFFF; font-weight: bold; border-bottom: 2px solid #0E639C; }
QTabBar::tab:hover:!selected { background-color: #3A3D41; }
QComboBox, QSpinBox, QLineEdit { background-color: #3C3C3C; border: 1px solid #555555; border-radius: 4px; padding: 4px 8px; color: #CCCCCC; }
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border: 1px solid #0E639C; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; background-color: #3C3C3C; border: 1px solid #555555; border-radius: 4px; }
QCheckBox::indicator:checked { background-color: #0E639C; border: 1px solid #0E639C; }
QPlainTextEdit { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 4px; padding: 5px; font-family: 'Consolas', monospace; font-size: 12px; color: #D4D4D4; }
QStatusBar { background-color: #252526; border-top: 1px solid #333333; color: #969696; }
"""

PASTEL_STYLE = """
* {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
    font-size: 13px;
    color: #5A5A5A;
}
QMainWindow, QWidget { background-color: #FDF6F5; }
#headerPanel { background-color: #F4EAE6; border-bottom: 1px solid #E1D5D8; }
#appTitle { font-size: 18px; font-weight: bold; color: #8CA6C6; padding: 10px; }
#btnPrimary { background-color: #A8C5E6; border: none; border-radius: 4px; color: #FFFFFF; font-weight: bold; padding: 6px 16px; }
#btnPrimary:hover { background-color: #92B4D6; }
#btnPrimary:pressed { background-color: #7D9DC0; }
#btnPrimary:checked { background-color: #B5D5C5; }
#btnSecondary { background-color: #E1D5D8; border: none; border-radius: 4px; color: #5A5A5A; font-weight: bold; padding: 6px 16px; }
#btnSecondary:hover { background-color: #D2C4C7; }
#btnDanger { background-color: #F4A6A6; border: none; border-radius: 4px; color: #FFFFFF; font-weight: bold; padding: 6px 16px; }
#btnDanger:hover { background-color: #E49595; }
QGroupBox { background-color: #F4EAE6; border: 1px solid #E1D5D8; border-radius: 8px; margin-top: 15px; padding-top: 15px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 5px; color: #8CA6C6; font-weight: bold; font-size: 12px; }
QSlider::groove:horizontal { height: 6px; background: #E1D5D8; border-radius: 3px; }
QSlider::handle:horizontal { background: #A8C5E6; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #92B4D6; }
QSlider::sub-page:horizontal { background: #A8C5E6; border-radius: 3px; }
QTabWidget::pane { border: 1px solid #E1D5D8; background-color: #F4EAE6; border-radius: 4px; }
QTabBar::tab { background-color: #E1D5D8; color: #7A7A7A; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; border-right: 1px solid #D2C4C7; }
QTabBar::tab:selected { background-color: #F4EAE6; color: #8CA6C6; font-weight: bold; border-bottom: 2px solid #A8C5E6; }
QTabBar::tab:hover:!selected { background-color: #D2C4C7; }
QComboBox, QSpinBox, QLineEdit { background-color: #FDF6F5; border: 1px solid #E1D5D8; border-radius: 4px; padding: 4px 8px; color: #5A5A5A; }
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border: 1px solid #A8C5E6; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; background-color: #FDF6F5; border: 1px solid #E1D5D8; border-radius: 4px; }
QCheckBox::indicator:checked { background-color: #A8C5E6; border: 1px solid #A8C5E6; }
QPlainTextEdit { background-color: #E1D5D8; border: 1px solid #D2C4C7; border-radius: 4px; padding: 5px; font-family: 'Consolas', monospace; font-size: 12px; color: #4A4A4A; }
QStatusBar { background-color: #F4EAE6; border-top: 1px solid #E1D5D8; color: #7A7A7A; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RoboArmPro()
    win.show()
    sys.exit(app.exec())
