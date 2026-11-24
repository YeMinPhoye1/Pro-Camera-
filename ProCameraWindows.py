import os
# Must be set BEFORE importing cv2 so Windows 11 MSMF opens fast without software transform lag
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import re
import math
import time
import wave
import queue
import struct
import shutil
import tempfile
import threading
import subprocess
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QCheckBox, QFrame,
    QDialog, QListWidget, QListWidgetItem, QMessageBox, QSizePolicy,
    QButtonGroup, QTabWidget, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF, QRect
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QIcon,
    QPainter, QPainterPath, QLinearGradient, QColor, QPen
)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    sd = None
    HAS_AUDIO = False

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = shutil.which("ffmpeg")


# ==========================================
# Storage Directories & Resource Helper
# ==========================================
BASE_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "ProCam_Windows")
DIRS = {
    "photos": os.path.join(BASE_DIR, "Photos"),
    "live": os.path.join(BASE_DIR, "LivePhotos"),
    "videos": os.path.join(BASE_DIR, "Videos"),
    "timelapse": os.path.join(BASE_DIR, "TimeLapse"),
    "luts": os.path.join(BASE_DIR, "LUTs"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


def resource_path(filename):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)


def upscale_hq_still(frame, target_w, target_h):
    """High-quality Lanczos4 + micro-detail upscale for 2K/4K still photos."""
    h, w = frame.shape[:2]
    if w >= target_w and h >= target_h:
        return frame
    up = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.2)
    return cv2.addWeighted(up, 1.20, blur, -0.20, 0)


# ==========================================
# Smart Zero-Latency Camera Thread (Auto-Negotiates Max Smooth FPS)
# ==========================================
class CameraReader(threading.Thread):
    def __init__(self, index, width, height, target_fps=60, backend_name="MSMF"):
        super().__init__(daemon=True)
        self.index = index
        self.req_w = width
        self.req_h = height
        self.req_fps = int(target_fps)
        self.backend_name = backend_name

        self._lock = threading.Lock()
        self._running = True
        self._paused = False
        self._frame = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._pending_props = {}
        self._cam_times = deque(maxlen=45)

        self.cam_fps = 0.0
        self.ready = False
        self.error_msg = ""
        self.actual_w = width
        self.actual_h = height
        self.active_backend = backend_name
        self._init_done = threading.Event()

    def _try_open(self, api_flag, w, h, fps, use_mjpg=True):
        cap = cv2.VideoCapture(self.index, api_flag)
        if not cap.isOpened():
            return None, None
        if use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frm = cap.read()
        if not ret or frm is None:
            cap.release()
            return None, None
        return cap, frm

    def set_prop(self, prop_id, value):
        with self._lock:
            self._pending_props[prop_id] = value

    def set_paused(self, paused):
        with self._lock:
            self._paused = paused

    def set_fast_shutter_lock(self, enabled):
        exp_val = -6 if self.req_fps >= 60 else -5
        if self.active_backend == "MSMF":
            if enabled:
                self.set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                self.set_prop(cv2.CAP_PROP_EXPOSURE, exp_val)
            else:
                self.set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            if enabled:
                self.set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.set_prop(cv2.CAP_PROP_EXPOSURE, exp_val)
            else:
                self.set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def get_latest(self):
        with self._lock:
            return self._frame_id, self._frame_ts, self._frame, self.cam_fps

    def stop(self):
        self._running = False

    def run(self):
        # Prevent asking a 1080p laptop cam for unsupported 4K/60 that drops it to slow YUY2
        primary_api = cv2.CAP_MSMF if self.backend_name == "MSMF" else cv2.CAP_DSHOW
        alt_api = cv2.CAP_DSHOW if self.backend_name == "MSMF" else cv2.CAP_MSMF
        alt_name = "DShow" if self.backend_name == "MSMF" else "MSMF"

        # Smart negotiation order:
        # 1. Exact requested resolution & FPS (MJPG)
        # 2. If 2K/4K failed or sensor returned smaller size -> reopen clean 1080p/720p MJPG @ max FPS
        # 3. Fallback to 30fps MJPG if 60fps caused low-fps driver fallback
        cap, first_frame = self._try_open(primary_api, self.req_w, self.req_h, self.req_fps, use_mjpg=True)
        chosen_backend = self.backend_name

        if cap is not None and first_frame is not None:
            fh, fw = first_frame.shape[:2]
            reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0
            # If user asked for 2K/4K on a non-4K webcam, or 60fps caused driver to report <25fps:
            need_reopen = False
            best_w, best_h = fw, fh
            best_fps = self.req_fps
            if self.req_w > 1920 and fw < self.req_w:
                # Sensor isn't native 2K/4K -> reopen cleanly at its best 1080p/720p MJPG
                best_w, best_h = min(fw, 1920), min(fh, 1080)
                need_reopen = True
            if self.req_fps >= 60 and 0 < reported_fps < 25:
                best_fps = 30
                need_reopen = True

            if need_reopen:
                cap.release()
                cap, first_frame = self._try_open(primary_api, best_w, best_h, best_fps, use_mjpg=True)

        if cap is None:
            for bname, api, w, h, fps, mjpg in [
                (self.backend_name, primary_api, min(self.req_w, 1920), min(self.req_h, 1080), 30, True),
                (self.backend_name, primary_api, 1280, 720, 30, True),
                (self.backend_name, primary_api, 1280, 720, 30, False),
                (alt_name, alt_api, min(self.req_w, 1920), min(self.req_h, 1080), 30, True),
                (alt_name, alt_api, 1280, 720, 30, False),
            ]:
                cap, first_frame = self._try_open(api, w, h, fps, use_mjpg=mjpg)
                if cap is not None:
                    chosen_backend = bname
                    break

        if cap is None or first_frame is None:
            self.error_msg = f"Camera {self.index} not detected."
            self._init_done.set()
            return

        self.active_backend = chosen_backend
        self.actual_h, self.actual_w = first_frame.shape[:2]
        now = time.perf_counter()
        self._cam_times.append(now)
        with self._lock:
            self._frame = first_frame
            self._frame_id = 1
            self._frame_ts = now
        self.ready = True
        self._init_done.set()

        try:
            while self._running:
                if self._paused:
                    # Drain capture lightly so camera stays warm without eating CPU
                    cap.grab()
                    time.sleep(0.05)
                    continue

                props = None
                with self._lock:
                    if self._pending_props:
                        props = self._pending_props.copy()
                        self._pending_props.clear()
                if props:
                    for pid, val in props.items():
                        try:
                            cap.set(pid, val)
                        except Exception:
                            pass

                ret, frame = cap.read()
                if ret and frame is not None:
                    now = time.perf_counter()
                    self._cam_times.append(now)
                    fps_val = self.cam_fps
                    if len(self._cam_times) >= 2:
                        span = self._cam_times[-1] - self._cam_times[0]
                        if span > 0.05:
                            fps_val = (len(self._cam_times) - 1) / span
                    with self._lock:
                        self._frame = frame
                        self._frame_id += 1
                        self._frame_ts = now
                        self.cam_fps = fps_val
                else:
                    time.sleep(0.002)
        finally:
            cap.release()


# ==========================================
# Zero-Drop Async H.264 Recorder (Native or Fast 2K/4K FFmpeg Scale)
# ==========================================
class AsyncVideoRecorder:
    def __init__(self, in_w, in_h, out_w, out_h, target_fps=30.0):
        self.in_w = in_w & ~1
        self.in_h = in_h & ~1
        self.out_w = out_w & ~1
        self.out_h = out_h & ~1
        self.target_fps = float(target_fps)
        self.use_ffmpeg = bool(FFMPEG_EXE)

        self.temp_path = tempfile.mktemp(suffix=".h264" if self.use_ffmpeg else "_raw.mp4")
        self._queue = queue.Queue(maxsize=120)
        self._stop_event = threading.Event()
        self.frames_written = 0
        self.start_time = time.perf_counter()
        self.end_time = self.start_time

        self._proc = None
        self._cv_writer = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def push_frame(self, bgr_frame):
        if self._stop_event.is_set():
            return False
        try:
            self._queue.put_nowait(bgr_frame)
            return True
        except queue.Full:
            return False

    def _worker(self):
        if self.use_ffmpeg:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            cmd = [
                FFMPEG_EXE, "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{self.in_w}x{self.in_h}",
                "-r", f"{self.target_fps:.2f}",
                "-i", "pipe:0",
                "-an",
            ]
            if self.out_w != self.in_w or self.out_h != self.in_h:
                cmd += ["-vf", f"scale={self.out_w}:{self.out_h}:flags=bicubic"]

            cmd += [
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-crf", "20", "-pix_fmt", "yuv420p",
                "-f", "h264", self.temp_path
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags
                )
            except Exception:
                self.use_ffmpeg = False
                self._proc = None

        if not self.use_ffmpeg:
            self.temp_path = tempfile.mktemp(suffix="_raw.mp4")
            self._cv_writer = cv2.VideoWriter(
                self.temp_path, cv2.VideoWriter_fourcc(*"mp4v"), self.target_fps, (self.out_w, self.out_h)
            )

        while True:
            item = self._queue.get()
            if item is None:
                break
            frm = item
            if self._proc is not None and self._proc.stdin is not None:
                if frm.shape[1] != self.in_w or frm.shape[0] != self.in_h:
                    frm = cv2.resize(frm, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                try:
                    self._proc.stdin.write(frm.tobytes())
                    self.frames_written += 1
                except Exception:
                    break
            elif self._cv_writer is not None:
                if frm.shape[1] != self.out_w or frm.shape[0] != self.out_h:
                    frm = cv2.resize(frm, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
                self._cv_writer.write(frm)
                self.frames_written += 1

        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=12)
            except Exception:
                pass
        if self._cv_writer is not None:
            self._cv_writer.release()

    def stop_and_get_info(self):
        self.end_time = time.perf_counter()
        self._stop_event.set()
        try:
            self._queue.put(None, timeout=2.0)
        except Exception:
            pass
        self._thread.join(timeout=15.0)
        elapsed = max(0.2, self.end_time - self.start_time)
        return self.temp_path, self.use_ffmpeg, self.frames_written, elapsed


def finalize_stream_to_mp4(temp_path, is_raw_h264, temp_wav, final_mp4, output_fps):
    if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 64:
        return False

    if not FFMPEG_EXE:
        shutil.move(temp_path, final_mp4)
        return True

    has_audio = temp_wav is not None and os.path.exists(temp_wav)
    safe_fps = max(5.0, min(60.0, float(output_fps)))

    if is_raw_h264:
        cmd = [FFMPEG_EXE, "-y", "-fflags", "+genpts", "-r", f"{safe_fps:.3f}", "-i", temp_path]
        if has_audio:
            cmd += ["-i", temp_wav]
        cmd += ["-c:v", "copy"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += ["-movflags", "+faststart", final_mp4]
    else:
        cmd = [FFMPEG_EXE, "-y", "-i", temp_path]
        if has_audio:
            cmd += ["-i", temp_wav]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "20"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += ["-movflags", "+faststart", final_mp4]

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
        ok = (res.returncode == 0 and os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 0)
    except Exception:
        ok = False

    for p in (temp_path, temp_wav):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    return ok


def encode_smooth_cfr_live_mp4(frames_with_ts, temp_wav, out_mp4, target_fps=60.0):
    """
    Encodes Live Photo onto an exact constant-FPS timeline (30 or 60 FPS) with
    smooth sub-frame motion interpolation so even a 20-30fps webcam produces
    silky smooth 60fps Live Photos synced 1:1 with audio.
    """
    if len(frames_with_ts) < 2:
        return False

    first_ts = frames_with_ts[0][0]
    duration = max(0.4, frames_with_ts[-1][0] - first_ts)
    out_fps = 60.0 if target_fps >= 55 else 30.0
    total_out = max(2, int(round(duration * out_fps)))

    in_h, in_w = frames_with_ts[0][1].shape[:2]
    # Keep Live Photo motion video crisp 720p/1080p max so phones & PC play it smoothly
    if in_w > 1920:
        scale = 1920.0 / in_w
        in_w = 1920
        in_h = int(in_h * scale)
    in_w &= ~1
    in_h &= ~1

    ts_arr = [t - first_ts for (t, _) in frames_with_ts]
    n_in = len(ts_arr)

    def _get_resized(idx):
        f = frames_with_ts[idx][1]
        if f.shape[1] != in_w or f.shape[0] != in_h:
            return cv2.resize(f, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        return f

    def _iter_smooth_frames():
        ptr = 0
        cached_idx = -1
        cached_f0 = None
        cached_f1 = None
        for k in range(total_out):
            t_target = k / out_fps
            while ptr + 1 < n_in - 1 and ts_arr[ptr + 1] < t_target:
                ptr += 1

            t0 = ts_arr[ptr]
            t1 = ts_arr[min(ptr + 1, n_in - 1)]
            dt = t1 - t0
            alpha = 0.0 if dt <= 1e-4 else max(0.0, min(1.0, (t_target - t0) / dt))

            if ptr != cached_idx:
                cached_f0 = _get_resized(ptr)
                cached_f1 = _get_resized(min(ptr + 1, n_in - 1))
                cached_idx = ptr

            if alpha < 0.16:
                yield cached_f0
            elif alpha > 0.84:
                yield cached_f1
            else:
                # Smooth temporal blend between adjacent frames for fluid 60fps playback
                yield cv2.addWeighted(cached_f0, 1.0 - alpha, cached_f1, alpha, 0)

    if FFMPEG_EXE:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        cmd = [
            FFMPEG_EXE, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{in_w}x{in_h}",
            "-r", f"{out_fps:.2f}",
            "-i", "pipe:0",
        ]
        has_audio = temp_wav is not None and os.path.exists(temp_wav)
        if has_audio:
            cmd += ["-i", temp_wav]
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += ["-movflags", "+faststart", out_mp4]

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
            for frm in _iter_smooth_frames():
                proc.stdin.write(frm.tobytes())
            proc.stdin.close()
            proc.wait(timeout=20)
            ok = (proc.returncode == 0 and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0)
        except Exception:
            ok = False

        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass
        return ok
    else:
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (in_w, in_h))
        for frm in _iter_smooth_frames():
            writer.write(frm)
        writer.release()
        return True


# ==========================================
# Ultra-Fast Zero-Copy Preview Widget
# ==========================================
class FastVideoView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._rgb_buf = None
        self._qimg = None
        self._message = "Starting Windows Camera..."

    def set_message(self, msg):
        self._qimg = None
        self._rgb_buf = None
        self._message = msg
        self.update()

    def set_rgb_buffer(self, rgb_img):
        h, w, ch = rgb_img.shape
        self._rgb_buf = rgb_img
        self._qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#08080b"))

        if self._qimg is not None:
            iw, ih = self._qimg.width(), self._qimg.height()
            vw, vh = rect.width(), rect.height()
            scale = min(vw / max(1, iw), vh / max(1, ih))
            dw, dh = int(iw * scale), int(ih * scale)
            dx = (vw - dw) // 2
            dy = (vh - dh) // 2
            painter.drawImage(QRect(dx, dy, dw, dh), self._qimg)
        else:
            painter.setPen(QColor("#a0a2b8"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._message)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#282834"), 1))
        painter.drawRect(0, 0, rect.width() - 1, rect.height() - 1)


# ==========================================
# Android Motion Photo (MVIMG) Engine
# ==========================================
def build_exif_app1_xiaomi(dt_str=None):
    if dt_str is None:
        dt_str = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    make_b = b"ProCam\x00"
    model_b = b"ProCam Windows\x00"
    dt_b = dt_str.encode("ascii") + b"\x00"

    make_off, model_off, dt_off, exif_ifd_off = 62, 70, 86, 106
    header = b"II" + struct.pack("<HI", 42, 8)
    ifd0 = struct.pack("<H", 4)
    ifd0 += struct.pack("<HHII", 0x010F, 2, len(make_b), make_off)
    ifd0 += struct.pack("<HHII", 0x0110, 2, len(model_b), model_off)
    ifd0 += struct.pack("<HHII", 0x0132, 2, len(dt_b), dt_off)
    ifd0 += struct.pack("<HHII", 0x8769, 4, 1, exif_ifd_off)
    ifd0 += struct.pack("<I", 0)

    data_block = make_b + b"\x00" + model_b + b"\x00" + dt_b
    exif_ifd = struct.pack("<H", 3)
    exif_ifd += struct.pack("<HHI", 0x8897, 1, 1) + struct.pack("<B3x", 1)
    exif_ifd += struct.pack("<HHII", 0x9003, 2, len(dt_b), dt_off)
    exif_ifd += struct.pack("<HHII", 0x9004, 2, len(dt_b), dt_off)
    exif_ifd += struct.pack("<I", 0)

    payload = b"Exif\x00\x00" + header + ifd0 + data_block + exif_ifd
    return b"\xFF\xE1" + struct.pack(">H", len(payload) + 2) + payload


def build_motion_photo_xmp_app1(mp4_size, timestamp_us):
    xmp_xml = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.1.0-jc003">\n'
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '    <rdf:Description rdf:about=""\n'
        '      xmlns:GCamera="http://ns.google.com/photos/1.0/camera/"\n'
        '      xmlns:Container="http://ns.google.com/photos/1.0/container/"\n'
        '      xmlns:Item="http://ns.google.com/photos/1.0/container/item/"\n'
        '      GCamera:MotionPhoto="1"\n'
        '      GCamera:MotionPhotoVersion="1"\n'
        f'      GCamera:MotionPhotoPresentationTimestampUs="{timestamp_us}"\n'
        '      GCamera:MicroVideo="1"\n'
        '      GCamera:MicroVideoVersion="1"\n'
        f'      GCamera:MicroVideoOffset="{mp4_size}"\n'
        f'      GCamera:MicroVideoPresentationTimestampUs="{timestamp_us}">\n'
        '      <Container:Directory>\n'
        '        <rdf:Seq>\n'
        '          <rdf:li rdf:parseType="Resource">\n'
        '            <Container:Item\n'
        '              Item:Mime="image/jpeg"\n'
        '              Item:Semantic="Primary"\n'
        '              Item:Length="0"\n'
        '              Item:Padding="0"/>\n'
        '          </rdf:li>\n'
        '          <rdf:li rdf:parseType="Resource">\n'
        '            <Container:Item\n'
        '              Item:Mime="video/mp4"\n'
        '              Item:Semantic="MotionPhoto"\n'
        f'              Item:Length="{mp4_size}"\n'
        '              Item:Padding="0"/>\n'
        '          </rdf:li>\n'
        '        </rdf:Seq>\n'
        '      </Container:Directory>\n'
        '    </rdf:Description>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp_xml.encode("utf-8")
    return b"\xFF\xE1" + struct.pack(">H", len(payload) + 2) + payload


def pack_android_motion_photo(still_bgr, mp4_path, output_mvimg_path, keyframe_us=1500000):
    ok, enc = cv2.imencode(".jpg", still_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise RuntimeError("Failed to encode still JPEG")
    jpg_bytes = enc.tobytes()

    with open(mp4_path, "rb") as vf:
        mp4_bytes = vf.read()
    mp4_size = len(mp4_bytes)
    if mp4_size < 32:
        raise RuntimeError("Encoded MP4 is empty")

    exif_app1 = build_exif_app1_xiaomi()
    xmp_app1 = build_motion_photo_xmp_app1(mp4_size, int(max(0, keyframe_us)))

    insert_pos = 2
    if len(jpg_bytes) > 6 and jpg_bytes[2:4] == b"\xFF\xE0":
        app0_len = struct.unpack(">H", jpg_bytes[4:6])[0]
        insert_pos = 4 + app0_len

    final_jpg = jpg_bytes[:insert_pos] + exif_app1 + xmp_app1 + jpg_bytes[insert_pos:]

    tmp_out = output_mvimg_path + ".tmp"
    with open(tmp_out, "wb") as f:
        f.write(final_jpg)
        f.write(mp4_bytes)
    os.replace(tmp_out, output_mvimg_path)
    return output_mvimg_path


def find_embedded_mp4_offset(jpg_path):
    try:
        file_size = os.path.getsize(jpg_path)
        if file_size < 1024:
            return None
        with open(jpg_path, "rb") as f:
            head = f.read(min(file_size, 65536))
            offset_val = None
            m1 = re.search(rb'MicroVideoOffset="(\d+)"', head)
            if m1:
                offset_val = int(m1.group(1))
            else:
                m2 = re.search(rb'Item:Semantic="MotionPhoto"[^>]*Item:Length="(\d+)"', head)
                if not m2:
                    m2 = re.search(rb'Item:Length="(\d+)"[^>]*Item:Semantic="MotionPhoto"', head)
                if m2:
                    offset_val = int(m2.group(1))

            if offset_val and 32 < offset_val < file_size:
                start = file_size - offset_val
                f.seek(start + 4)
                if f.read(4) == b"ftyp":
                    return start

            f.seek(0)
            data = f.read()
            eoi = data.find(b"\xFF\xD9")
            if eoi != -1:
                ftyp_pos = data.find(b"ftyp", eoi + 2)
                if ftyp_pos >= 4:
                    return ftyp_pos - 4
    except Exception:
        pass
    return None


def extract_embedded_mp4(jpg_path, dest_mp4_path):
    start = find_embedded_mp4_offset(jpg_path)
    if start is None:
        return None
    with open(jpg_path, "rb") as src, open(dest_mp4_path, "wb") as dst:
        src.seek(start)
        shutil.copyfileobj(src, dst)
    return dest_mp4_path


# ==========================================
# Custom 3D/1D .cube LUT Engine
# ==========================================
class CubeLUT:
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.is_1d = False
        self.lut_1d = None
        self.baked_3d = None
        self._load(filepath)

    def _load(self, filepath):
        size_3d = None
        size_1d = None
        dmin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        dmax = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        data = []

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                upper = line.upper()
                if upper.startswith("TITLE"):
                    continue
                elif upper.startswith("LUT_3D_SIZE"):
                    size_3d = int(line.split()[1])
                elif upper.startswith("LUT_1D_SIZE"):
                    size_1d = int(line.split()[1])
                elif upper.startswith("DOMAIN_MIN"):
                    dmin = np.array([float(x) for x in line.split()[1:4]], dtype=np.float32)
                elif upper.startswith("DOMAIN_MAX"):
                    dmax = np.array([float(x) for x in line.split()[1:4]], dtype=np.float32)
                else:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            data.append((float(parts[0]), float(parts[1]), float(parts[2])))
                        except ValueError:
                            pass

        arr = np.array(data, dtype=np.float32)

        if size_3d is not None:
            expected = size_3d ** 3
            if len(arr) < expected:
                raise ValueError(f"Invalid 3D LUT: expected {expected} entries, found {len(arr)}.")
            table = arr[:expected].reshape((size_3d, size_3d, size_3d, 3))
            self.baked_3d = self._bake_128_table(table, size_3d, dmin, dmax)
            self.is_1d = False
        elif size_1d is not None:
            if len(arr) < size_1d:
                raise ValueError(f"Invalid 1D LUT: expected {size_1d} entries, found {len(arr)}.")
            arr = arr[:size_1d]
            lin = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            out_1d = np.empty((1, 256, 3), dtype=np.uint8)
            for ch in range(3):
                norm = np.clip((lin - dmin[ch]) / max(1e-6, dmax[ch] - dmin[ch]), 0.0, 1.0)
                vals = np.interp(norm * (size_1d - 1), np.arange(size_1d), arr[:, ch])
                out_1d[0, :, 2 - ch] = np.clip(vals * 255.0 + 0.5, 0, 255).astype(np.uint8)
            self.lut_1d = out_1d
            self.is_1d = True
        else:
            raise ValueError("Unsupported .cube file (missing LUT_3D_SIZE or LUT_1D_SIZE).")

    def _bake_128_table(self, table, n, dmin, dmax):
        grid = 128
        lin = np.linspace(0.0, 1.0, grid, dtype=np.float32)
        r_n = np.clip((lin - dmin[0]) / max(1e-6, dmax[0] - dmin[0]), 0.0, 1.0) * (n - 1)
        g_n = np.clip((lin - dmin[1]) / max(1e-6, dmax[1] - dmin[1]), 0.0, 1.0) * (n - 1)
        b_n = np.clip((lin - dmin[2]) / max(1e-6, dmax[2] - dmin[2]), 0.0, 1.0) * (n - 1)

        g0 = np.floor(g_n).astype(np.int32)
        g1 = np.minimum(g0 + 1, n - 1)
        dg = (g_n - g0)[:, None, None]

        r0 = np.floor(r_n).astype(np.int32)
        r1 = np.minimum(r0 + 1, n - 1)
        dr = (r_n - r0)[None, :, None]

        baked = np.empty((grid, grid, grid, 3), dtype=np.uint8)
        for bi in range(grid):
            bv = b_n[bi]
            b0 = int(math.floor(bv))
            b1 = min(b0 + 1, n - 1)
            db = bv - b0

            t0, t1 = table[b0], table[b1]
            c00 = t0[g0[:, None], r0[None, :]] * (1.0 - dr) + t0[g0[:, None], r1[None, :]] * dr
            c01 = t0[g1[:, None], r0[None, :]] * (1.0 - dr) + t0[g1[:, None], r1[None, :]] * dr
            c10 = t1[g0[:, None], r0[None, :]] * (1.0 - dr) + t1[g0[:, None], r1[None, :]] * dr
            c11 = t1[g1[:, None], r0[None, :]] * (1.0 - dr) + t1[g1[:, None], r1[None, :]] * dr

            c0 = c00 * (1.0 - dg) + c01 * dg
            c1 = c10 * (1.0 - dg) + c11 * dg
            c = c0 * (1.0 - db) + c1 * db
            baked[bi] = np.clip(c[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return baked

    def apply(self, bgr_img, strength=1.0):
        if strength <= 0.01:
            return bgr_img
        if self.is_1d:
            out = cv2.LUT(bgr_img, self.lut_1d)
        else:
            out = self.baked_3d[bgr_img[:, :, 0] >> 1, bgr_img[:, :, 1] >> 1, bgr_img[:, :, 2] >> 1]
        if strength < 0.99:
            return cv2.addWeighted(out, strength, bgr_img, 1.0 - strength, 0)
        return out


# ==========================================
# Continuous Audio Ring Buffer & Recorder
# ==========================================
class AudioEngine:
    def __init__(self, samplerate=44100, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.enabled = False
        self.stream = None
        self.lock = threading.Lock()
        self.ring_buffer = deque(maxlen=150)

        self.live_capturing = False
        self.live_pre_chunks = []
        self.live_post_chunks = []

        self.video_recording = False
        self.video_chunks = []

    def start(self):
        if not HAS_AUDIO:
            return False, "sounddevice not installed"
        if self.stream is not None:
            return True, "Mic active"
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="int16",
                blocksize=1024,
                callback=self._callback
            )
            self.stream.start()
            self.enabled = True
            return True, "Mic active"
        except Exception as e:
            self.stream = None
            self.enabled = False
            return False, f"Mic error: {e}"

    def stop(self):
        self.enabled = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _callback(self, indata, frames, time_info, status):
        now = time.perf_counter()
        chunk = indata.copy()
        with self.lock:
            self.ring_buffer.append((now, chunk))
            if self.live_capturing:
                self.live_post_chunks.append(chunk)
            if self.video_recording:
                self.video_chunks.append(chunk)

    def begin_live_capture(self, pre_seconds=1.5):
        if not self.enabled:
            return
        now = time.perf_counter()
        with self.lock:
            self.live_pre_chunks = [c for (t, c) in self.ring_buffer if t >= now - pre_seconds]
            self.live_post_chunks = []
            self.live_capturing = True

    def finish_live_capture(self):
        with self.lock:
            self.live_capturing = False
            all_chunks = self.live_pre_chunks + self.live_post_chunks
            self.live_pre_chunks = []
            self.live_post_chunks = []
        return np.concatenate(all_chunks, axis=0) if all_chunks else None

    def start_video_recording(self):
        if not self.enabled:
            return
        with self.lock:
            self.video_chunks = []
            self.video_recording = True

    def stop_video_recording(self):
        with self.lock:
            self.video_recording = False
            chunks = self.video_chunks
            self.video_chunks = []
        return np.concatenate(chunks, axis=0) if chunks else None

    def save_wav(self, audio_np, wav_path):
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_np.tobytes())


# ==========================================
# High-Speed Cached Color, Denoise & HDR Engine
# ==========================================
HDR_CLAHE = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
BW_CLAHE = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
HDR_GAMMA_LUT = np.clip((np.linspace(0.0, 1.0, 256, dtype=np.float32) ** 0.80) * 255.0 + 0.5, 0, 255).astype(np.uint8)
SEPIA_KERNEL = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]], dtype=np.float32)


def kelvin_to_bgr_gains(kelvin):
    temp = max(2500.0, min(8500.0, float(kelvin))) / 100.0

    def _calc(t):
        if t <= 66:
            r = 255.0
            g = 99.4708025861 * math.log(t) - 161.1195681661
            b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
        else:
            r = 329.698727446 * ((t - 60) ** -0.1332047592)
            g = 288.1221695283 * ((t - 60) ** -0.0755148492)
            b = 255.0
        return np.array([np.clip(b, 1.0, 255.0), np.clip(g, 1.0, 255.0), np.clip(r, 1.0, 255.0)], dtype=np.float32)

    gains = _calc(temp) / _calc(55.0)
    return gains / gains[1]


class FastToneLUTCache:
    def __init__(self):
        self._last_key = None
        self._lut = None

    def get_lut(self, ev, iso, wb_kelvin, brightness, contrast, flt):
        key = (round(ev, 2), iso, wb_kelvin, brightness, round(contrast, 2), flt)
        if key == self._last_key:
            return self._lut
        self._last_key = key

        is_neutral = (
            abs(ev) < 0.01 and iso == 400 and wb_kelvin is None
            and brightness == 0 and abs(contrast - 1.0) < 0.01
            and flt not in ("Warm Film", "Cool Cinema")
        )
        if is_neutral:
            self._lut = None
            return None

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        total_gain = (2.0 ** (ev * 0.55)) * ((iso / 400.0) ** 0.50)
        y = x * total_gain
        if total_gain > 1.0:
            y = y / (1.0 + 0.38 * np.maximum(0.0, y - 0.65))

        bgr = np.stack([y, y, y], axis=1) * 255.0
        if wb_kelvin is not None:
            bgr *= kelvin_to_bgr_gains(wb_kelvin)[None, :]
        bgr = bgr * contrast + float(brightness)

        if flt == "Warm Film":
            bgr[:, 0] -= 14.0
            bgr[:, 1] += 6.0
            bgr[:, 2] += 18.0
            bgr = bgr * 0.94 + 12.0
        elif flt == "Cool Cinema":
            bgr[:, 0] += 18.0
            bgr[:, 1] += 5.0
            bgr[:, 2] -= 8.0
            bgr = bgr * 1.12 - 6.0

        self._lut = np.clip(bgr + 0.5, 0, 255).astype(np.uint8).reshape((1, 256, 3))
        return self._lut


TONE_CACHE = FastToneLUTCache()


def apply_fast_clean_denoise(img, prev_frame, strength=0.85):
    out = img
    h, w = out.shape[:2]

    if prev_frame is not None and prev_frame.shape == out.shape:
        sw, sh = max(80, w // 4), max(45, h // 4)
        s_cur = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_NEAREST)
        s_prv = cv2.resize(prev_frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
        gray_diff = cv2.cvtColor(cv2.absdiff(s_cur, s_prv), cv2.COLOR_BGR2GRAY)
        motion = np.clip(gray_diff.astype(np.float32) / (16.0 + 12.0 * strength), 0.0, 1.0)
        motion = cv2.GaussianBlur(motion, (5, 5), 0)
        min_a = max(0.28, 0.58 - 0.28 * strength)
        alpha = cv2.resize(min_a + (1.0 - min_a) * motion, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
        out = np.clip(
            out.astype(np.float32) * alpha + prev_frame.astype(np.float32) * (1.0 - alpha),
            0, 255
        ).astype(np.uint8)

    ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    hw, hh = max(160, w // 2), max(90, h // 2)
    cr = cv2.resize(cv2.GaussianBlur(cv2.resize(cr, (hw, hh)), (5, 5), 0), (w, h), interpolation=cv2.INTER_LINEAR)
    cb = cv2.resize(cv2.GaussianBlur(cv2.resize(cb, (hw, hh)), (5, 5), 0), (w, h), interpolation=cv2.INTER_LINEAR)

    y_small = cv2.resize(y, (hw, hh), interpolation=cv2.INTER_AREA)
    sig = int(14 + 16 * strength)
    y_bilat = cv2.resize(cv2.bilateralFilter(y_small, 5, sig, sig), (w, h), interpolation=cv2.INTER_LINEAR)
    y_mix = cv2.addWeighted(y, 0.45, y_bilat, 0.55, 0)
    y_blur = cv2.GaussianBlur(y_mix, (0, 0), 1.2)
    y_clean = cv2.addWeighted(y_mix, 1.0 + 0.25 * strength, y_blur, -0.25 * strength, 0)

    return cv2.cvtColor(cv2.merge((y_clean, cr, cb)), cv2.COLOR_YCrCb2BGR)


def apply_hdr_effect(img, strength=0.85):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_hdr = cv2.addWeighted(HDR_CLAHE.apply(l), 0.55, cv2.LUT(l, HDR_GAMMA_LUT), 0.45, 0)
    hdr_bgr = cv2.cvtColor(cv2.merge((l_hdr, a, b)), cv2.COLOR_LAB2BGR)
    if strength < 0.99:
        return cv2.addWeighted(hdr_bgr, strength, img, 1.0 - strength, 0)
    return hdr_bgr


def apply_image_pipeline(frame, settings, active_lut, prev_temporal=None):
    img = frame

    if settings["mirror"]:
        img = cv2.flip(img, 1)

    zoom = settings["zoom"]
    if zoom > 1.01:
        h, w = img.shape[:2]
        nw, nh = int(w / zoom), int(h / zoom)
        x1, y1 = (w - nw) // 2, (h - nh) // 2
        img = cv2.resize(img[y1:y1 + nh, x1:x1 + nw], (w, h), interpolation=cv2.INTER_LINEAR)

    temporal_next = None
    if settings["clean_enabled"] and settings["clean_strength"] > 0.05:
        img = apply_fast_clean_denoise(img, prev_temporal, settings["clean_strength"])
        temporal_next = img

    ev_eff = 0.0 if settings["auto_exposure"] else settings["ev_offset"]
    iso_eff = 400 if settings["auto_iso"] else settings["iso_val"]
    wb_eff = None if settings["auto_wb"] else settings["wb_kelvin"]
    flt = settings["filter"]

    tone_lut = TONE_CACHE.get_lut(ev_eff, iso_eff, wb_eff, settings["brightness"], settings["contrast"], flt)
    if tone_lut is not None:
        img = cv2.LUT(img, tone_lut)

    if settings["hdr_enabled"]:
        img = apply_hdr_effect(img, settings["hdr_strength"])

    sat = settings["saturation"] * (1.35 if flt == "Vivid" else 1.0)
    if abs(sat - 1.0) > 0.01:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_lut = np.clip(np.arange(256, dtype=np.float32) * sat + 0.5, 0, 255).astype(np.uint8)
        hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], s_lut)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if flt == "Leica B&W":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(cv2.convertScaleAbs(BW_CLAHE.apply(gray), alpha=1.18, beta=-12), cv2.COLOR_GRAY2BGR)
    elif flt == "Vintage Sepia":
        img = cv2.transform(img, SEPIA_KERNEL)

    sharp = settings["sharpness"]
    if sharp > 0.01:
        blur = cv2.GaussianBlur(img, (0, 0), 1.3)
        img = cv2.addWeighted(img, 1.0 + sharp, blur, -sharp, 0)

    if active_lut is not None and settings["lut_strength"] > 0.01:
        img = active_lut.apply(img, settings["lut_strength"])

    if settings["watermark"]:
        h, w = img.shape[:2]
        hdr_tag = "  |  HDR" if settings["hdr_enabled"] else ""
        lut_tag = f"  |  {active_lut.name}" if active_lut else ""
        text = f"PROCAM WIN BY YE  |  {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}{hdr_tag}{lut_tag}"
        scale = max(0.45, w / 1920.0 * 0.7)
        thick = max(1, int(scale * 2))
        pos = (24, h - 22)
        cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), thick, cv2.LINE_AA)

    return img, temporal_next


# ==========================================
# Throttled Histogram & Fast Preview Overlays
# ==========================================
class CachedHistogramOverlay:
    def __init__(self):
        self._counter = 0
        self._cached_pts = None

    def draw(self, vis):
        h, w = vis.shape[:2]
        hw, hh = 152, 66
        x0, y0 = w - hw - 12, 12
        if x0 < 0 or y0 + hh > h:
            return

        vis[y0:y0 + hh, x0:x0 + hw] = (vis[y0:y0 + hh, x0:x0 + hw] >> 1)
        cv2.rectangle(vis, (x0, y0), (x0 + hw, y0 + hh), (85, 85, 100), 1)

        self._counter += 1
        if self._cached_pts is None or (self._counter % 4 == 0):
            small = cv2.resize(vis, (128, 72), interpolation=cv2.INTER_NEAREST)
            bins = 48
            pts_list = []
            for ch in range(3):
                hist = cv2.calcHist([small], [ch], None, [bins], [0, 256]).flatten()
                m = hist.max()
                if m > 0:
                    hist = (hist / m) * (hh - 10)
                pts = np.zeros((bins, 1, 2), dtype=np.int32)
                for i in range(bins):
                    pts[i, 0, 0] = x0 + 4 + int(i * (hw - 8) / (bins - 1))
                    pts[i, 0, 1] = y0 + hh - 4 - int(hist[i])
                pts_list.append(pts)
            self._cached_pts = (w, pts_list)

        if self._cached_pts and self._cached_pts[0] == w:
            for pts, col in zip(self._cached_pts[1], [(255, 110, 70), (80, 235, 110), (75, 85, 255)]):
                cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)


HIST_OVERLAY = CachedHistogramOverlay()


def make_preview_rgb(clean_frame, target_w, target_h, settings, countdown_val, flash_alpha, live_capturing, rec_text, hud_text):
    fh, fw = clean_frame.shape[:2]
    tw = max(320, min(fw, target_w))
    th = max(180, min(fh, target_h))
    scale = min(tw / fw, th / fh)
    pw, ph = max(160, int(fw * scale)), max(90, int(fh * scale))

    if pw < fw or ph < fh:
        vis = cv2.resize(clean_frame, (pw, ph), interpolation=cv2.INTER_LINEAR)
    else:
        vis = clean_frame.copy()

    h, w = vis.shape[:2]

    if settings["focus_peaking"]:
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
        vis[lap > 36] = (60, 255, 90)

    if settings["zebra"]:
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        yy, xx = np.indices((h, w))
        vis[(gray >= 245) & (((xx + yy) // 5) % 2 == 0)] = (50, 60, 255)

    if settings["grid"]:
        for i in (1, 2):
            cv2.line(vis, (w * i // 3, 0), (w * i // 3, h), (205, 205, 205), 1)
            cv2.line(vis, (0, h * i // 3), (w, h * i // 3), (205, 205, 205), 1)

    if settings["histogram"]:
        HIST_OVERLAY.draw(vis)

    pill_x = 12
    if settings["hdr_enabled"] and not rec_text:
        cv2.rectangle(vis, (pill_x, 12), (pill_x + 54, 34), (0, 175, 255), -1)
        cv2.putText(vis, "HDR", (pill_x + 9, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (15, 15, 18), 2, cv2.LINE_AA)
        pill_x += 60
    if settings["clean_enabled"] and not rec_text:
        cv2.rectangle(vis, (pill_x, 12), (pill_x + 70, 34), (45, 200, 130), -1)
        cv2.putText(vis, "CLEAN", (pill_x + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 18), 2, cv2.LINE_AA)

    if live_capturing:
        cv2.rectangle(vis, (w // 2 - 70, 12), (w // 2 + 70, 42), (0, 184, 245), -1)
        cv2.putText(vis, "MVIMG ● REC", (w // 2 - 60, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (15, 15, 18), 2, cv2.LINE_AA)

    if rec_text:
        cv2.rectangle(vis, (12, 12), (285, 44), (20, 20, 26), -1)
        cv2.circle(vis, (30, 28), 6, (48, 59, 255), -1)
        cv2.putText(vis, rec_text, (44, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

    if hud_text:
        bar_h = 26
        vis[h - bar_h:h, 0:w] = (vis[h - bar_h:h, 0:w] >> 1)
        cv2.putText(vis, hud_text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 245), 1, cv2.LINE_AA)

    if countdown_val > 0:
        txt = str(countdown_val)
        (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)
        pos = ((w - tw2) // 2, (h + th2) // 2)
        cv2.putText(vis, txt, (pos[0] + 3, pos[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8, cv2.LINE_AA)

    if flash_alpha > 0.0:
        vis = cv2.addWeighted(np.full_like(vis, 255), flash_alpha, vis, 1.0 - flash_alpha, 0)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ==========================================
# RAM Pre-Cached Butter-Smooth Live Photo & Video Player (Yellow Bug Fixed!)
# ==========================================
class LivePhotoView(QWidget):
    """
    Pre-caches short Live Photos (~3s) into RAM for zero-stutter 60fps instant playback,
    and streams longer 2K/4K videos with wall-clock real-time sync.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.still_path = None
        self.video_path = None
        self.cap = None
        self.playing = False
        self.vid_fps = 30.0
        self.play_start_time = 0.0
        self.cur_frame_idx = -1

        self.audio_pcm = None
        self.audio_sr = 44100

        # RAM frame cache for Live Photos
        self._load_gen = 0
        self._cache_lock = threading.Lock()
        self.cached_frames = []   # list of (rgb_np, QImage)
        self.cached_fps = 30.0
        self.cache_ready = False

        self._rgb_buf = None
        self._qimg = None
        self._message = "No Media Selected"

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick_playback)

    def _bgr_to_scaled_qimg(self, bgr_img, max_w=860, max_h=540):
        fh, fw = bgr_img.shape[:2]
        if fw > max_w or fh > max_h:
            scale = min(max_w / fw, max_h / fh)
            nw, nh = max(64, int(fw * scale)), max(64, int(fh * scale))
            bgr_img = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return rgb, qimg

    def _set_bgr(self, bgr_img):
        vw = max(480, self.width())
        vh = max(300, self.height())
        rgb, qimg = self._bgr_to_scaled_qimg(bgr_img, vw, vh)
        self._rgb_buf = rgb
        self._qimg = qimg
        self.update()

    def load_media(self, path, playable_video=None, is_live_clip=False):
        self.stop_playback()
        self._load_gen += 1
        gen = self._load_gen

        self.still_path = path
        self.video_path = playable_video
        self.audio_pcm = None
        with self._cache_lock:
            self.cached_frames = []
            self.cache_ready = False

        self._show_still()

        target_vid = playable_video or (path if path and path.lower().endswith(".mp4") else None)
        if target_vid and os.path.exists(target_vid):
            if HAS_AUDIO and FFMPEG_EXE:
                threading.Thread(target=self._preload_audio, args=(target_vid, gen), daemon=True).start()
            if is_live_clip:
                threading.Thread(target=self._preload_live_frames, args=(target_vid, gen), daemon=True).start()

    def _preload_audio(self, vid_path, gen):
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            cmd = [FFMPEG_EXE, "-i", vid_path, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(self.audio_sr), "pipe:1"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=creationflags)
            if proc.returncode == 0 and proc.stdout and gen == self._load_gen:
                self.audio_pcm = np.frombuffer(proc.stdout, dtype=np.int16)
        except Exception:
            pass

    def _preload_live_frames(self, vid_path, gen):
        """Pre-decodes short Live Photo clip into RAM so Hold-to-Play runs at locked 60fps."""
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 5 or fps > 120:
            fps = 30.0
        frames = []
        max_frames = 240  # up to 4s @ 60fps
        while len(frames) < max_frames:
            if gen != self._load_gen:
                cap.release()
                return
            ret, frm = cap.read()
            if not ret or frm is None:
                break
            frames.append(self._bgr_to_scaled_qimg(frm, 820, 520))
        cap.release()

        if gen == self._load_gen and len(frames) >= 2:
            with self._cache_lock:
                self.cached_frames = frames
                self.cached_fps = float(fps)
                self.cache_ready = True

    def _show_still(self):
        if self.still_path and os.path.exists(self.still_path):
            if self.still_path.lower().endswith(".mp4"):
                cap = cv2.VideoCapture(self.still_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self._set_bgr(frame)
                else:
                    self._qimg = None
                    self._message = "Video Preview Unavailable"
                    self.update()
            else:
                img = cv2.imread(self.still_path)
                if img is not None:
                    self._set_bgr(img)
        else:
            self._qimg = None
            self._message = "No Media Selected"
            self.update()

    def _play_audio_track(self):
        if HAS_AUDIO and self.audio_pcm is not None and len(self.audio_pcm) > 0:
            try:
                sd.stop()
                sd.play(self.audio_pcm, self.audio_sr)
            except Exception:
                pass

    def start_playback(self):
        target_vid = self.video_path if self.video_path else (
            self.still_path if self.still_path and self.still_path.lower().endswith(".mp4") else None
        )
        if not target_vid or not os.path.exists(target_vid):
            return
        self.stop_playback()

        with self._cache_lock:
            use_cache = self.cache_ready and len(self.cached_frames) >= 2
            cfps = self.cached_fps

        if use_cache:
            self.vid_fps = cfps
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(target_vid)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps < 5 or fps > 120:
                fps = 30.0
            self.vid_fps = float(fps)

        self.playing = True
        self.cur_frame_idx = -1
        self.play_start_time = time.perf_counter()
        self._play_audio_track()
        self.play_timer.start(11)  # ~90Hz smooth sync tick

    def stop_playback(self):
        self.playing = False
        self.play_timer.stop()
        if HAS_AUDIO:
            try:
                sd.stop()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.still_path:
            self._show_still()

    def _restart_loop(self):
        self.cur_frame_idx = -1
        self.play_start_time = time.perf_counter()
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._play_audio_track()

    def _tick_playback(self):
        if not self.playing:
            return

        elapsed = time.perf_counter() - self.play_start_time
        target_idx = int(elapsed * self.vid_fps)

        # Path A: Zero-lag RAM-cached Live Photo playback
        if self.cap is None:
            with self._cache_lock:
                n = len(self.cached_frames)
                if n == 0:
                    return
                if target_idx >= n:
                    self._restart_loop()
                    target_idx = 0
                if target_idx == self.cur_frame_idx:
                    return
                self.cur_frame_idx = target_idx
                rgb, qimg = self.cached_frames[target_idx]
            self._rgb_buf = rgb
            self._qimg = qimg
            self.update()
            return

        # Path B: Real-time streaming playback for long videos
        if target_idx <= self.cur_frame_idx:
            return

        skip = target_idx - self.cur_frame_idx - 1
        for _ in range(skip):
            if not self.cap.grab():
                self._restart_loop()
                return
            self.cur_frame_idx += 1

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._restart_loop()
            return

        self.cur_frame_idx += 1
        self._set_bgr(frame)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#0d0d10"))

        if self._qimg is not None:
            iw, ih = self._qimg.width(), self._qimg.height()
            vw, vh = rect.width(), rect.height()
            scale = min(vw / max(1, iw), vh / max(1, ih))
            dw, dh = int(iw * scale), int(ih * scale)
            dx = (vw - dw) // 2
            dy = (vh - dh) // 2
            painter.drawImage(QRect(dx, dy, dw, dh), self._qimg)

            if self.playing:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#f5b800"))
                painter.drawRoundedRect(QRectF(dx + 14, dy + 14, 124, 28), 6, 6)
                painter.setPen(QColor("#111116"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                badge = f"LIVE {int(round(self.vid_fps))}fps" + (" 🔊" if self.audio_pcm is not None else "")
                painter.drawText(QRectF(dx + 14, dy + 14, 124, 28), Qt.AlignmentFlag.AlignCenter, badge)
        else:
            painter.setPen(QColor("#9a9cb2"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._message)

        # CRITICAL: Reset brush to NoBrush so outer border never fills the widget!
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#2a2a33"), 1))
        painter.drawRect(0, 0, rect.width() - 1, rect.height() - 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_playback()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.stop_playback()
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        if not self.playing:
            self._show_still()
        super().resizeEvent(event)


# ==========================================
# Gallery & MVIMG Viewer Dialog
# ==========================================
class GalleryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProCam Gallery — Instant 60fps MVIMG & 4K Viewer")
        self.resize(1020, 620)
        self.temp_dir = tempfile.mkdtemp(prefix="procam_mvimg_")
        self.extracted_cache = {}

        self.setStyleSheet("""
            QDialog { background-color: #141418; color: #f0f0f5; }
            QListWidget {
                background-color: #1c1c23; border: 1px solid #2e2e38;
                border-radius: 8px; padding: 6px; color: #e8e8ee;
            }
            QListWidget::item { padding: 8px; border-radius: 6px; }
            QListWidget::item:selected { background-color: #2f5bff; color: white; }
            QPushButton {
                background-color: #262630; color: #f0f0f5; border: 1px solid #383846;
                border-radius: 6px; padding: 8px 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #343442; }
        """)

        layout = QHBoxLayout(self)

        left_box = QVBoxLayout()
        title = QLabel("Captured Media (Newest First)")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        left_box.addWidget(title)

        self.file_list = QListWidget()
        self.file_list.setFixedWidth(300)
        self.file_list.currentItemChanged.connect(self.on_select_item)
        left_box.addWidget(self.file_list)

        btn_open_folder = QPushButton("Open Captures Folder")
        btn_open_folder.clicked.connect(lambda: os.startfile(BASE_DIR))
        left_box.addWidget(btn_open_folder)
        layout.addLayout(left_box)

        right_box = QVBoxLayout()
        self.hint_label = QLabel("Select a photo or MVIMG Motion Photo")
        self.hint_label.setStyleSheet("color: #f5b800; font-size: 13px; font-weight: bold;")
        right_box.addWidget(self.hint_label)

        self.viewer = LivePhotoView()
        right_box.addWidget(self.viewer, stretch=1)

        action_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play / Stop LIVE")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_extract = QPushButton("Extract MP4")
        self.btn_extract.clicked.connect(self.extract_mp4_to_folder)
        self.btn_gif = QPushButton("Export GIF")
        self.btn_gif.clicked.connect(self.export_gif)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #6b1d24; border-color: #9e2a35;")
        self.btn_delete.clicked.connect(self.delete_selected)

        action_row.addWidget(self.btn_play)
        action_row.addWidget(self.btn_extract)
        action_row.addWidget(self.btn_gif)
        action_row.addStretch()
        action_row.addWidget(self.btn_delete)
        right_box.addLayout(action_row)

        layout.addLayout(right_box)
        self.items_data = []
        self.refresh_gallery()

    def _get_playable_video(self, tag, primary, sibling_vid):
        if tag == "MVIMG":
            if primary not in self.extracted_cache:
                tmp_mp4 = os.path.join(self.temp_dir, f"mv_{len(self.extracted_cache)}.mp4")
                self.extracted_cache[primary] = extract_embedded_mp4(primary, tmp_mp4)
            return self.extracted_cache[primary]
        return sibling_vid

    def refresh_gallery(self):
        self.viewer.stop_playback()
        self.file_list.clear()
        self.items_data.clear()

        entries = []
        for fname in os.listdir(DIRS["live"]):
            if fname.lower().endswith((".jpg", ".jpeg")):
                img_path = os.path.join(DIRS["live"], fname)
                sibling_mp4 = os.path.splitext(img_path)[0] + ".mp4"
                if find_embedded_mp4_offset(img_path) is not None:
                    entries.append((os.path.getmtime(img_path), "MVIMG", fname, img_path, None))
                elif os.path.exists(sibling_mp4):
                    entries.append((os.path.getmtime(img_path), "LIVE", fname, img_path, sibling_mp4))
                else:
                    entries.append((os.path.getmtime(img_path), "PHOTO", fname, img_path, None))

        for fname in os.listdir(DIRS["photos"]):
            if fname.lower().endswith((".jpg", ".png")):
                p = os.path.join(DIRS["photos"], fname)
                tag = "PRO-PNG" if fname.lower().endswith(".png") else "PHOTO"
                entries.append((os.path.getmtime(p), tag, fname, p, None))

        for tag, folder in [("VIDEO", DIRS["videos"]), ("TIMELAPSE", DIRS["timelapse"])]:
            for fname in os.listdir(folder):
                if fname.lower().endswith(".mp4"):
                    p = os.path.join(folder, fname)
                    entries.append((os.path.getmtime(p), tag, fname, p, p))

        entries.sort(key=lambda x: x[0], reverse=True)

        for _, tag, fname, primary, video in entries:
            self.file_list.addItem(QListWidgetItem(f"[{tag}]  {fname}"))
            self.items_data.append((tag, primary, video))

        if self.items_data:
            self.file_list.setCurrentRow(0)
        else:
            self.viewer.load_media(None, None)
            self.hint_label.setText("No captures yet. Shoot a Live Photo first!")

    def on_select_item(self, current, previous):
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self.items_data):
            return
        tag, primary, sibling_vid = self.items_data[row]
        playable = self._get_playable_video(tag, primary, sibling_vid)
        is_live = tag in ("MVIMG", "LIVE")
        self.viewer.load_media(primary, playable, is_live_clip=is_live)

        res_info = ""
        if primary and primary.lower().endswith((".jpg", ".png")):
            im = cv2.imread(primary)
            if im is not None:
                res_info = f" ({im.shape[1]}x{im.shape[0]})"
        elif playable and os.path.exists(playable):
            c = cv2.VideoCapture(playable)
            vw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vfps = int(round(c.get(cv2.CAP_PROP_FPS) or 30))
            c.release()
            res_info = f" ({vw}x{vh} @ {vfps}fps)"

        if tag == "MVIMG":
            self.hint_label.setText(f"ANDROID MOTION PHOTO{res_info} — Press & Hold for smooth real-time motion + audio")
            self.btn_play.setEnabled(True)
            self.btn_extract.setEnabled(True)
            self.btn_gif.setEnabled(True)
        elif tag == "LIVE":
            self.hint_label.setText(f"LIVE PHOTO{res_info} — Press & Hold on image for motion + audio")
            self.btn_play.setEnabled(True)
            self.btn_extract.setEnabled(False)
            self.btn_gif.setEnabled(True)
        elif tag in ("VIDEO", "TIMELAPSE"):
            self.hint_label.setText(f"{tag}{res_info} — Click Play or Hold image to watch")
            self.btn_play.setEnabled(True)
            self.btn_extract.setEnabled(False)
            self.btn_gif.setEnabled(True)
        else:
            self.hint_label.setText(f"{tag}{res_info} — Still Capture")
            self.btn_play.setEnabled(False)
            self.btn_extract.setEnabled(False)
            self.btn_gif.setEnabled(False)

    def toggle_play(self):
        if self.viewer.playing:
            self.viewer.stop_playback()
        else:
            self.viewer.start_playback()

    def extract_mp4_to_folder(self):
        row = self.file_list.currentRow()
        if row < 0:
            return
        tag, primary, _ = self.items_data[row]
        if tag != "MVIMG":
            return
        out_mp4 = os.path.splitext(primary)[0] + "_extracted.mp4"
        if extract_embedded_mp4(primary, out_mp4):
            QMessageBox.information(self, "MP4 Extracted", f"Extracted video to:\n{out_mp4}")

    def export_gif(self):
        if not HAS_PIL:
            QMessageBox.warning(self, "Pillow Missing", "Install Pillow first: pip install Pillow")
            return
        row = self.file_list.currentRow()
        if row < 0:
            return
        tag, primary, sibling_vid = self.items_data[row]
        video = self._get_playable_video(tag, primary, sibling_vid)
        if not video or not os.path.exists(video):
            return

        gif_path = os.path.splitext(primary)[0] + ".gif"
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        pil_frames = []
        step = max(1, int(round(fps / 15.0)))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * 640.0 / w)), interpolation=cv2.INTER_AREA)
                pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            idx += 1
        cap.release()

        if pil_frames:
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=int(1000 / min(15.0, fps)), loop=0, optimize=True
            )
            QMessageBox.information(self, "GIF Exported", f"Saved GIF to:\n{gif_path}")

    def delete_selected(self):
        row = self.file_list.currentRow()
        if row < 0:
            return
        _, primary, sibling_vid = self.items_data[row]
        self.viewer.stop_playback()
        for p in {primary, sibling_vid}:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    QMessageBox.warning(self, "Delete Error", str(e))
        self.refresh_gallery()

    def closeEvent(self, event):
        self.viewer.stop_playback()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().closeEvent(event)


# ==========================================
# Main ProCam Windows Application (v2.7 — Full Background Pipeline)
# ==========================================
class ProCameraWindow(QMainWindow):
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProCam Windows By Ye v2.7 — Butter-Smooth 60fps & 4K Engine")
        self.resize(1220, 695)

        self.cam_reader = None
        self._proc_thread = None
        self._proc_running = False
        self._proc_paused = False
        self._render_lock = threading.Lock()
        self._ready_rgb = None
        self._ready_id = 0
        self._shown_id = 0

        self.ui_fps_times = deque(maxlen=45)
        self.ui_fps = 0.0
        self.cam_fps = 0.0

        self.camera_index = 0
        self.backend_name = "MSMF"
        self.res_options = [
            ("720p HD", (1280, 720)),
            ("1080p FHD", (1920, 1080)),
            ("2K QHD", (2560, 1440)),
            ("4K UHD", (3840, 2160)),
        ]
        self.current_res = (1280, 720)   # Default 720p for max smooth 30/60 FPS on laptops
        self.target_fps = 60

        self.mode = "PRO"
        self.live_photo_enabled = True
        self.mvimg_single_file = True
        self.fast_shutter_lock = False
        self.mic_enabled = True

        self.audio_engine = AudioEngine()
        if HAS_AUDIO:
            ok, _ = self.audio_engine.start()
            self.mic_enabled = ok
        else:
            self.mic_enabled = False

        self.capture_lock = threading.Lock()
        self.live_buffer = deque(maxlen=130)
        self.live_post_capturing = False
        self.live_pre_frames = []
        self.live_post_frames = []
        self.live_still_frame = None
        self.live_shutter_ts = 0.0
        self.live_end_time = 0.0
        self.live_base_path = ""

        self.lut_paths = {}
        self.active_lut = None

        self.is_recording = False
        self.async_recorder = None
        self.active_video_path = ""
        self.rec_start_time = 0.0
        self.rec_frame_count = 0

        self.timelapse_interval = 1.0
        self.timelapse_fps = 30.0
        self.timelapse_last_grab = 0.0

        self.burst_remaining = 0
        self.countdown_val = 0
        self.flash_alpha = 0.0
        self.last_clean_frame = None
        self.prev_temporal_frame = None

        self.settings = {
            "mirror": True,
            "zoom": 1.0,
            "brightness": 0,
            "contrast": 1.0,
            "saturation": 1.0,
            "sharpness": 0.0,
            "filter": "Normal",
            "lut_strength": 1.0,
            "watermark": False,
            "grid": True,
            "timer": 0,
            "hdr_enabled": False,
            "hdr_strength": 0.85,
            "clean_enabled": False,
            "clean_strength": 0.85,
            "auto_exposure": True,
            "ev_offset": 0.0,
            "auto_iso": True,
            "iso_val": 400,
            "auto_focus": True,
            "focus_val": 50,
            "auto_wb": True,
            "wb_kelvin": 5500,
            "histogram": True,
            "focus_peaking": False,
            "zebra": False,
            "save_png": False,
        }

        self._build_ui()
        self._apply_styles()
        self._scan_luts_folder()

        self.status_signal.connect(self.set_status)

        # Lightweight UI paint timer (only blits pre-rendered RGB buffer -> ~0.3ms!)
        self.paint_timer = QTimer(self)
        self.paint_timer.timeout.connect(self._present_frame)

        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._countdown_tick)

        self.burst_timer = QTimer(self)
        self.burst_timer.timeout.connect(self._burst_tick)

        self.open_camera(0)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(6)

        # --- Top Bar ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(5)

        self.btn_live = QPushButton("◉ LIVE")
        self.btn_live.setCheckable(True)
        self.btn_live.setChecked(True)
        self.btn_live.setObjectName("LiveBtn")
        self.btn_live.clicked.connect(self.toggle_live_photo)
        top_bar.addWidget(self.btn_live)

        self.btn_mvimg = QPushButton("MVIMG")
        self.btn_mvimg.setCheckable(True)
        self.btn_mvimg.setChecked(True)
        self.btn_mvimg.clicked.connect(self.toggle_mvimg_format)
        top_bar.addWidget(self.btn_mvimg)

        self.btn_hdr = QPushButton("HDR")
        self.btn_hdr.setCheckable(True)
        self.btn_hdr.setChecked(False)
        self.btn_hdr.setObjectName("HdrBtn")
        self.btn_hdr.clicked.connect(self.toggle_hdr)
        top_bar.addWidget(self.btn_hdr)

        self.btn_clean = QPushButton("✨ Clean")
        self.btn_clean.setCheckable(True)
        self.btn_clean.setChecked(False)
        self.btn_clean.setObjectName("CleanBtn")
        self.btn_clean.clicked.connect(self.toggle_clean)
        top_bar.addWidget(self.btn_clean)

        self.btn_fast_fps = QPushButton("⚡ Fast")
        self.btn_fast_fps.setCheckable(True)
        self.btn_fast_fps.setChecked(False)
        self.btn_fast_fps.setObjectName("FpsBtn")
        self.btn_fast_fps.setToolTip("Lock hardware shutter to boost FPS in dim light")
        self.btn_fast_fps.clicked.connect(self.toggle_fast_shutter)
        top_bar.addWidget(self.btn_fast_fps)

        self.btn_mic = QPushButton("🎙 Mic" if self.mic_enabled else "🔇 Mic")
        self.btn_mic.setCheckable(True)
        self.btn_mic.setChecked(self.mic_enabled)
        self.btn_mic.clicked.connect(self.toggle_mic)
        top_bar.addWidget(self.btn_mic)

        self.btn_grid = QPushButton("Grid")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(True)
        self.btn_grid.clicked.connect(self.toggle_grid)
        top_bar.addWidget(self.btn_grid)

        self.btn_mirror = QPushButton("Mirror")
        self.btn_mirror.setCheckable(True)
        self.btn_mirror.setChecked(True)
        self.btn_mirror.clicked.connect(self.toggle_mirror)
        top_bar.addWidget(self.btn_mirror)

        self.btn_watermark = QPushButton("WM")
        self.btn_watermark.setCheckable(True)
        self.btn_watermark.clicked.connect(lambda v: self.settings.__setitem__("watermark", v))
        top_bar.addWidget(self.btn_watermark)

        self.combo_timer = QComboBox()
        self.combo_timer.addItems(["Timer: Off", "3s", "5s", "10s"])
        self.combo_timer.currentIndexChanged.connect(lambda i: self.settings.__setitem__("timer", [0, 3, 5, 10][i]))
        top_bar.addWidget(self.combo_timer)

        top_bar.addStretch()

        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["MSMF", "DShow"])
        self.combo_backend.currentTextChanged.connect(self.change_backend)
        top_bar.addWidget(self.combo_backend)

        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["Cam 0", "Cam 1", "Cam 2"])
        self.combo_cam.currentIndexChanged.connect(self.open_camera)
        top_bar.addWidget(self.combo_cam)

        self.combo_res = QComboBox()
        for label, _ in self.res_options:
            self.combo_res.addItem(label)
        self.combo_res.setCurrentIndex(0)  # Default 720p HD for instant 30/60 FPS smoothness
        self.combo_res.currentIndexChanged.connect(self.change_resolution)
        top_bar.addWidget(self.combo_res)

        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["30 FPS", "60 FPS"])
        self.combo_fps.setCurrentIndex(1)
        self.combo_fps.currentIndexChanged.connect(self.change_fps)
        top_bar.addWidget(self.combo_fps)

        root_layout.addLayout(top_bar)

        # --- Middle: Fast Preview + Right Tabbed Panel ---
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(10)

        self.preview_view = FastVideoView()
        mid_layout.addWidget(self.preview_view, stretch=3)

        self.right_tabs = QTabWidget()
        self.right_tabs.setFixedWidth(310)

        # TAB 1: PRO MANUAL
        pro_tab = QWidget()
        pro_l = QVBoxLayout(pro_tab)
        pro_l.setSpacing(5)

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Shutter / EV"))
        self.chk_auto_exp = QCheckBox("Auto")
        self.chk_auto_exp.setChecked(True)
        self.chk_auto_exp.toggled.connect(self.on_auto_exposure_toggled)
        self.lbl_exp_val = QLabel("AUTO")
        self.lbl_exp_val.setStyleSheet("color: #f5b800; font-weight: bold;")
        exp_row.addStretch()
        exp_row.addWidget(self.lbl_exp_val)
        exp_row.addWidget(self.chk_auto_exp)
        pro_l.addLayout(exp_row)

        self.slider_exp = QSlider(Qt.Orientation.Horizontal)
        self.slider_exp.setRange(-30, 30)
        self.slider_exp.setValue(0)
        self.slider_exp.valueChanged.connect(self.on_exposure_changed)
        pro_l.addWidget(self.slider_exp)

        iso_row = QHBoxLayout()
        iso_row.addWidget(QLabel("ISO / Gain"))
        self.chk_auto_iso = QCheckBox("Auto")
        self.chk_auto_iso.setChecked(True)
        self.chk_auto_iso.toggled.connect(self.on_auto_iso_toggled)
        self.lbl_iso_val = QLabel("AUTO")
        self.lbl_iso_val.setStyleSheet("color: #f5b800; font-weight: bold;")
        iso_row.addStretch()
        iso_row.addWidget(self.lbl_iso_val)
        iso_row.addWidget(self.chk_auto_iso)
        pro_l.addLayout(iso_row)

        self.slider_iso = QSlider(Qt.Orientation.Horizontal)
        self.slider_iso.setRange(100, 3200)
        self.slider_iso.setSingleStep(50)
        self.slider_iso.setValue(400)
        self.slider_iso.valueChanged.connect(self.on_iso_changed)
        pro_l.addWidget(self.slider_iso)

        foc_row = QHBoxLayout()
        foc_row.addWidget(QLabel("Focus (Lens)"))
        self.chk_auto_foc = QCheckBox("AF")
        self.chk_auto_foc.setChecked(True)
        self.chk_auto_foc.toggled.connect(self.on_auto_focus_toggled)
        self.lbl_foc_val = QLabel("AF")
        self.lbl_foc_val.setStyleSheet("color: #f5b800; font-weight: bold;")
        foc_row.addStretch()
        foc_row.addWidget(self.lbl_foc_val)
        foc_row.addWidget(self.chk_auto_foc)
        pro_l.addLayout(foc_row)

        self.slider_foc = QSlider(Qt.Orientation.Horizontal)
        self.slider_foc.setRange(0, 255)
        self.slider_foc.setValue(50)
        self.slider_foc.valueChanged.connect(self.on_focus_changed)
        pro_l.addWidget(self.slider_foc)

        wb_row = QHBoxLayout()
        wb_row.addWidget(QLabel("White Balance"))
        self.chk_auto_wb = QCheckBox("AWB")
        self.chk_auto_wb.setChecked(True)
        self.chk_auto_wb.toggled.connect(self.on_auto_wb_toggled)
        self.lbl_wb_val = QLabel("AWB")
        self.lbl_wb_val.setStyleSheet("color: #f5b800; font-weight: bold;")
        wb_row.addStretch()
        wb_row.addWidget(self.lbl_wb_val)
        wb_row.addWidget(self.chk_auto_wb)
        pro_l.addLayout(wb_row)

        self.slider_wb = QSlider(Qt.Orientation.Horizontal)
        self.slider_wb.setRange(2500, 8500)
        self.slider_wb.setSingleStep(100)
        self.slider_wb.setValue(5500)
        self.slider_wb.valueChanged.connect(self.on_wb_changed)
        pro_l.addWidget(self.slider_wb)

        self.sliders = {}
        self._add_slider(pro_l, "Clean / Denoise Level", "clean_strength", 0, 100, 85, lambda v: f"{v}%", lambda v: v / 100.0)
        self._add_slider(pro_l, "HDR Intensity", "hdr_strength", 0, 100, 85, lambda v: f"{v}%", lambda v: v / 100.0)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color: #2f2f3d;")
        pro_l.addWidget(div)

        self.chk_hist = QCheckBox("Live RGB Histogram")
        self.chk_hist.setChecked(True)
        self.chk_hist.toggled.connect(lambda v: self.settings.__setitem__("histogram", v))
        pro_l.addWidget(self.chk_hist)

        self.chk_peaking = QCheckBox("Focus Peaking (Sharp Edges)")
        self.chk_peaking.toggled.connect(lambda v: self.settings.__setitem__("focus_peaking", v))
        pro_l.addWidget(self.chk_peaking)

        self.chk_zebra = QCheckBox("Zebra Stripes (Overexposure)")
        self.chk_zebra.toggled.connect(lambda v: self.settings.__setitem__("zebra", v))
        pro_l.addWidget(self.chk_zebra)

        self.chk_png = QCheckBox("Save Lossless PNG (PRO Mode)")
        self.chk_png.toggled.connect(lambda v: self.settings.__setitem__("save_png", v))
        pro_l.addWidget(self.chk_png)

        btn_reset_pro = QPushButton("Reset Pro to Auto")
        btn_reset_pro.clicked.connect(self.reset_hardware_pro)
        pro_l.addWidget(btn_reset_pro)

        pro_l.addStretch()
        self.right_tabs.addTab(pro_tab, "PRO MANUAL")

        # TAB 2: LUT & COLOR
        lut_tab = QWidget()
        lut_l = QVBoxLayout(lut_tab)
        lut_l.setSpacing(6)

        lut_hdr = QLabel("CUSTOM 3D LUT (.cube)")
        lut_hdr.setStyleSheet("color: #f5b800; font-weight: bold;")
        lut_l.addWidget(lut_hdr)

        lut_btn_row = QHBoxLayout()
        btn_load_lut = QPushButton("📂 Load .cube LUT...")
        btn_load_lut.clicked.connect(self.browse_cube_lut)
        btn_clear_lut = QPushButton("Clear")
        btn_clear_lut.clicked.connect(self.clear_lut)
        lut_btn_row.addWidget(btn_load_lut, stretch=2)
        lut_btn_row.addWidget(btn_clear_lut, stretch=1)
        lut_l.addLayout(lut_btn_row)

        self.combo_lut = QComboBox()
        self.combo_lut.currentTextChanged.connect(self.on_lut_selected)
        lut_l.addWidget(self.combo_lut)

        self._add_slider(lut_l, "LUT Strength", "lut_strength", 0, 100, 100, lambda v: f"{v}%", lambda v: v / 100.0)

        lut_l.addWidget(QLabel("Built-in Color Look"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Normal", "Vivid", "Leica B&W", "Warm Film", "Cool Cinema", "Vintage Sepia"])
        self.combo_filter.currentTextChanged.connect(lambda txt: self.settings.__setitem__("filter", txt))
        lut_l.addWidget(self.combo_filter)

        self._add_slider(lut_l, "Digital Zoom", "zoom", 10, 40, 10, lambda v: f"{v/10:.1f}x", lambda v: v / 10.0)
        self._add_slider(lut_l, "Brightness", "brightness", -60, 60, 0, lambda v: f"{v:+d}", lambda v: v)
        self._add_slider(lut_l, "Contrast", "contrast", 50, 200, 100, lambda v: f"{v}%", lambda v: v / 100.0)
        self._add_slider(lut_l, "Saturation", "saturation", 0, 200, 100, lambda v: f"{v}%", lambda v: v / 100.0)
        self._add_slider(lut_l, "Sharpness / Clarity", "sharpness", 0, 150, 0, lambda v: f"{v}%", lambda v: v / 100.0)

        btn_reset_color = QPushButton("Reset Color & LUT")
        btn_reset_color.clicked.connect(self.reset_color_settings)
        lut_l.addWidget(btn_reset_color)

        lut_l.addStretch()
        self.right_tabs.addTab(lut_tab, "LUT & COLOR")

        mid_layout.addWidget(self.right_tabs, stretch=1)
        root_layout.addLayout(mid_layout, stretch=1)

        # --- Time-Lapse Control Bar ---
        self.tl_bar = QFrame()
        self.tl_bar.setObjectName("TimelapseBar")
        tl_layout = QHBoxLayout(self.tl_bar)
        tl_layout.setContentsMargins(12, 6, 12, 6)
        tl_layout.setSpacing(12)

        tl_title = QLabel("⏱ TIMELAPSE SETTINGS")
        tl_title.setStyleSheet("color: #f5b800; font-weight: bold;")
        tl_layout.addWidget(tl_title)

        tl_layout.addWidget(QLabel("Interval:"))
        self.combo_tl_interval = QComboBox()
        self.tl_intervals = [
            ("0.2 sec (Fast)", 0.2),
            ("0.5 sec", 0.5),
            ("1 sec (Standard)", 1.0),
            ("2 sec", 2.0),
            ("3 sec", 3.0),
            ("5 sec (Clouds)", 5.0),
            ("10 sec (Sunset)", 10.0),
            ("30 sec (Long)", 30.0),
            ("60 sec (1 min)", 60.0),
        ]
        for lbl, _ in self.tl_intervals:
            self.combo_tl_interval.addItem(lbl)
        self.combo_tl_interval.setCurrentIndex(2)
        self.combo_tl_interval.currentIndexChanged.connect(self.update_timelapse_settings)
        tl_layout.addWidget(self.combo_tl_interval)

        tl_layout.addWidget(QLabel("Output FPS:"))
        self.combo_tl_fps = QComboBox()
        self.tl_fps_vals = [24.0, 30.0, 60.0]
        self.combo_tl_fps.addItems(["24 fps (Cinema)", "30 fps (Smooth)", "60 fps (Ultra)"])
        self.combo_tl_fps.setCurrentIndex(1)
        self.combo_tl_fps.currentIndexChanged.connect(self.update_timelapse_settings)
        tl_layout.addWidget(self.combo_tl_fps)

        self.lbl_tl_speed = QLabel("Speed: 30x  (1 min real → 2.0s video)")
        self.lbl_tl_speed.setStyleSheet("color: #46e095; font-weight: bold;")
        tl_layout.addWidget(self.lbl_tl_speed)
        tl_layout.addStretch()

        self.tl_bar.setVisible(False)
        root_layout.addWidget(self.tl_bar)

        # --- Bottom Bar ---
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 2, 4, 2)

        left_bottom = QVBoxLayout()
        left_bottom.setSpacing(2)
        self.btn_gallery = QPushButton("🖼 Gallery & MVIMG")
        self.btn_gallery.setFixedHeight(38)
        self.btn_gallery.clicked.connect(self.open_gallery)
        left_bottom.addWidget(self.btn_gallery)

        self.status_label = QLabel("Ready — Dual-Thread Smooth Engine")
        self.status_label.setStyleSheet("color: #a6a7bd; font-size: 11px;")
        left_bottom.addWidget(self.status_label)
        bottom_bar.addLayout(left_bottom)

        bottom_bar.addStretch()

        center_bottom = QVBoxLayout()
        center_bottom.setSpacing(4)
        center_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        for m in ["PHOTO", "PRO", "VIDEO", "BURST", "TIMELAPSE"]:
            btn = QPushButton(m)
            btn.setCheckable(True)
            btn.setObjectName("ModeBtn")
            if m == "PRO":
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, mode=m: self.set_mode(mode))
            self.mode_group.addButton(btn)
            mode_row.addWidget(btn)
        center_bottom.addLayout(mode_row)

        shutter_row = QHBoxLayout()
        shutter_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_shutter = QPushButton("●")
        self.btn_shutter.setFixedSize(58, 58)
        self.btn_shutter.setObjectName("ShutterBtn")
        self.btn_shutter.clicked.connect(self.trigger_shutter)
        shutter_row.addWidget(self.btn_shutter)
        center_bottom.addLayout(shutter_row)

        bottom_bar.addLayout(center_bottom)
        bottom_bar.addStretch()

        right_bottom = QVBoxLayout()
        right_bottom.setSpacing(2)
        btn_folder = QPushButton("📁 Open Captures")
        btn_folder.setFixedHeight(38)
        btn_folder.clicked.connect(lambda: os.startfile(BASE_DIR))
        right_bottom.addWidget(btn_folder)

        hint = QLabel("Space: Shoot | L: Live | D: Clean | H: HDR")
        hint.setStyleSheet("color: #7d7e94; font-size: 11px;")
        right_bottom.addWidget(hint)
        bottom_bar.addLayout(right_bottom)

        root_layout.addLayout(bottom_bar)

    def _add_slider(self, layout, label, key, min_v, max_v, default_v, fmt_fn, val_fn):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        val_lbl = QLabel(fmt_fn(default_v))
        val_lbl.setStyleSheet("color: #c8c9db; font-weight: bold;")
        row.addStretch()
        row.addWidget(val_lbl)
        layout.addLayout(row)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default_v)

        def on_change(v):
            val_lbl.setText(fmt_fn(v))
            self.settings[key] = val_fn(v)

        slider.valueChanged.connect(on_change)
        layout.addWidget(slider)
        self.sliders[key] = (slider, default_v)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #111116; color: #f0f0f6; font-family: 'Segoe UI'; font-size: 12px; }
            QTabWidget::pane { background-color: #191921; border: 1px solid #2b2b36; border-radius: 10px; padding: 6px; }
            QTabBar::tab {
                background-color: #1c1c25; color: #9a9bb2; padding: 7px 14px;
                border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; margin-right: 4px;
            }
            QTabBar::tab:selected { background-color: #282836; color: #f5b800; }
            QFrame#TimelapseBar { background-color: #1a1b24; border: 1px solid #35374a; border-radius: 8px; }
            QPushButton {
                background-color: #23232e; color: #f0f0f6; border: 1px solid #343444;
                border-radius: 7px; padding: 5px 8px; font-weight: 600;
            }
            QPushButton:hover { background-color: #2f2f3d; }
            QPushButton:checked { background-color: #2d5bff; border-color: #557bff; color: white; }
            QPushButton#LiveBtn:checked { background-color: #f5b800; border-color: #ffd042; color: #121216; }
            QPushButton#HdrBtn:checked { background-color: #00b4ff; border-color: #66d4ff; color: #0e1015; }
            QPushButton#CleanBtn:checked { background-color: #20b878; border-color: #4ee0a1; color: #0c1410; }
            QPushButton#FpsBtn:checked { background-color: #ff6b2b; border-color: #ff9466; color: #111116; }
            QPushButton#ModeBtn {
                background-color: transparent; border: none; color: #8f90a6;
                padding: 3px 10px; font-size: 12px; letter-spacing: 1px;
            }
            QPushButton#ModeBtn:checked {
                background-color: transparent; color: #f5b800; border-bottom: 2px solid #f5b800; border-radius: 0px;
            }
            QPushButton#ShutterBtn {
                background-color: #f4f4f8; color: #15151a; border: 4px solid #6a6a7c; border-radius: 29px; font-size: 22px;
            }
            QPushButton#ShutterBtn:hover { background-color: #ffffff; border-color: #f5b800; }
            QComboBox { background-color: #23232e; border: 1px solid #353545; border-radius: 6px; padding: 4px 6px; color: #f0f0f6; }
            QSlider::groove:horizontal { height: 5px; background: #2c2c38; border-radius: 2px; }
            QSlider::handle:horizontal { background: #f5b800; width: 13px; height: 13px; margin: -4px 0; border-radius: 6px; }
        """)

    # ==========================================
    # Time-Lapse Settings
    # ==========================================
    def update_timelapse_settings(self):
        idx = self.combo_tl_interval.currentIndex()
        fidx = self.combo_tl_fps.currentIndex()
        self.timelapse_interval = self.tl_intervals[idx][1]
        self.timelapse_fps = self.tl_fps_vals[fidx]
        speed = self.timelapse_interval * self.timelapse_fps
        out_sec = 60.0 / speed
        self.lbl_tl_speed.setText(f"Speed: {speed:.0f}x  (1 min real → {out_sec:.1f}s video)")
        self.set_status(f"Timelapse: 1 frame every {self.timelapse_interval}s @ {int(self.timelapse_fps)}fps")

    # ==========================================
    # LUT Management
    # ==========================================
    def _scan_luts_folder(self):
        self.combo_lut.blockSignals(True)
        self.combo_lut.clear()
        self.lut_paths.clear()
        self.combo_lut.addItem("None (No LUT)")
        for fname in sorted(os.listdir(DIRS["luts"])):
            if fname.lower().endswith(".cube"):
                self.lut_paths[fname] = os.path.join(DIRS["luts"], fname)
                self.combo_lut.addItem(fname)
        self.combo_lut.blockSignals(False)

    def browse_cube_lut(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Custom .cube LUT", DIRS["luts"], "Cube LUT Files (*.cube);;All Files (*)")
        if not path:
            return
        fname = os.path.basename(path)
        dest = os.path.join(DIRS["luts"], fname)
        if os.path.abspath(path) != os.path.abspath(dest):
            try:
                shutil.copy2(path, dest)
                path = dest
            except Exception:
                pass
        self._scan_luts_folder()
        self.lut_paths[fname] = path
        if self.combo_lut.findText(fname) < 0:
            self.combo_lut.addItem(fname)
        self.combo_lut.setCurrentText(fname)

    def on_lut_selected(self, name):
        if not name or name.startswith("None"):
            self.active_lut = None
            self.set_status("LUT cleared")
            return
        path = self.lut_paths.get(name)
        if not path or not os.path.exists(path):
            return
        try:
            self.set_status(f"Loading LUT: {name}...")
            QApplication.processEvents()
            self.active_lut = CubeLUT(path)
            self.set_status(f"LUT Applied: {name}")
        except Exception as e:
            self.active_lut = None
            QMessageBox.warning(self, "LUT Load Error", f"Failed to load {name}:\n{e}")

    def clear_lut(self):
        self.combo_lut.setCurrentIndex(0)
        self.active_lut = None

    # ==========================================
    # PRO Mode Controls
    # ==========================================
    def on_auto_exposure_toggled(self, auto_on):
        self.settings["auto_exposure"] = auto_on
        if auto_on:
            self.slider_exp.blockSignals(True)
            self.slider_exp.setValue(0)
            self.slider_exp.blockSignals(False)
            self.settings["ev_offset"] = 0.0
            self.lbl_exp_val.setText("AUTO")
        else:
            self.on_exposure_changed(self.slider_exp.value())

    def on_exposure_changed(self, raw_val):
        if self.chk_auto_exp.isChecked():
            self.chk_auto_exp.blockSignals(True)
            self.chk_auto_exp.setChecked(False)
            self.chk_auto_exp.blockSignals(False)
        self.settings["auto_exposure"] = False
        ev = raw_val / 10.0
        self.settings["ev_offset"] = ev
        shutter_denom = max(4, min(1000, int(round(60.0 / (2.0 ** ev)))))
        self.lbl_exp_val.setText(f"1/{shutter_denom}s ({ev:+.1f}EV)")

    def on_auto_iso_toggled(self, auto_on):
        self.settings["auto_iso"] = auto_on
        if auto_on:
            self.slider_iso.blockSignals(True)
            self.slider_iso.setValue(400)
            self.slider_iso.blockSignals(False)
            self.settings["iso_val"] = 400
            self.lbl_iso_val.setText("AUTO")
        else:
            self.on_iso_changed(self.slider_iso.value())

    def on_iso_changed(self, iso):
        if self.chk_auto_iso.isChecked():
            self.chk_auto_iso.blockSignals(True)
            self.chk_auto_iso.setChecked(False)
            self.chk_auto_iso.blockSignals(False)
        self.settings["auto_iso"] = False
        self.settings["iso_val"] = iso
        self.lbl_iso_val.setText(f"ISO {iso}")

    def on_auto_focus_toggled(self, auto_on):
        self.settings["auto_focus"] = auto_on
        if auto_on:
            self.lbl_foc_val.setText("AF")
            if self.cam_reader is not None:
                self.cam_reader.set_prop(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            self.on_focus_changed(self.slider_foc.value())

    def on_focus_changed(self, val):
        if self.chk_auto_foc.isChecked():
            self.chk_auto_foc.blockSignals(True)
            self.chk_auto_foc.setChecked(False)
            self.chk_auto_foc.blockSignals(False)
        self.settings["auto_focus"] = False
        self.settings["focus_val"] = val
        self.lbl_foc_val.setText(f"MF {val}")
        if self.cam_reader is not None:
            self.cam_reader.set_prop(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cam_reader.set_prop(cv2.CAP_PROP_FOCUS, float(val))

    def on_auto_wb_toggled(self, auto_on):
        self.settings["auto_wb"] = auto_on
        if auto_on:
            self.slider_wb.blockSignals(True)
            self.slider_wb.setValue(5500)
            self.slider_wb.blockSignals(False)
            self.settings["wb_kelvin"] = 5500
            self.lbl_wb_val.setText("AWB")
        else:
            self.on_wb_changed(self.slider_wb.value())

    def on_wb_changed(self, kelvin):
        if self.chk_auto_wb.isChecked():
            self.chk_auto_wb.blockSignals(True)
            self.chk_auto_wb.setChecked(False)
            self.chk_auto_wb.blockSignals(False)
        self.settings["auto_wb"] = False
        self.settings["wb_kelvin"] = kelvin
        self.lbl_wb_val.setText(f"{kelvin}K")

    def reset_hardware_pro(self):
        self.chk_auto_exp.setChecked(True)
        self.chk_auto_iso.setChecked(True)
        self.chk_auto_foc.setChecked(True)
        self.chk_auto_wb.setChecked(True)
        self.on_auto_exposure_toggled(True)
        self.on_auto_iso_toggled(True)
        self.on_auto_focus_toggled(True)
        self.on_auto_wb_toggled(True)
        self.set_status("Pro Manual controls reset to Auto")

    def reset_color_settings(self):
        self.clear_lut()
        self.combo_filter.setCurrentText("Normal")
        for key in ("lut_strength", "zoom", "brightness", "contrast", "saturation", "sharpness"):
            slider, default_v = self.sliders[key]
            slider.setValue(default_v)
        self.set_status("Color & LUT settings reset")

    # ==========================================
    # Dual-Thread Camera & Processing Lifecycle
    # ==========================================
    def _stop_pipeline(self):
        self.paint_timer.stop()
        self._proc_running = False
        if self._proc_thread is not None:
            self._proc_thread.join(timeout=1.5)
            self._proc_thread = None
        if self.cam_reader is not None:
            self.cam_reader.stop()
            self.cam_reader.join(timeout=1.5)
            self.cam_reader = None

    def open_camera(self, index):
        if self.is_recording:
            self.stop_recording()
        self._stop_pipeline()

        self.camera_index = index
        self.prev_temporal_frame = None
        self.ui_fps_times.clear()
        self.live_buffer.clear()

        w, h = self.current_res
        self.preview_view.set_message(f"Opening Cam {index} ({w}x{h} @ {self.target_fps}fps)...")
        QApplication.processEvents()

        reader = CameraReader(index, w, h, target_fps=self.target_fps, backend_name=self.backend_name)
        reader.start()
        reader._init_done.wait(timeout=4.5)

        if not reader.ready:
            reader.stop()
            self.preview_view.set_message(f"{reader.error_msg}\nConnect a webcam or DroidCam/Camo.")
            self.set_status(reader.error_msg)
            return

        self.cam_reader = reader
        if self.fast_shutter_lock:
            self.cam_reader.set_fast_shutter_lock(True)

        if reader.actual_w < w or reader.actual_h < h:
            res_msg = f"Sensor {reader.actual_w}x{reader.actual_h} → HQ {w}x{h}"
        else:
            res_msg = f"Native {reader.actual_w}x{reader.actual_h}"
        self.set_status(f"Cam {index} ({reader.active_backend} | {res_msg})")

        # Start dedicated background processing thread so UI thread is 0% loaded
        self._proc_running = True
        self._proc_paused = False
        self._proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._proc_thread.start()
        self.paint_timer.start(12)

    def _build_hud_text(self, cfg, active_lut):
        ev_txt = "AUTO" if cfg["auto_exposure"] else f"{cfg['ev_offset']:+.1f}EV"
        iso = "ISO AUTO" if cfg["auto_iso"] else f"ISO {cfg['iso_val']}"
        wb = "AWB" if cfg["auto_wb"] else f"{cfg['wb_kelvin']}K"
        hdr = "HDR" if cfg["hdr_enabled"] else "STD"
        res_lbl = self.res_options[self.combo_res.currentIndex()][0].split()[0]
        return f" CAM {self.cam_fps:.0f} | UI {self.ui_fps:.0f} FPS | {res_lbl} | {self.mode} | EV {ev_txt} | {iso} | WB {wb} | {hdr}"

    def _processing_loop(self):
        last_fid = -1
        while self._proc_running:
            if self._proc_paused or self.cam_reader is None:
                time.sleep(0.03)
                continue

            fid, frame_ts, raw, cam_fps = self.cam_reader.get_latest()
            if raw is None or fid == last_fid:
                time.sleep(0.002)
                continue
            last_fid = fid
            self.cam_fps = cam_fps

            now = time.perf_counter()
            cfg = dict(self.settings)
            lut = self.active_lut

            clean_frame, self.prev_temporal_frame = apply_image_pipeline(
                raw, cfg, lut, self.prev_temporal_frame
            )
            self.last_clean_frame = clean_frame

            # Live Photo buffer + Async Recording (all off UI thread!)
            rec_text = ""
            live_active_now = False
            with self.capture_lock:
                if self.mode in ("PHOTO", "PRO") and self.live_photo_enabled:
                    self.live_buffer.append((now, clean_frame))

                if self.live_post_capturing:
                    live_active_now = True
                    self.live_post_frames.append((now, clean_frame))
                    if now >= self.live_end_time:
                        self.live_post_capturing = False
                        combined = self.live_pre_frames + self.live_post_frames
                        audio_np = self.audio_engine.finish_live_capture() if self.mic_enabled else None
                        args = (
                            combined, audio_np, self.live_still_frame,
                            self.live_shutter_ts, self.live_base_path,
                            self.mvimg_single_file, self.current_res, self.target_fps
                        )
                        threading.Thread(target=self._encode_live_av, args=args, daemon=True).start()

                if self.is_recording and self.async_recorder is not None:
                    elapsed = int(now - self.rec_start_time)
                    mm, ss = divmod(elapsed, 60)
                    if self.mode == "VIDEO":
                        if self.async_recorder.push_frame(clean_frame):
                            self.rec_frame_count += 1
                        res_tag = self.res_options[self.combo_res.currentIndex()][0].split()[0]
                        rec_text = f"REC {res_tag} {mm:02d}:{ss:02d} ({self.cam_fps:.0f}fps)"
                    elif self.mode == "TIMELAPSE":
                        if now - self.timelapse_last_grab >= self.timelapse_interval:
                            if self.async_recorder.push_frame(clean_frame):
                                self.rec_frame_count += 1
                            self.timelapse_last_grab = now
                        vid_sec = self.rec_frame_count / max(1.0, self.timelapse_fps)
                        rec_text = f"TL {mm:02d}:{ss:02d} | {self.rec_frame_count}f ({vid_sec:.1f}s)"

            if self.flash_alpha > 0.0:
                self.flash_alpha = max(0.0, self.flash_alpha - 0.18)

            vw = max(480, self.preview_view.width())
            vh = max(300, self.preview_view.height())
            rgb_vis = make_preview_rgb(
                clean_frame, vw, vh, cfg,
                self.countdown_val, self.flash_alpha,
                live_active_now, rec_text, self._build_hud_text(cfg, lut)
            )

            with self._render_lock:
                self._ready_rgb = rgb_vis
                self._ready_id += 1

    def _present_frame(self):
        with self._render_lock:
            if self._ready_id == self._shown_id or self._ready_rgb is None:
                return
            rgb = self._ready_rgb
            self._shown_id = self._ready_id

        now = time.perf_counter()
        self.ui_fps_times.append(now)
        if len(self.ui_fps_times) >= 2:
            span = self.ui_fps_times[-1] - self.ui_fps_times[0]
            if span > 0.05:
                self.ui_fps = (len(self.ui_fps_times) - 1) / span

        self.preview_view.set_rgb_buffer(rgb)

    def change_backend(self, name):
        self.backend_name = name
        self.open_camera(self.camera_index)

    def change_resolution(self, idx):
        self.current_res = self.res_options[idx][1]
        self.open_camera(self.camera_index)

    def change_fps(self, idx):
        self.target_fps = 30 if idx == 0 else 60
        self.open_camera(self.camera_index)

    def toggle_fast_shutter(self, checked):
        self.fast_shutter_lock = checked
        if self.cam_reader is not None:
            self.cam_reader.set_fast_shutter_lock(checked)
        self.set_status(
            f"⚡ Fast Shutter Lock ON (boosts FPS toward {self.target_fps}fps)"
            if checked else "⚡ Fast Shutter Lock OFF (Auto Low-Light)"
        )

    def toggle_live_photo(self, checked):
        self.live_photo_enabled = checked
        self.btn_live.setText("◉ LIVE" if checked else "○ LIVE")
        if not checked:
            with self.capture_lock:
                self.live_buffer.clear()

    def toggle_mvimg_format(self, checked):
        self.mvimg_single_file = checked
        self.set_status("Output: Single-file MVIMG_*.jpg" if checked else "Output: Paired JPG + MP4")

    def toggle_hdr(self, checked):
        self.settings["hdr_enabled"] = checked
        self.set_status("HDR Mode ON" if checked else "HDR Mode OFF")

    def toggle_clean(self, checked):
        self.settings["clean_enabled"] = checked
        self.prev_temporal_frame = None
        self.set_status("Clean Denoise & Clarity ON" if checked else "Clean Denoise OFF")

    def toggle_mic(self, checked):
        if checked:
            ok, msg = self.audio_engine.start()
            self.mic_enabled = ok
            self.btn_mic.setChecked(ok)
            self.btn_mic.setText("🎙 Mic" if ok else "🔇 Mic")
            self.set_status(msg)
        else:
            self.audio_engine.stop()
            self.mic_enabled = False
            self.btn_mic.setText("🔇 Mic")
            self.set_status("Microphone muted")

    def toggle_grid(self, checked):
        self.settings["grid"] = checked

    def toggle_mirror(self, checked):
        self.settings["mirror"] = checked
        self.prev_temporal_frame = None

    def set_mode(self, mode):
        if self.is_recording:
            self.stop_recording()
        self.mode = mode
        self.tl_bar.setVisible(mode == "TIMELAPSE")
        if mode == "PRO":
            self.right_tabs.setCurrentIndex(0)
        if mode in ("VIDEO", "TIMELAPSE"):
            self.btn_shutter.setStyleSheet(
                "background-color: #ff3b30; color: white; border: 4px solid #6a6a7c; border-radius: 29px; font-size: 20px;"
            )
        else:
            self.btn_shutter.setStyleSheet("")
        self.btn_shutter.setText("●")
        if mode == "TIMELAPSE":
            self.update_timelapse_settings()
        else:
            self.set_status(f"Mode: {mode}")

    def set_status(self, text):
        self.status_label.setText(text)

    # ==========================================
    # Capture Actions
    # ==========================================
    def trigger_shutter(self):
        if self.last_clean_frame is None:
            return
        if self.is_recording:
            self.stop_recording()
            return
        if self.settings["timer"] > 0 and self.countdown_val == 0:
            self.countdown_val = self.settings["timer"]
            self.countdown_timer.start(1000)
            return
        self._execute_capture()

    def _countdown_tick(self):
        self.countdown_val -= 1
        if self.countdown_val <= 0:
            self.countdown_timer.stop()
            self.countdown_val = 0
            self._execute_capture()

    def _execute_capture(self):
        if self.mode in ("PHOTO", "PRO"):
            self.capture_photo(is_live=self.live_photo_enabled)
        elif self.mode == "BURST":
            self.burst_remaining = 5
            self.burst_timer.start(180)
        elif self.mode in ("VIDEO", "TIMELAPSE"):
            self.start_recording()

    def _polish_and_scale_still(self, frame, target_res):
        out = frame
        if self.settings["clean_enabled"]:
            try:
                h_str = max(3, int(6 * self.settings["clean_strength"]))
                out = cv2.fastNlMeansDenoisingColored(out, None, h_str, h_str, 5, 11)
            except Exception:
                pass
        tw, th = target_res
        return upscale_hq_still(out, tw, th)

    def capture_photo(self, is_live=False, suffix=""):
        if self.last_clean_frame is None:
            return
        frame = self.last_clean_frame.copy()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        target_res = self.current_res
        self.flash_alpha = 0.75

        if is_live:
            with self.capture_lock:
                if not self.live_post_capturing:
                    now = time.perf_counter()
                    prefix = "MVIMG_" if self.mvimg_single_file else "LIVE_"
                    self.live_base_path = os.path.join(DIRS["live"], f"{prefix}{ts}")
                    self.live_still_frame = frame
                    self.live_shutter_ts = now

                    if not self.mvimg_single_file:
                        threading.Thread(
                            target=self._save_polished_still,
                            args=(frame, self.live_base_path + ".jpg", False, target_res),
                            daemon=True
                        ).start()

                    if self.mode == "PRO" and self.settings["save_png"]:
                        threading.Thread(
                            target=self._save_polished_still,
                            args=(frame, os.path.join(DIRS["photos"], f"PRO_{ts}.png"), True, target_res),
                            daemon=True
                        ).start()

                    self.live_pre_frames = [
                        (t, f) for (t, f) in list(self.live_buffer) if t >= now - 1.5
                    ]
                    self.live_post_frames = [(now, frame)]
                    self.live_end_time = now + 1.5
                    if self.mic_enabled:
                        self.audio_engine.begin_live_capture(pre_seconds=1.5)
                    self.live_post_capturing = True
                    self.set_status("Capturing Smooth 60fps MVIMG + Audio (1.5s)...")
        else:
            is_png = (self.mode == "PRO" and self.settings["save_png"])
            out_path = os.path.join(
                DIRS["photos"],
                f"PRO_{ts}{suffix}.png" if is_png else f"IMG_{ts}{suffix}.jpg"
            )
            self.set_status(f"Saving {target_res[0]}x{target_res[1]} photo...")
            threading.Thread(
                target=self._save_polished_still,
                args=(frame, out_path, is_png, target_res),
                daemon=True
            ).start()

    def _save_polished_still(self, frame, out_path, is_png, target_res):
        final_img = self._polish_and_scale_still(frame, target_res)
        if is_png:
            cv2.imwrite(out_path, final_img)
        else:
            cv2.imwrite(out_path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        self.status_signal.emit(f"Saved ({final_img.shape[1]}x{final_img.shape[0]}): {os.path.basename(out_path)}")

    def _encode_live_av(self, frames_with_ts, audio_np, still_frame, shutter_ts, base_path, as_mvimg, target_res, target_fps):
        if len(frames_with_ts) < 2:
            return
        first_ts = frames_with_ts[0][0]
        keyframe_us = max(0, int((shutter_ts - first_ts) * 1_000_000))

        temp_wav = None
        if audio_np is not None and len(audio_np) > 0:
            temp_wav = tempfile.mktemp(suffix="_live.wav")
            self.audio_engine.save_wav(audio_np, temp_wav)

        muxed_mp4 = tempfile.mktemp(suffix="_live_mux.mp4") if as_mvimg else (base_path + ".mp4")
        ok = encode_smooth_cfr_live_mp4(frames_with_ts, temp_wav, muxed_mp4, target_fps=target_fps)
        tag = "Audio" if audio_np is not None and len(audio_np) > 0 else "Silent"

        if not ok:
            self.status_signal.emit("Live Photo encode failed")
            return

        if as_mvimg:
            final_mvimg = base_path + ".jpg"
            try:
                final_still = self._polish_and_scale_still(still_frame, target_res)
                pack_android_motion_photo(final_still, muxed_mp4, final_mvimg, keyframe_us=keyframe_us)
                self.status_signal.emit(
                    f"Saved Smooth MVIMG ({final_still.shape[1]}x{final_still.shape[0]} | {tag}): {os.path.basename(final_mvimg)}"
                )
            except Exception as e:
                self.status_signal.emit(f"MVIMG Error: {e}")
            finally:
                if os.path.exists(muxed_mp4):
                    try:
                        os.remove(muxed_mp4)
                    except Exception:
                        pass
        else:
            self.status_signal.emit(f"Saved Paired LIVE ({tag}): {os.path.basename(muxed_mp4)}")

    def _burst_tick(self):
        if self.burst_remaining <= 0:
            self.burst_timer.stop()
            self.set_status("Burst capture complete (5 photos)")
            return
        self.capture_photo(is_live=False, suffix=f"_B{6 - self.burst_remaining}")
        self.burst_remaining -= 1

    def start_recording(self):
        if self.last_clean_frame is None:
            return
        in_h, in_w = self.last_clean_frame.shape[:2]
        out_w, out_h = self.current_res
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        with self.capture_lock:
            self.rec_frame_count = 0
            self.rec_start_time = time.perf_counter()

            if self.mode == "VIDEO":
                self.active_video_path = os.path.join(DIRS["videos"], f"VID_{ts}.mp4")
                self.async_recorder = AsyncVideoRecorder(in_w, in_h, out_w, out_h, target_fps=self.target_fps)
                if self.mic_enabled:
                    self.audio_engine.start_video_recording()
            else:
                self.active_video_path = os.path.join(DIRS["timelapse"], f"TL_{ts}.mp4")
                self.async_recorder = AsyncVideoRecorder(in_w, in_h, out_w, out_h, target_fps=self.timelapse_fps)
                self.timelapse_last_grab = 0.0

            self.is_recording = True

        self.btn_shutter.setText("■")
        self.set_status(f"Recording {self.mode} ({out_w}x{out_h} Async)...")

    def stop_recording(self):
        with self.capture_lock:
            self.is_recording = False
            recorder = self.async_recorder
            self.async_recorder = None

        self.btn_shutter.setText("●")
        if recorder is None:
            return

        mode_now = self.mode
        final_mp4 = self.active_video_path
        tl_fps = self.timelapse_fps
        audio_np = (
            self.audio_engine.stop_video_recording()
            if (mode_now == "VIDEO" and self.mic_enabled)
            else None
        )

        self.set_status(f"Finalizing {mode_now}...")
        threading.Thread(
            target=self._finalize_async_recording,
            args=(recorder, audio_np, final_mp4, mode_now, tl_fps),
            daemon=True
        ).start()

    def _finalize_async_recording(self, recorder, audio_np, final_mp4, mode_now, tl_fps):
        temp_path, is_raw_h264, frames_written, elapsed = recorder.stop_and_get_info()
        if frames_written < 2:
            self.status_signal.emit(f"{mode_now} too short (no frames saved)")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return

        temp_wav = None
        if audio_np is not None and len(audio_np) > 0:
            temp_wav = tempfile.mktemp(suffix="_vid.wav")
            self.audio_engine.save_wav(audio_np, temp_wav)

        out_fps = (frames_written / elapsed) if mode_now == "VIDEO" else tl_fps
        finalize_stream_to_mp4(temp_path, is_raw_h264, temp_wav, final_mp4, output_fps=out_fps)
        self.status_signal.emit(f"Saved {mode_now} ({recorder.out_w}x{recorder.out_h}): {os.path.basename(final_mp4)}")

    def open_gallery(self):
        # Pause main camera pipeline while Gallery is open for 100% smooth playback
        self._proc_paused = True
        self.paint_timer.stop()
        if self.cam_reader is not None:
            self.cam_reader.set_paused(True)

        try:
            dlg = GalleryDialog(self)
            dlg.exec()
        finally:
            if self.cam_reader is not None:
                self.cam_reader.set_paused(False)
            self._proc_paused = False
            self.paint_timer.start(12)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.trigger_shutter()
        elif key == Qt.Key.Key_L:
            self.btn_live.click()
        elif key == Qt.Key.Key_H:
            self.btn_hdr.click()
        elif key == Qt.Key.Key_D:
            self.btn_clean.click()
        elif key == Qt.Key.Key_G:
            self.btn_grid.click()
        elif key == Qt.Key.Key_M:
            self.btn_mirror.click()
        elif key == Qt.Key.Key_P:
            self.chk_peaking.setChecked(not self.chk_peaking.isChecked())
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            slider, _ = self.sliders["zoom"]
            slider.setValue(min(slider.maximum(), slider.value() + 2))
        elif key == Qt.Key.Key_Minus:
            slider, _ = self.sliders["zoom"]
            slider.setValue(max(slider.minimum(), slider.value() - 2))
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording()
        self.audio_engine.stop()
        self._stop_pipeline()
        super().closeEvent(event)


# ==========================================
# Animated Splash Screen (Zoom + Gold Sweep + Pulse)
# ==========================================
class AnimatedSplashScreen(QWidget):
    def __init__(self, icon_path=""):
        super().__init__()
        self.icon_path = icon_path
        self.setFixedSize(560, 355)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.SplashScreen
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.logo_pix = None
        for fname in ("logo.jpg", "logo.png", "app_icon.ico"):
            p = resource_path(fname)
            if os.path.exists(p):
                pix = QPixmap(p)
                if not pix.isNull():
                    self.logo_pix = pix
                    break

        self.elapsed = 0.0
        self.duration = 2.0
        self.progress = 0.0
        self.opacity_val = 0.0
        self.fading_out = False
        self.status_text = "Initializing Smooth 60fps Engine..."
        self.main_win = None

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            screen.center().x() - self.width() // 2,
            screen.center().y() - self.height() // 2
        )
        self.setWindowOpacity(0.0)

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._tick)
        self.anim_timer.start(16)

    def _tick(self):
        if self.fading_out:
            self.opacity_val = max(0.0, self.opacity_val - 0.08)
            self.setWindowOpacity(self.opacity_val)
            if self.opacity_val <= 0.0:
                self.anim_timer.stop()
                self.close()
                if self.main_win is not None:
                    self.main_win.show()
            return

        self.elapsed += 0.016
        if self.elapsed < 0.30:
            self.opacity_val = min(1.0, self.elapsed / 0.30)
            self.setWindowOpacity(self.opacity_val)
        elif self.opacity_val < 1.0:
            self.opacity_val = 1.0
            self.setWindowOpacity(1.0)

        raw_p = min(1.0, self.elapsed / self.duration)
        self.progress = 1.0 - (1.0 - raw_p) ** 2

        if self.progress < 0.30:
            self.status_text = "Initializing Smooth 60fps Engine..."
        elif self.progress < 0.65:
            self.status_text = "Loading RAM-Cached MVIMG & Audio..."
        elif self.progress < 0.90:
            self.status_text = "Preparing Pro Controls & 3D LUT..."
        else:
            self.status_text = "Starting Camera..."

        self.update()

        if raw_p >= 1.0 and self.main_win is None:
            self.status_text = "Ready!"
            self.repaint()
            QApplication.processEvents()

            self.main_win = ProCameraWindow()
            if self.icon_path and os.path.exists(self.icon_path):
                self.main_win.setWindowIcon(QIcon(self.icon_path))
            self.fading_out = True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        rect = QRectF(4, 4, w - 8, h - 8)

        clip_path = QPainterPath()
        clip_path.addRoundedRect(rect, 22, 22)
        painter.setClipPath(clip_path)
        painter.fillRect(rect, QColor("#0c0d11"))

        if self.logo_pix is not None:
            scale = 1.0 + 0.07 * self.progress
            float_y = math.sin(self.elapsed * 2.6) * 2.5

            pw, ph = self.logo_pix.width(), self.logo_pix.height()
            base_scale = max(w / pw, h / ph)
            draw_w = pw * base_scale * scale
            draw_h = ph * base_scale * scale
            dx = (w - draw_w) / 2.0
            dy = (h - draw_h) / 2.0 - 10.0 + float_y

            painter.drawPixmap(QRectF(dx, dy, draw_w, draw_h), self.logo_pix, QRectF(0, 0, pw, ph))

        sheen_cycle = (self.elapsed * 0.72) % 1.6
        if sheen_cycle <= 1.0:
            sx = -w * 0.4 + (w * 1.8) * sheen_cycle
            sheen = QLinearGradient(sx - 90, 0, sx + 90, h)
            sheen.setColorAt(0.0, QColor(255, 255, 255, 0))
            sheen.setColorAt(0.5, QColor(255, 232, 160, 75))
            sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(rect, sheen)

        btm_grad = QLinearGradient(0, h - 92, 0, h)
        btm_grad.setColorAt(0.0, QColor(10, 11, 15, 0))
        btm_grad.setColorAt(0.55, QColor(10, 11, 15, 205))
        btm_grad.setColorAt(1.0, QColor(10, 11, 15, 245))
        painter.fillRect(QRectF(0, h - 92, w, 92), btm_grad)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QColor("#d4d6e4"))
        painter.drawText(QRectF(28, h - 54, w - 120, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.status_text)

        pct_text = f"{int(self.progress * 100)}%"
        painter.setPen(QColor("#f5b800"))
        painter.drawText(QRectF(w - 92, h - 54, 64, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, pct_text)

        bar_x, bar_y, bar_w, bar_h = 28.0, h - 28.0, w - 56.0, 5.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 35))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2.5, 2.5)

        fill_w = max(8.0, bar_w * self.progress)
        bar_grad = QLinearGradient(bar_x, 0, bar_x + fill_w, 0)
        bar_grad.setColorAt(0.0, QColor("#2d9cff"))
        bar_grad.setColorAt(0.6, QColor("#f5b800"))
        bar_grad.setColorAt(1.0, QColor("#fff0a6"))
        painter.setBrush(bar_grad)
        painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2.5, 2.5)

        painter.setClipping(False)
        pulse = int(110 + 95 * (0.5 + 0.5 * math.sin(self.elapsed * 5.0)))
        pen = QPen(QColor(245, 184, 0, pulse), 1.8)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 22, 22)


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ProCam.Windows.ByYe.2.7")

    app = QApplication(sys.argv)
    icon_file = resource_path("app_icon.ico")
    if os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))

    splash = AnimatedSplashScreen(icon_path=icon_file)
    splash.show()
    sys.exit(app.exec())
