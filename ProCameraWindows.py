import sys
import os
import re
import math
import time
import wave
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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

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
# Storage Directories
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


# ==========================================
# Android Motion Photo (MVIMG) Engine
# Compatible with Google Photos & Xiaomi/HyperOS Gallery
# ==========================================
def build_exif_app1_xiaomi(dt_str=None):
    """Builds a minimal EXIF APP1 segment including Xiaomi Motion Photo tag 0x8897 = 1."""
    if dt_str is None:
        dt_str = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    make_b = b"ProCam\x00"              # 7 bytes (+1 pad = 8)
    model_b = b"ProCam Windows\x00"     # 15 bytes (+1 pad = 16)
    dt_b = dt_str.encode("ascii") + b"\x00"  # 20 bytes

    make_off = 62
    model_off = 70
    dt_off = 86
    exif_ifd_off = 106

    # Little-endian TIFF header (8 bytes)
    header = b"II" + struct.pack("<HI", 42, 8)

    # IFD0: 4 entries (2 + 48 + 4 = 54 bytes -> 8..62)
    ifd0 = struct.pack("<H", 4)
    ifd0 += struct.pack("<HHII", 0x010F, 2, len(make_b), make_off)       # Make
    ifd0 += struct.pack("<HHII", 0x0110, 2, len(model_b), model_off)     # Model
    ifd0 += struct.pack("<HHII", 0x0132, 2, len(dt_b), dt_off)           # DateTime
    ifd0 += struct.pack("<HHII", 0x8769, 4, 1, exif_ifd_off)             # ExifIFDPointer
    ifd0 += struct.pack("<I", 0)

    # String data block (44 bytes -> 62..106)
    data_block = make_b + b"\x00" + model_b + b"\x00" + dt_b

    # Exif SubIFD: 3 entries (starts at 106)
    exif_ifd = struct.pack("<H", 3)
    # Xiaomi Gallery Live Photo identifier: Tag 0x8897 (BYTE, count=1, value=1)
    exif_ifd += struct.pack("<HHI", 0x8897, 1, 1) + struct.pack("<B3x", 1)
    exif_ifd += struct.pack("<HHII", 0x9003, 2, len(dt_b), dt_off)       # DateTimeOriginal
    exif_ifd += struct.pack("<HHII", 0x9004, 2, len(dt_b), dt_off)       # DateTimeDigitized
    exif_ifd += struct.pack("<I", 0)

    tiff = header + ifd0 + data_block + exif_ifd
    payload = b"Exif\x00\x00" + tiff
    return b"\xFF\xE1" + struct.pack(">H", len(payload) + 2) + payload


def build_motion_photo_xmp_app1(mp4_size, timestamp_us):
    """Builds Google Motion Photo 1.0 (Container:Directory) + Legacy GCamera MicroVideo XMP APP1."""
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
    """Packs a BGR still frame + MP4 video into a single-file Android Motion Photo (MVIMG_*.jpg)."""
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

    # Insert EXIF + XMP right after SOI (\xFF\xD8) and optional APP0 JFIF (\xFF\xE0)
    insert_pos = 2
    if len(jpg_bytes) > 6 and jpg_bytes[2:4] == b"\xFF\xE0":
        app0_len = struct.unpack(">H", jpg_bytes[4:6])[0]
        insert_pos = 4 + app0_len

    final_jpg = jpg_bytes[:insert_pos] + exif_app1 + xmp_app1 + jpg_bytes[insert_pos:]

    # Atomic write: [JPEG + EXIF(0x8897) + XMP] + [MP4 bytes]
    tmp_out = output_mvimg_path + ".tmp"
    with open(tmp_out, "wb") as f:
        f.write(final_jpg)
        f.write(mp4_bytes)
    os.replace(tmp_out, output_mvimg_path)
    return output_mvimg_path


def find_embedded_mp4_offset(jpg_path):
    """Returns byte start index of embedded MP4 inside an Android Motion Photo, or None."""
    try:
        file_size = os.path.getsize(jpg_path)
        if file_size < 1024:
            return None
        with open(jpg_path, "rb") as f:
            head = f.read(min(file_size, 65536))

            # 1. Check XMP MicroVideoOffset or Container Item:Length
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

            # 2. Fallback: scan for MP4 'ftyp' box after JPEG EOI (\xFF\xD9)
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
    """Extracts the embedded MP4 track from a Motion Photo JPG byte-for-byte."""
    start = find_embedded_mp4_offset(jpg_path)
    if start is None:
        return None
    with open(jpg_path, "rb") as src, open(dest_mp4_path, "wb") as dst:
        src.seek(start)
        shutil.copyfileobj(src, dst)
    return dest_mp4_path


# ==========================================
# Custom 3D/1D .cube LUT Engine (Fast Trilinear Pre-Bake)
# ==========================================
class CubeLUT:
    """Loads standard Adobe .cube 3D/1D LUTs and bakes a fast 128^3 BGR lookup table."""

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
            out_1d = np.empty((256, 3), dtype=np.uint8)
            for ch in range(3):
                norm = np.clip((lin - dmin[ch]) / max(1e-6, dmax[ch] - dmin[ch]), 0.0, 1.0)
                vals = np.interp(norm * (size_1d - 1), np.arange(size_1d), arr[:, ch])
                out_1d[:, 2 - ch] = np.clip(vals * 255.0 + 0.5, 0, 255).astype(np.uint8)
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
            out = np.empty_like(bgr_img)
            out[:, :, 0] = self.lut_1d[bgr_img[:, :, 0], 0]
            out[:, :, 1] = self.lut_1d[bgr_img[:, :, 1], 1]
            out[:, :, 2] = self.lut_1d[bgr_img[:, :, 2], 2]
        else:
            out = self.baked_3d[bgr_img[:, :, 0] >> 1, bgr_img[:, :, 1] >> 1, bgr_img[:, :, 2] >> 1]
        if strength < 0.99:
            return cv2.addWeighted(out, strength, bgr_img, 1.0 - strength, 0)
        return out


# ==========================================
# Continuous Audio Ring Buffer & Recorder
# ==========================================
class AudioEngine:
    """Captures continuous microphone audio for Live Photo pre/post buffer and Video recording."""

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
        now = time.time()
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
        now = time.time()
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


def mux_video_audio_ffmpeg(temp_video, temp_wav, final_mp4, pts_scale=1.0):
    """Muxes OpenCV video + WAV audio into standard H.264/AAC MP4 (required by Android Motion Photo)."""
    if not FFMPEG_EXE or not os.path.exists(temp_video):
        if os.path.exists(temp_video):
            shutil.move(temp_video, final_mp4)
        return False

    cmd = [FFMPEG_EXE, "-y", "-i", temp_video]
    has_audio = temp_wav is not None and os.path.exists(temp_wav)
    if has_audio:
        cmd += ["-i", temp_wav]

    if abs(pts_scale - 1.0) > 0.02:
        cmd += ["-filter:v", f"setpts={pts_scale:.5f}*PTS"]

    # H.264 baseline/main yuv420p + AAC + faststart for universal Android/Google Photos playback
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

    if not ok and os.path.exists(temp_video):
        shutil.move(temp_video, final_mp4)
    else:
        for p in (temp_video, temp_wav):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    return ok


# ==========================================
# Image Processing Pipeline
# ==========================================
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


def apply_image_pipeline(frame, settings, active_lut):
    img = frame

    if settings["mirror"]:
        img = cv2.flip(img, 1)

    zoom = settings["zoom"]
    if zoom > 1.01:
        h, w = img.shape[:2]
        nw, nh = int(w / zoom), int(h / zoom)
        x1, y1 = (w - nw) // 2, (h - nh) // 2
        img = cv2.resize(img[y1:y1 + nh, x1:x1 + nw], (w, h), interpolation=cv2.INTER_LINEAR)

    if settings["use_sw_wb"]:
        gains = kelvin_to_bgr_gains(settings["wb_kelvin"])
        img = np.clip(img.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)

    alpha = settings["contrast"] * settings["sw_iso_mult"]
    if settings["brightness"] != 0 or abs(alpha - 1.0) > 0.01:
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=settings["brightness"])

    sat = settings["saturation"]
    if abs(sat - 1.0) > 0.01:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    sharp = settings["sharpness"]
    if sharp > 0.01:
        blur = cv2.GaussianBlur(img, (0, 0), 1.5)
        img = cv2.addWeighted(img, 1.0 + sharp, blur, -sharp, 0)

    flt = settings["filter"]
    if flt == "Vivid":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        img = cv2.convertScaleAbs(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), alpha=1.1, beta=0)
    elif flt == "Leica B&W":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        img = cv2.cvtColor(cv2.convertScaleAbs(clahe.apply(gray), alpha=1.18, beta=-12), cv2.COLOR_GRAY2BGR)
    elif flt == "Warm Film":
        b, g, r = cv2.split(img)
        r = np.clip(r.astype(np.int16) + 18, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.int16) + 6, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.int16) - 14, 0, 255).astype(np.uint8)
        img = np.clip(cv2.merge((b, g, r)).astype(np.float32) * 0.94 + 12, 0, 255).astype(np.uint8)
    elif flt == "Cool Cinema":
        b, g, r = cv2.split(img)
        b = np.clip(b.astype(np.int16) + 18, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        r = np.clip(r.astype(np.int16) - 8, 0, 255).astype(np.uint8)
        img = cv2.convertScaleAbs(cv2.merge((b, g, r)), alpha=1.12, beta=-6)
    elif flt == "Vintage Sepia":
        kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        img = np.clip(cv2.transform(img, kernel), 0, 255).astype(np.uint8)

    if active_lut is not None and settings["lut_strength"] > 0.01:
        img = active_lut.apply(img, settings["lut_strength"])

    if settings["watermark"]:
        h, w = img.shape[:2]
        lut_tag = f"  |  LUT: {active_lut.name}" if active_lut else ""
        text = f"PROCAM WIN PRO  |  {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}{lut_tag}"
        scale = max(0.45, w / 1920.0 * 0.7)
        thick = max(1, int(scale * 2))
        pos = (24, h - 22)
        cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), thick, cv2.LINE_AA)

    return img


# ==========================================
# Non-Destructive Pro Viewfinder Overlays
# ==========================================
def draw_rgb_histogram(vis):
    h, w = vis.shape[:2]
    hw, hh = 196, 86
    x0, y0 = w - hw - 16, 16
    if x0 < 0 or y0 + hh > h:
        return

    roi = vis[y0:y0 + hh, x0:x0 + hw]
    vis[y0:y0 + hh, x0:x0 + hw] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    cv2.rectangle(vis, (x0, y0), (x0 + hw, y0 + hh), (80, 80, 95), 1)

    small = cv2.resize(vis, (240, 135), interpolation=cv2.INTER_NEAREST)
    bins = 64
    for ch, col in enumerate([(255, 110, 70), (80, 235, 110), (75, 85, 255)]):
        hist = cv2.calcHist([small], [ch], None, [bins], [0, 256]).flatten()
        if hist.max() > 0:
            hist = (hist / hist.max()) * (hh - 12)
        pts = np.zeros((bins, 1, 2), dtype=np.int32)
        for i in range(bins):
            pts[i, 0, 0] = x0 + 4 + int(i * (hw - 8) / (bins - 1))
            pts[i, 0, 1] = y0 + hh - 4 - int(hist[i])
        cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)


def draw_preview_overlays(frame, settings, countdown_val, flash_alpha, live_capturing, rec_text, hud_text):
    vis = frame.copy()
    h, w = vis.shape[:2]

    if settings["focus_peaking"]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
        vis[lap > 38] = (60, 255, 90)

    if settings["zebra"]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yy, xx = np.indices((h, w))
        vis[(gray >= 245) & (((xx + yy) // 5) % 2 == 0)] = (50, 60, 255)

    if settings["grid"]:
        for i in (1, 2):
            cv2.line(vis, (w * i // 3, 0), (w * i // 3, h), (210, 210, 210), 1, cv2.LINE_AA)
            cv2.line(vis, (0, h * i // 3), (w, h * i // 3), (210, 210, 210), 1, cv2.LINE_AA)

    if settings["histogram"]:
        draw_rgb_histogram(vis)

    if live_capturing:
        cv2.rectangle(vis, (w // 2 - 78, 16), (w // 2 + 78, 50), (0, 184, 245), -1)
        cv2.putText(vis, "MVIMG ● REC", (w // 2 - 66, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 15, 18), 2, cv2.LINE_AA)

    if rec_text:
        cv2.rectangle(vis, (16, 16), (250, 52), (20, 20, 26), -1)
        cv2.circle(vis, (36, 34), 7, (48, 59, 255), -1)
        cv2.putText(vis, rec_text, (52, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    if hud_text:
        bar_h = 32
        roi = vis[h - bar_h:h, 0:w]
        vis[h - bar_h:h, 0:w] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
        cv2.putText(vis, hud_text, (16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 245), 1, cv2.LINE_AA)

    if countdown_val > 0:
        txt = str(countdown_val)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 10)
        pos = ((w - tw) // 2, (h + th) // 2)
        cv2.putText(vis, txt, (pos[0] + 4, pos[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), 12, cv2.LINE_AA)
        cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 10, cv2.LINE_AA)

    if flash_alpha > 0.0:
        vis = cv2.addWeighted(np.full_like(vis, 255), flash_alpha, vis, 1.0 - flash_alpha, 0)

    return vis


def bgr_to_qpixmap(bgr_img, target_size=None):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if target_size is not None:
        pix = pix.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pix


# ==========================================
# Interactive Live Photo / MVIMG Preview Widget
# ==========================================
class LivePhotoLabel(QLabel):
    """Displays still JPG; press-and-hold plays paired MP4 or embedded MVIMG video WITH audio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0d0d10; border-radius: 10px; border: 1px solid #2a2a33;")
        self.still_path = None
        self.video_path = None
        self.cap = None
        self.playing = False
        self.audio_pcm = None
        self.audio_sr = 44100

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._next_frame)

    def load_media(self, path, playable_video=None):
        self.stop_playback()
        self.still_path = path
        self.video_path = playable_video
        self.audio_pcm = None
        self._show_still()

        target_vid = playable_video or (path if path and path.lower().endswith(".mp4") else None)
        if target_vid and HAS_AUDIO and FFMPEG_EXE and os.path.exists(target_vid):
            threading.Thread(target=self._preload_audio, args=(target_vid,), daemon=True).start()

    def _preload_audio(self, vid_path):
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            cmd = [FFMPEG_EXE, "-i", vid_path, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(self.audio_sr), "pipe:1"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=creationflags)
            if proc.returncode == 0 and proc.stdout:
                pcm = np.frombuffer(proc.stdout, dtype=np.int16)
                if vid_path in (self.video_path, self.still_path):
                    self.audio_pcm = pcm
        except Exception:
            pass

    def _show_still(self):
        if self.still_path and os.path.exists(self.still_path):
            if self.still_path.lower().endswith(".mp4"):
                cap = cv2.VideoCapture(self.still_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.setPixmap(bgr_to_qpixmap(frame, self.size()))
                else:
                    self.setText("Video Preview Unavailable")
            else:
                img = cv2.imread(self.still_path)
                if img is not None:
                    self.setPixmap(bgr_to_qpixmap(img, self.size()))
        else:
            self.setText("No Media Selected")

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
        self.cap = cv2.VideoCapture(target_vid)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 5 or fps > 60:
            fps = 24.0
        self.playing = True
        self._play_audio_track()
        self.play_timer.start(int(1000 / fps))

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

    def _next_frame(self):
        if not self.cap or not self.playing:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._play_audio_track()
            ret, frame = self.cap.read()
            if not ret:
                self.stop_playback()
                return
        badge = "LIVE 🔊" if self.audio_pcm is not None else "LIVE"
        cv2.rectangle(frame, (16, 16), (138, 46), (0, 184, 245), -1)
        cv2.putText(frame, badge, (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (15, 15, 18), 2, cv2.LINE_AA)
        self.setPixmap(bgr_to_qpixmap(frame, self.size()))

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
# Gallery & MVIMG Motion Photo Viewer Dialog
# ==========================================
class GalleryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProCam Gallery — MVIMG Motion Photo & Audio Viewer")
        self.resize(1060, 660)
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
        self.file_list.setFixedWidth(315)
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

        self.viewer = LivePhotoLabel()
        self.viewer.setMinimumSize(640, 440)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_box.addWidget(self.viewer)

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
                res = extract_embedded_mp4(primary, tmp_mp4)
                self.extracted_cache[primary] = res
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
        self.viewer.load_media(primary, playable)

        if tag == "MVIMG":
            self.hint_label.setText("ANDROID MOTION PHOTO (Single File) — Press & Hold for motion + audio (Google Photos / Xiaomi Ready)")
            self.btn_play.setEnabled(True)
            self.btn_extract.setEnabled(True)
            self.btn_gif.setEnabled(True)
        elif tag == "LIVE":
            self.hint_label.setText("LIVE PHOTO (Paired) — Press & Hold on image for motion + audio")
            self.btn_play.setEnabled(True)
            self.btn_extract.setEnabled(False)
            self.btn_gif.setEnabled(True)
        elif tag in ("VIDEO", "TIMELAPSE"):
            self.hint_label.setText(f"{tag} — Click Play or Hold image to watch")
            self.btn_play.setEnabled(True)
            self.btn_extract.setEnabled(False)
            self.btn_gif.setEnabled(True)
        else:
            self.hint_label.setText(f"{tag} — Still Capture")
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
        res = extract_embedded_mp4(primary, out_mp4)
        if res:
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
# Main ProCam Windows Application
# ==========================================
class ProCameraWindow(QMainWindow):
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProCam Windows v2.1 — MVIMG Motion Photo, Custom .cube LUT & Pro Mode")
        self.resize(1340, 810)

        self.cap = None
        self.camera_index = 0
        self.current_res = (1280, 720)
        self.mode = "PRO"
        self.live_photo_enabled = True
        self.mvimg_single_file = True  # Saves single-file MVIMG_*.jpg by default
        self.mic_enabled = True

        self.audio_engine = AudioEngine()
        if HAS_AUDIO:
            ok, _ = self.audio_engine.start()
            self.mic_enabled = ok
        else:
            self.mic_enabled = False

        # Live Photo / MVIMG buffer state
        self.live_buffer = deque(maxlen=90)
        self.live_post_capturing = False
        self.live_pre_frames = []
        self.live_post_frames = []
        self.live_still_frame = None
        self.live_shutter_ts = 0.0
        self.live_end_time = 0.0
        self.live_base_path = ""

        # LUT state
        self.lut_paths = {}
        self.active_lut = None

        # Recording / Burst / Timer states
        self.is_recording = False
        self.video_writer = None
        self.temp_video_path = ""
        self.active_video_path = ""
        self.rec_start_time = 0.0
        self.rec_frame_count = 0
        self.timelapse_interval = 1.0
        self.timelapse_last_grab = 0.0

        self.burst_remaining = 0
        self.countdown_val = 0
        self.flash_alpha = 0.0
        self.last_clean_frame = None

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
            "auto_exposure": True,
            "exposure_ev": -6,
            "iso_val": 400,
            "sw_iso_mult": 1.0,
            "auto_focus": True,
            "focus_val": 50,
            "auto_wb": True,
            "wb_kelvin": 5500,
            "use_sw_wb": False,
            "histogram": True,
            "focus_peaking": False,
            "zebra": False,
            "save_png": False,
        }

        self._build_ui()
        self._apply_styles()
        self._scan_luts_folder()

        self.status_signal.connect(self.set_status)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)

        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._countdown_tick)

        self.burst_timer = QTimer(self)
        self.burst_timer.timeout.connect(self._burst_tick)

        self.open_camera(0)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 10, 14, 10)
        root_layout.setSpacing(10)

        # --- Top Bar ---
        top_bar = QHBoxLayout()

        self.btn_live = QPushButton("◉ LIVE ON")
        self.btn_live.setCheckable(True)
        self.btn_live.setChecked(True)
        self.btn_live.setObjectName("LiveBtn")
        self.btn_live.clicked.connect(self.toggle_live_photo)
        top_bar.addWidget(self.btn_live)

        self.btn_mvimg = QPushButton("MVIMG: ON")
        self.btn_mvimg.setCheckable(True)
        self.btn_mvimg.setChecked(True)
        self.btn_mvimg.setToolTip("ON: Single-file MVIMG_*.jpg (Google Photos & Xiaomi Gallery)\nOFF: Separate JPG + MP4")
        self.btn_mvimg.clicked.connect(self.toggle_mvimg_format)
        top_bar.addWidget(self.btn_mvimg)

        self.btn_mic = QPushButton("🎙 Mic: ON" if self.mic_enabled else "🔇 Mic: OFF")
        self.btn_mic.setCheckable(True)
        self.btn_mic.setChecked(self.mic_enabled)
        self.btn_mic.clicked.connect(self.toggle_mic)
        top_bar.addWidget(self.btn_mic)

        self.btn_grid = QPushButton("Grid: ON")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(True)
        self.btn_grid.clicked.connect(self.toggle_grid)
        top_bar.addWidget(self.btn_grid)

        self.btn_mirror = QPushButton("Mirror: ON")
        self.btn_mirror.setCheckable(True)
        self.btn_mirror.setChecked(True)
        self.btn_mirror.clicked.connect(self.toggle_mirror)
        top_bar.addWidget(self.btn_mirror)

        self.btn_watermark = QPushButton("Watermark")
        self.btn_watermark.setCheckable(True)
        self.btn_watermark.clicked.connect(lambda v: self.settings.__setitem__("watermark", v))
        top_bar.addWidget(self.btn_watermark)

        top_bar.addWidget(QLabel("Timer:"))
        self.combo_timer = QComboBox()
        self.combo_timer.addItems(["Off", "3s", "5s", "10s"])
        self.combo_timer.currentIndexChanged.connect(lambda i: self.settings.__setitem__("timer", [0, 3, 5, 10][i]))
        top_bar.addWidget(self.combo_timer)

        top_bar.addStretch()

        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.combo_cam.currentIndexChanged.connect(self.open_camera)
        top_bar.addWidget(self.combo_cam)

        self.combo_res = QComboBox()
        self.combo_res.addItems(["1280x720 (HD)", "1920x1080 (FHD)"])
        self.combo_res.currentIndexChanged.connect(self.change_resolution)
        top_bar.addWidget(self.combo_res)

        root_layout.addLayout(top_bar)

        # --- Middle: Preview + Right Tabbed Panel ---
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(14)

        self.preview_label = QLabel("Starting Windows Camera...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(860, 540)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setObjectName("PreviewBox")
        mid_layout.addWidget(self.preview_label, stretch=3)

        self.right_tabs = QTabWidget()
        self.right_tabs.setFixedWidth(330)

        # TAB 1: PRO MANUAL
        pro_tab = QWidget()
        pro_l = QVBoxLayout(pro_tab)
        pro_l.setSpacing(8)

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Shutter / Exposure"))
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
        self.slider_exp.setRange(-13, -1)
        self.slider_exp.setValue(-6)
        self.slider_exp.setEnabled(False)
        self.slider_exp.valueChanged.connect(self.on_exposure_changed)
        pro_l.addWidget(self.slider_exp)

        iso_row = QHBoxLayout()
        iso_row.addWidget(QLabel("ISO / Gain"))
        self.lbl_iso_val = QLabel("ISO 400")
        self.lbl_iso_val.setStyleSheet("color: #f5b800; font-weight: bold;")
        iso_row.addStretch()
        iso_row.addWidget(self.lbl_iso_val)
        pro_l.addLayout(iso_row)

        self.slider_iso = QSlider(Qt.Orientation.Horizontal)
        self.slider_iso.setRange(100, 3200)
        self.slider_iso.setSingleStep(100)
        self.slider_iso.setValue(400)
        self.slider_iso.valueChanged.connect(self.on_iso_changed)
        pro_l.addWidget(self.slider_iso)

        foc_row = QHBoxLayout()
        foc_row.addWidget(QLabel("Focus"))
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
        self.slider_foc.setEnabled(False)
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
        self.slider_wb.setEnabled(False)
        self.slider_wb.valueChanged.connect(self.on_wb_changed)
        pro_l.addWidget(self.slider_wb)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color: #2f2f3d;")
        pro_l.addWidget(div)

        pro_l.addWidget(QLabel("PRO VIEWFINDER TOOLS"))

        self.chk_hist = QCheckBox("Live RGB Histogram")
        self.chk_hist.setChecked(True)
        self.chk_hist.toggled.connect(lambda v: self.settings.__setitem__("histogram", v))
        pro_l.addWidget(self.chk_hist)

        self.chk_peaking = QCheckBox("Focus Peaking (Sharp Edge Highlight)")
        self.chk_peaking.toggled.connect(lambda v: self.settings.__setitem__("focus_peaking", v))
        pro_l.addWidget(self.chk_peaking)

        self.chk_zebra = QCheckBox("Zebra Stripes (Overexposure >95%)")
        self.chk_zebra.toggled.connect(lambda v: self.settings.__setitem__("zebra", v))
        pro_l.addWidget(self.chk_zebra)

        self.chk_png = QCheckBox("Also Save Lossless PNG (PRO Mode)")
        self.chk_png.toggled.connect(lambda v: self.settings.__setitem__("save_png", v))
        pro_l.addWidget(self.chk_png)

        btn_reset_pro = QPushButton("Reset All to Auto")
        btn_reset_pro.clicked.connect(self.reset_hardware_pro)
        pro_l.addWidget(btn_reset_pro)

        pro_l.addStretch()
        self.right_tabs.addTab(pro_tab, "PRO MANUAL")

        # TAB 2: CUSTOM .CUBE LUT & COLOR
        lut_tab = QWidget()
        lut_l = QVBoxLayout(lut_tab)
        lut_l.setSpacing(8)

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

        self.sliders = {}
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
        self._add_slider(lut_l, "Sharpness", "sharpness", 0, 150, 0, lambda v: f"{v}%", lambda v: v / 100.0)

        btn_reset_color = QPushButton("Reset Color & LUT")
        btn_reset_color.clicked.connect(self.reset_color_settings)
        lut_l.addWidget(btn_reset_color)

        lut_l.addStretch()
        self.right_tabs.addTab(lut_tab, "LUT & COLOR")

        mid_layout.addWidget(self.right_tabs, stretch=1)
        root_layout.addLayout(mid_layout)

        # --- Bottom Bar ---
        bottom_bar = QHBoxLayout()

        left_bottom = QVBoxLayout()
        self.btn_gallery = QPushButton("🖼 Gallery & MVIMG Viewer")
        self.btn_gallery.setFixedHeight(44)
        self.btn_gallery.clicked.connect(self.open_gallery)
        left_bottom.addWidget(self.btn_gallery)

        self.status_label = QLabel("Ready — Android MVIMG Motion Photo (Audio + Xiaomi/Google Ready)")
        self.status_label.setStyleSheet("color: #a6a7bd; font-size: 12px;")
        left_bottom.addWidget(self.status_label)
        bottom_bar.addLayout(left_bottom)

        bottom_bar.addStretch()

        center_bottom = QVBoxLayout()
        center_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mode_row = QHBoxLayout()
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
        self.btn_shutter.setFixedSize(68, 68)
        self.btn_shutter.setObjectName("ShutterBtn")
        self.btn_shutter.clicked.connect(self.trigger_shutter)
        shutter_row.addWidget(self.btn_shutter)
        center_bottom.addLayout(shutter_row)

        bottom_bar.addLayout(center_bottom)
        bottom_bar.addStretch()

        right_bottom = QVBoxLayout()
        btn_folder = QPushButton("📁 Open Captures")
        btn_folder.setFixedHeight(44)
        btn_folder.clicked.connect(lambda: os.startfile(BASE_DIR))
        right_bottom.addWidget(btn_folder)

        hint = QLabel("Space: Shoot | L: Live | P: Peaking | H: Hist")
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
            QMainWindow, QWidget { background-color: #111116; color: #f0f0f6; font-family: 'Segoe UI'; }
            QTabWidget::pane { background-color: #191921; border: 1px solid #2b2b36; border-radius: 10px; padding: 8px; }
            QTabBar::tab {
                background-color: #1c1c25; color: #9a9bb2; padding: 8px 16px;
                border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; margin-right: 4px;
            }
            QTabBar::tab:selected { background-color: #282836; color: #f5b800; }
            QLabel#PreviewBox { background-color: #08080b; border: 1px solid #282834; border-radius: 12px; }
            QPushButton {
                background-color: #23232e; color: #f0f0f6; border: 1px solid #343444;
                border-radius: 8px; padding: 7px 12px; font-weight: 600;
            }
            QPushButton:hover { background-color: #2f2f3d; }
            QPushButton:checked { background-color: #2d5bff; border-color: #557bff; color: white; }
            QPushButton#LiveBtn:checked { background-color: #f5b800; border-color: #ffd042; color: #121216; }
            QPushButton#ModeBtn {
                background-color: transparent; border: none; color: #8f90a6;
                padding: 4px 12px; font-size: 12px; letter-spacing: 1px;
            }
            QPushButton#ModeBtn:checked {
                background-color: transparent; color: #f5b800; border-bottom: 2px solid #f5b800; border-radius: 0px;
            }
            QPushButton#ShutterBtn {
                background-color: #f4f4f8; color: #15151a; border: 4px solid #6a6a7c; border-radius: 34px; font-size: 26px;
            }
            QPushButton#ShutterBtn:hover { background-color: #ffffff; border-color: #f5b800; }
            QComboBox { background-color: #23232e; border: 1px solid #353545; border-radius: 6px; padding: 5px 10px; color: #f0f0f6; }
            QSlider::groove:horizontal { height: 5px; background: #2c2c38; border-radius: 2px; }
            QSlider::handle:horizontal { background: #f5b800; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
        """)

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
    # PRO Mode Hardware Controls
    # ==========================================
    def on_auto_exposure_toggled(self, auto_on):
        self.settings["auto_exposure"] = auto_on
        self.slider_exp.setEnabled(not auto_on)
        if self.cap is None:
            return
        if auto_on:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.lbl_exp_val.setText("AUTO")
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.on_exposure_changed(self.slider_exp.value())

    def on_exposure_changed(self, ev):
        self.settings["exposure_ev"] = ev
        self.lbl_exp_val.setText(f"1/{int(2 ** abs(ev))}s (EV {ev})")
        if self.cap is not None and not self.settings["auto_exposure"]:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(ev))

    def on_iso_changed(self, iso):
        self.settings["iso_val"] = iso
        self.lbl_iso_val.setText(f"ISO {iso}")
        hw_ok = False
        if self.cap is not None:
            hw_ok = self.cap.set(cv2.CAP_PROP_GAIN, float(int((iso - 100) / 3100.0 * 100.0)))
        self.settings["sw_iso_mult"] = 1.0 if hw_ok else max(0.65, min(2.2, (iso / 400.0) ** 0.45))

    def on_auto_focus_toggled(self, auto_on):
        self.settings["auto_focus"] = auto_on
        self.slider_foc.setEnabled(not auto_on)
        if self.cap is None:
            return
        if auto_on:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.lbl_foc_val.setText("AF")
        else:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.on_focus_changed(self.slider_foc.value())

    def on_focus_changed(self, val):
        self.settings["focus_val"] = val
        self.lbl_foc_val.setText(f"MF {val}")
        if self.cap is not None and not self.settings["auto_focus"]:
            self.cap.set(cv2.CAP_PROP_FOCUS, float(val))

    def on_auto_wb_toggled(self, auto_on):
        self.settings["auto_wb"] = auto_on
        self.slider_wb.setEnabled(not auto_on)
        if auto_on:
            self.settings["use_sw_wb"] = False
            self.lbl_wb_val.setText("AWB")
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        else:
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.on_wb_changed(self.slider_wb.value())

    def on_wb_changed(self, kelvin):
        self.settings["wb_kelvin"] = kelvin
        self.lbl_wb_val.setText(f"{kelvin}K")
        hw_ok = False
        if self.cap is not None and not self.settings["auto_wb"]:
            hw_ok = self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(kelvin))
        self.settings["use_sw_wb"] = (not self.settings["auto_wb"]) and (not hw_ok)

    def reset_hardware_pro(self):
        self.chk_auto_exp.setChecked(True)
        self.chk_auto_foc.setChecked(True)
        self.chk_auto_wb.setChecked(True)
        self.slider_iso.setValue(400)
        self.settings["sw_iso_mult"] = 1.0
        self.set_status("Pro Manual settings reset to Auto")

    def reset_color_settings(self):
        self.clear_lut()
        self.combo_filter.setCurrentText("Normal")
        for key, (slider, default_v) in self.sliders.items():
            slider.setValue(default_v)
        self.set_status("Color & LUT settings reset")

    # ==========================================
    # Camera & Toggles
    # ==========================================
    def open_camera(self, index):
        self.frame_timer.stop()
        if self.is_recording:
            self.stop_recording()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera_index = index
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self.preview_label.setText(f"Camera {index} not detected.\nConnect a webcam or DroidCam/Camo.")
            self.set_status(f"Camera {index} unavailable")
            return

        w, h = self.current_res
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.cap = cap
        self.live_buffer.clear()
        aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.set_status(f"Camera {index} Ready ({aw}x{ah}) | MVIMG Format: {'Single File' if self.mvimg_single_file else 'Paired'}")
        self.frame_timer.start(30)

    def change_resolution(self, idx):
        self.current_res = (1280, 720) if idx == 0 else (1920, 1080)
        self.open_camera(self.camera_index)

    def toggle_live_photo(self, checked):
        self.live_photo_enabled = checked
        self.btn_live.setText("◉ LIVE ON" if checked else "○ LIVE OFF")
        if not checked:
            self.live_buffer.clear()

    def toggle_mvimg_format(self, checked):
        self.mvimg_single_file = checked
        self.btn_mvimg.setText("MVIMG: ON" if checked else "MVIMG: OFF")
        self.set_status(
            "Output: Single-file MVIMG_*.jpg (Google Photos & Xiaomi Gallery)"
            if checked else "Output: Paired LIVE_*.jpg + LIVE_*.mp4"
        )

    def toggle_mic(self, checked):
        if checked:
            ok, msg = self.audio_engine.start()
            self.mic_enabled = ok
            self.btn_mic.setChecked(ok)
            self.btn_mic.setText("🎙 Mic: ON" if ok else "🔇 Mic: OFF")
            self.set_status(msg)
        else:
            self.audio_engine.stop()
            self.mic_enabled = False
            self.btn_mic.setText("🔇 Mic: OFF")
            self.set_status("Microphone muted")

    def toggle_grid(self, checked):
        self.settings["grid"] = checked
        self.btn_grid.setText("Grid: ON" if checked else "Grid: OFF")

    def toggle_mirror(self, checked):
        self.settings["mirror"] = checked
        self.btn_mirror.setText("Mirror: ON" if checked else "Mirror: OFF")

    def set_mode(self, mode):
        if self.is_recording:
            self.stop_recording()
        self.mode = mode
        if mode == "PRO":
            self.right_tabs.setCurrentIndex(0)
        if mode in ("VIDEO", "TIMELAPSE"):
            self.btn_shutter.setStyleSheet(
                "background-color: #ff3b30; color: white; border: 4px solid #6a6a7c; border-radius: 34px; font-size: 24px;"
            )
        else:
            self.btn_shutter.setStyleSheet("")
        self.btn_shutter.setText("●")
        self.set_status(f"Mode: {mode}")

    def set_status(self, text):
        self.status_label.setText(text)

    # ==========================================
    # Main Frame Loop
    # ==========================================
    def _build_hud_text(self):
        if self.mode != "PRO" and self.active_lut is None:
            return ""
        ss = "AUTO" if self.settings["auto_exposure"] else f"1/{int(2**abs(self.settings['exposure_ev']))}s"
        iso = f"ISO {self.settings['iso_val']}"
        foc = "AF" if self.settings["auto_focus"] else f"MF {self.settings['focus_val']}"
        wb = "AWB" if self.settings["auto_wb"] else f"{self.settings['wb_kelvin']}K"
        lut = f"LUT: {self.active_lut.name} ({int(self.settings['lut_strength']*100)}%)" if self.active_lut else "LUT: OFF"
        fmt = "MVIMG" if self.mvimg_single_file else "JPG+MP4"
        return f"  {self.mode}  |  SS {ss}  |  {iso}  |  {foc}  |  WB {wb}  |  {lut}  |  {fmt}"

    def update_frame(self):
        if self.cap is None:
            return
        ret, raw = self.cap.read()
        if not ret or raw is None:
            return

        now = time.time()
        clean_frame = apply_image_pipeline(raw, self.settings, self.active_lut)
        self.last_clean_frame = clean_frame

        if self.mode in ("PHOTO", "PRO") and self.live_photo_enabled:
            self.live_buffer.append((now, clean_frame))

        if self.live_post_capturing:
            self.live_post_frames.append((now, clean_frame.copy()))
            if now >= self.live_end_time:
                self.live_post_capturing = False
                combined_frames = self.live_pre_frames + self.live_post_frames
                audio_np = self.audio_engine.finish_live_capture() if self.mic_enabled else None
                threading.Thread(
                    target=self._encode_live_av,
                    args=(
                        combined_frames,
                        audio_np,
                        self.live_still_frame,
                        self.live_shutter_ts,
                        self.live_base_path,
                        self.mvimg_single_file
                    ),
                    daemon=True
                ).start()

        rec_text = ""
        if self.is_recording and self.video_writer is not None:
            elapsed = int(now - self.rec_start_time)
            mm, ss = divmod(elapsed, 60)
            if self.mode == "VIDEO":
                self.video_writer.write(clean_frame)
                self.rec_frame_count += 1
                rec_text = f"REC  {mm:02d}:{ss:02d}"
            elif self.mode == "TIMELAPSE":
                if now - self.timelapse_last_grab >= self.timelapse_interval:
                    self.video_writer.write(clean_frame)
                    self.timelapse_last_grab = now
                    self.rec_frame_count += 1
                rec_text = f"TL {mm:02d}:{ss:02d} ({self.rec_frame_count}f)"

        if self.flash_alpha > 0.0:
            self.flash_alpha = max(0.0, self.flash_alpha - 0.16)

        vis = draw_preview_overlays(
            clean_frame, self.settings, self.countdown_val,
            self.flash_alpha, self.live_post_capturing, rec_text, self._build_hud_text()
        )
        self.preview_label.setPixmap(bgr_to_qpixmap(vis, self.preview_label.size()))

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

    def capture_photo(self, is_live=False, suffix=""):
        if self.last_clean_frame is None:
            return
        frame = self.last_clean_frame.copy()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.flash_alpha = 0.75

        if is_live:
            if self.mode == "PRO" and self.settings["save_png"]:
                cv2.imwrite(os.path.join(DIRS["photos"], f"PRO_{ts}.png"), frame)

            if not self.live_post_capturing:
                now = time.time()
                prefix = "MVIMG_" if self.mvimg_single_file else "LIVE_"
                self.live_base_path = os.path.join(DIRS["live"], f"{prefix}{ts}")
                self.live_still_frame = frame
                self.live_shutter_ts = now

                if not self.mvimg_single_file:
                    cv2.imwrite(self.live_base_path + ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 96])

                self.live_pre_frames = [
                    (t, f.copy()) for (t, f) in list(self.live_buffer) if t >= now - 1.5
                ]
                self.live_post_frames = [(now, frame)]
                self.live_end_time = now + 1.5
                if self.mic_enabled:
                    self.audio_engine.begin_live_capture(pre_seconds=1.5)
                self.live_post_capturing = True
                self.set_status("Capturing MVIMG Motion Photo + Audio (1.5s)...")
        else:
            if self.mode == "PRO" and self.settings["save_png"]:
                out_path = os.path.join(DIRS["photos"], f"PRO_{ts}{suffix}.png")
                cv2.imwrite(out_path, frame)
            else:
                out_path = os.path.join(DIRS["photos"], f"IMG_{ts}{suffix}.jpg")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
            self.set_status(f"Saved: {os.path.basename(out_path)}")

    def _encode_live_av(self, frames_with_ts, audio_np, still_frame, shutter_ts, base_path, as_mvimg):
        if len(frames_with_ts) < 2:
            return
        first_ts = frames_with_ts[0][0]
        duration = max(0.5, frames_with_ts[-1][0] - first_ts)
        fps = max(10.0, min(30.0, len(frames_with_ts) / duration))
        keyframe_us = max(0, int((shutter_ts - first_ts) * 1_000_000))
        h, w = frames_with_ts[0][1].shape[:2]

        temp_vid = tempfile.mktemp(suffix="_live_raw.mp4")
        temp_wav = None
        muxed_mp4 = tempfile.mktemp(suffix="_live_mux.mp4") if as_mvimg else (base_path + ".mp4")

        writer = cv2.VideoWriter(temp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _, f in frames_with_ts:
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h))
            writer.write(f)
        writer.release()

        if audio_np is not None and len(audio_np) > 0:
            temp_wav = tempfile.mktemp(suffix="_live.wav")
            self.audio_engine.save_wav(audio_np, temp_wav)

        mux_video_audio_ffmpeg(temp_vid, temp_wav, muxed_mp4, pts_scale=1.0)
        tag = "Audio" if temp_wav else "Silent"

        if as_mvimg:
            final_mvimg = base_path + ".jpg"
            try:
                pack_android_motion_photo(still_frame, muxed_mp4, final_mvimg, keyframe_us=keyframe_us)
                self.status_signal.emit(f"Saved Android MVIMG ({tag}): {os.path.basename(final_mvimg)}")
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
        h, w = self.last_clean_frame.shape[:2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.rec_frame_count = 0
        self.rec_start_time = time.time()

        if self.mode == "VIDEO":
            self.active_video_path = os.path.join(DIRS["videos"], f"VID_{ts}.mp4")
            self.temp_video_path = tempfile.mktemp(suffix="_vid.mp4")
            self.video_writer = cv2.VideoWriter(self.temp_video_path, fourcc, 24.0, (w, h))
            if self.mic_enabled:
                self.audio_engine.start_video_recording()
        else:
            self.active_video_path = os.path.join(DIRS["timelapse"], f"TL_{ts}.mp4")
            self.temp_video_path = ""
            self.video_writer = cv2.VideoWriter(self.active_video_path, fourcc, 24.0, (w, h))
            self.timelapse_last_grab = 0.0

        self.is_recording = True
        self.btn_shutter.setText("■")
        self.set_status(f"Recording {self.mode}...")

    def stop_recording(self):
        self.is_recording = False
        elapsed = max(0.2, time.time() - self.rec_start_time)
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.btn_shutter.setText("●")

        if self.mode == "VIDEO" and self.temp_video_path:
            audio_np = self.audio_engine.stop_video_recording() if self.mic_enabled else None
            pts_scale = (elapsed * 24.0) / max(1, self.rec_frame_count)
            self.set_status("Saving Video & syncing audio...")
            threading.Thread(
                target=self._finalize_video_av,
                args=(self.temp_video_path, audio_np, self.active_video_path, pts_scale),
                daemon=True
            ).start()
        elif self.active_video_path:
            self.set_status(f"Saved {self.mode}: {os.path.basename(self.active_video_path)}")

    def _finalize_video_av(self, temp_vid, audio_np, final_mp4, pts_scale):
        temp_wav = None
        if audio_np is not None and len(audio_np) > 0:
            temp_wav = tempfile.mktemp(suffix="_vid.wav")
            self.audio_engine.save_wav(audio_np, temp_wav)
        mux_video_audio_ffmpeg(temp_vid, temp_wav, final_mp4, pts_scale=pts_scale)
        self.status_signal.emit(f"Saved Video: {os.path.basename(final_mp4)}")

    # ==========================================
    # Gallery & Shortcuts
    # ==========================================
    def open_gallery(self):
        dlg = GalleryDialog(self)
        dlg.exec()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.trigger_shutter()
        elif key == Qt.Key.Key_L:
            self.btn_live.click()
        elif key == Qt.Key.Key_G:
            self.btn_grid.click()
        elif key == Qt.Key.Key_M:
            self.btn_mirror.click()
        elif key == Qt.Key.Key_P:
            self.chk_peaking.setChecked(not self.chk_peaking.isChecked())
        elif key == Qt.Key.Key_H:
            self.chk_hist.setChecked(not self.chk_hist.isChecked())
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            slider, _ = self.sliders["zoom"]
            slider.setValue(min(slider.maximum(), slider.value() + 2))
        elif key == Qt.Key.Key_Minus:
            slider, _ = self.sliders["zoom"]
            slider.setValue(max(slider.minimum(), slider.value() - 2))
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.frame_timer.stop()
        if self.is_recording:
            self.stop_recording()
        self.audio_engine.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ProCameraWindow()
    win.show()
    sys.exit(app.exec())