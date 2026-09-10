#!/usr/bin/env python3
"""
eye_rig.py

Combines face_orientation_tracker.py's camera/CV pipeline and
eye_display.py's shader-ported eye renderer into ONE process. The two
used to talk over OSC (tracker sends, eye_display listens) purely so
eye_display could stay a lightweight, mediapipe-free process on a 4GB
Pi. Running them together trades that RAM separation away on purpose --
tracking and rendering now share memory directly, no network hop, no
LookReceiver, no OSC round-trip latency for the on-screen eyes.

Runs headless by default: the tracking pipeline (face + body/chest
fallback, exactly as in face_orientation_tracker.py) runs with no debug
overlay, and the one window shown is the eyes. Press 'd' to toggle that
SAME window to the tracker's debug view (landmarks/bbox/pose overlay) --
handy for confirming tracking is behaving without juggling two windows/
processes. Press 'q' to quit.

    d  -- toggle eyes / debug view
    q  -- quit

OSC output (for the physical animatronic rig / Max / TouchDesigner / a
PCA9685 receiver -- see the original tracker's docstring) still works
exactly as before via --osc-host/--osc-port, and is independent of
whether a window is even shown (--no-window for true unattended
deployment). python-osc is optional now, not required: the on-screen
eyes no longer depend on it for anything, so a missing install just
disables OSC sending with a printed note instead of exiting.

If you want the old, separate, RAM-light two-process architecture back
(e.g. running the eye visuals on a different, lighter device than the
one doing CV), face_orientation_tracker.py and eye_display.py are both
still here, unchanged and independently runnable.

Run:
    python3 eye_rig.py --camera 0
"""

import argparse
import math
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

try:
    import glfw
    import moderngl
except ImportError:
    glfw = None
    moderngl = None

try:
    from pythonosc.udp_client import SimpleUDPClient
except ImportError:
    SimpleUDPClient = None

try:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_tasks_vision
except ImportError:
    mp_tasks_python = None
    mp_tasks_vision = None


def _default_backend() -> int:
    """Pick the right OpenCV capture backend for the current OS.
    V4L2 is Linux-only; macOS needs AVFoundation. AVFoundation's fps/format
    negotiation with UVC cameras is less predictable than V4L2's -- always
    check the printed 'negotiated' line after opening."""
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class AutoBrighten:
    """Bipolar auto-brightness: each frame, computes the gain that would
    push its mean brightness exactly to target_mean -- brightening a dark
    frame (gain > 1) OR dimming an overexposed one (gain < 1), unlike the
    brighten-only first version of this. The ELP AR0234's global-shutter
    sensor doesn't do meaningful auto-exposure/gain adjustment on its own,
    so rather than fight that at the UVC control level (inconsistently
    exposed across backends), this is a simple, cheap, cross-platform
    stand-in.

    The gain actually APPLIED doesn't jump straight to that ideal value
    though -- it slews toward it at a bounded rate (max_gain_change_per_s)
    instead, same reason a real camera's AGC ramps gain/exposure smoothly
    rather than snapping: applying whatever gain a single frame's mean
    brightness implies would flicker visibly on ordinary per-frame noise
    (a shadow passing through the shot, a moment of motion blur), not
    just on a genuine lighting change. A longer slew trades slower
    response to a real lighting change for a lot less flicker -- default
    here (0.5 gain/second) means swinging the full 0.25-4.0 range takes
    several seconds."""

    def __init__(self, target_mean: float = 120.0, min_gain: float = 0.25,
                 max_gain: float = 4.0, max_gain_change_per_s: float = 0.5):
        self.target_mean = target_mean
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.max_gain_change_per_s = max_gain_change_per_s
        self._gain = 1.0
        self._last_time: Optional[float] = None

    def apply(self, frame_bgr: np.ndarray, now: float) -> np.ndarray:
        mean = float(frame_bgr.mean())
        target_gain = self.max_gain if mean <= 1e-6 else self.target_mean / mean
        target_gain = max(self.min_gain, min(target_gain, self.max_gain))

        dt = 0.0 if self._last_time is None else max(now - self._last_time, 0.0)
        self._last_time = now
        max_step = self.max_gain_change_per_s * dt
        if target_gain > self._gain:
            self._gain = min(self._gain + max_step, target_gain)
        else:
            self._gain = max(self._gain - max_step, target_gain)

        if abs(self._gain - 1.0) < 1e-3:
            return frame_bgr  # ~unity gain -- skip the arithmetic op
        return cv2.convertScaleAbs(frame_bgr, alpha=self._gain, beta=0)


# ==========================================================================
# Camera capture thread -- always-latest-frame, never blocks / never queues
# ==========================================================================
class CameraStream:
    def __init__(self, index: int, width: int, height: int, fps: int,
                 fourcc: str = "MJPG", backend: Optional[int] = None,
                 auto_brighten_enabled: bool = True,
                 target_brightness: float = 120.0, min_gain: float = 0.25,
                 max_gain: float = 4.0, brighten_slew: float = 0.5):
        backend = backend if backend is not None else _default_backend()
        self.cap = cv2.VideoCapture(index, backend)
        # Set FourCC before resolution/fps -- AVFoundation in particular
        # ignores fps/resolution requests made before the pixel format is set.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Ask the driver for the smallest internal buffer it supports so we
        # don't accumulate stale frames if processing falls behind. Not all
        # backends (e.g. AVFoundation) honor this -- harmless if ignored.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[camera] requested {width}x{height}@{fps} -> "
              f"negotiated {actual_w}x{actual_h}@{actual_fps:.1f}")
        if actual_fps < fps * 0.8:
            print("[camera] WARNING: negotiated fps is well below requested. "
                  "Check `v4l2-ctl -d /dev/videoX --list-formats-ext` for "
                  "the exact supported modes.")

        self.auto_brighten_enabled = auto_brighten_enabled
        self._auto_brighten = AutoBrighten(
            target_mean=target_brightness, min_gain=min_gain,
            max_gain=max_gain, max_gain_change_per_s=brighten_slew,
        ) if auto_brighten_enabled else None
        if auto_brighten_enabled:
            print(f"[camera] auto-brighten on (target mean brightness "
                  f"{target_brightness:.0f}, gain {min_gain:.2f}x-{max_gain:.1f}x, "
                  f"slew {brighten_slew:.2f} gain/s)")

        self.width, self.height = actual_w, actual_h
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp = 0.0
        self._frame_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            now = time.monotonic()
            if self._auto_brighten is not None:
                frame = self._auto_brighten.apply(frame, now)
            with self._lock:
                self._frame = frame
                self._timestamp = now
                self._frame_count += 1

    def latest(self):
        """Returns (frame, timestamp) for the most recently captured frame."""
        with self._lock:
            return self._frame, self._timestamp

    def fps_actual(self, window_start_count, window_start_time):
        with self._lock:
            count, t = self._frame_count, self._timestamp
        elapsed = t - window_start_time
        if elapsed <= 0:
            return 0.0
        return (count - window_start_count) / elapsed

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()


# ==========================================================================
# Head pose: 6-point generic 3D face model + solvePnP
# ==========================================================================
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye, outer corner
    (225.0, 170.0, -135.0),   # right eye, outer corner
    (-150.0, -150.0, -125.0), # mouth, left corner
    (150.0, -150.0, -125.0),  # mouth, right corner
], dtype=np.float64)

_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
}

_POSE_LANDMARK_IDX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}


def build_camera_matrix(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Approximate intrinsics from the lens's stated horizontal FOV."""
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


@dataclass
class HeadPoseResult:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    frustum_x: float   # -1 (left edge) .. +1 (right edge)
    frustum_y: float   # -1 (top edge)  .. +1 (bottom edge)
    approx_distance_mm: float
    bbox: tuple
    keypoints: tuple
    nose_line: tuple


class OneEuroFilter:
    """One Euro Filter (Casiez, Roussel, Vogel 2012)."""

    def __init__(self, t0: float, x0: float, dx0: float = 0.0,
                 min_cutoff: float = 0.8, beta: float = 0.07,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    @staticmethod
    def _alpha(t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class OneEuroSmoother:
    _FIELDS = ("yaw_deg", "pitch_deg", "roll_deg",
               "frustum_x", "frustum_y", "approx_distance_mm")

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.07):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, OneEuroFilter] = {}

    def update(self, pose: Optional[HeadPoseResult], t: float) -> Optional[HeadPoseResult]:
        if pose is None:
            self._filters = {}
            return None
        values = {f: getattr(pose, f) for f in self._FIELDS}
        smoothed = {}
        for f, x in values.items():
            filt = self._filters.get(f)
            if filt is None:
                filt = OneEuroFilter(t, x, min_cutoff=self.min_cutoff, beta=self.beta)
                self._filters[f] = filt
                smoothed[f] = x
            else:
                smoothed[f] = filt(t, x)
        return HeadPoseResult(
            yaw_deg=smoothed["yaw_deg"], pitch_deg=smoothed["pitch_deg"],
            roll_deg=smoothed["roll_deg"],
            frustum_x=smoothed["frustum_x"], frustum_y=smoothed["frustum_y"],
            approx_distance_mm=smoothed["approx_distance_mm"],
            bbox=pose.bbox, keypoints=pose.keypoints, nose_line=pose.nose_line,
        )


class HeadPoseEstimator:
    REAL_INTEROCULAR_MM = 63.0

    def __init__(self, width: int, height: int,
                 hfov_deg: float = 112.0, min_face_px: int = 60,
                 max_angular_velocity_deg_s: float = 500.0,
                 pose_hold_s: float = 0.2,
                 face_model_path: str = "face_landmarker.task",
                 face_detection_confidence: float = 0.5,
                 face_presence_confidence: float = 0.5,
                 face_tracking_confidence: float = 0.5):
        self.width, self.height = width, height
        self.camera_matrix = build_camera_matrix(width, height, hfov_deg)
        self.dist_coeffs = np.zeros((4, 1))
        self.min_face_px = min_face_px

        self.max_angular_velocity_deg_s = max_angular_velocity_deg_s
        self.pose_hold_s = pose_hold_s
        self._last_rot_mat: Optional[np.ndarray] = None
        self._last_rvec: Optional[np.ndarray] = None
        self._last_accept_time: Optional[float] = None
        self._reject_since: Optional[float] = None

        # mp.solutions.face_mesh is the known-good path (self-contained
        # model, stable on macOS for years) -- but it's been dropped
        # entirely as of mediapipe 1.0.0, and that's now the ONLY version
        # with wheels on some platforms (confirmed: Raspberry Pi/aarch64
        # only has 1.0.0+ available, no 0.10.x). So: use it when it's
        # actually there, fall back to the Tasks API FaceLandmarker (the
        # same approach BodyPoseEstimator already uses for the body
        # fallback) when it isn't. Both produce the same landmark
        # container shape (indexable, .x/.y/.z), so everything past
        # process()'s first few lines is identical either way.
        #
        # The fallback is a known compromise, not a clean swap: Tasks API
        # FaceLandmarker has previously SIGABRT-crashed on some macOS
        # configs (a GPU "service" registration failure) even with CPU
        # delegate explicitly set -- see the long-form note this replaced
        # for the exact error. That's why solutions was used at all
        # originally. If this fallback path crashes on macOS, that's this
        # known, previously-observed issue resurfacing, not a new bug --
        # it's expected to be reliable on Linux/the Pi, which is what
        # actually needs it.
        self._use_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
        if self._use_solutions:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            if mp_tasks_python is None or mp_tasks_vision is None:
                print("ERROR: mp.solutions.face_mesh is unavailable in this "
                      "mediapipe install, and mediapipe.tasks isn't either -- "
                      "no way to track a face at all.\n"
                      f"  mediapipe version: {getattr(mp, '__version__', 'unknown')}")
                sys.exit(1)
            if not os.path.exists(face_model_path):
                print(f"ERROR: mp.solutions.face_mesh is unavailable in this "
                      f"mediapipe install (dropped as of mediapipe 1.0.0), and "
                      f"the Tasks API fallback needs a FaceLandmarker model at "
                      f"'{face_model_path}', which wasn't found. Download one:\n"
                      f"  curl -L -o {face_model_path} \\\n"
                      f"    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
                sys.exit(1)
            print("[face] mp.solutions.face_mesh unavailable -- using Tasks "
                  "API FaceLandmarker instead (CPU delegate). Known to have "
                  "crashed on some macOS configs previously; expected to be "
                  "fine on Linux/the Pi.")
            base_options = mp_tasks_python.BaseOptions(
                model_asset_path=face_model_path,
                delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
            )
            options = mp_tasks_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_tasks_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=face_detection_confidence,
                min_face_presence_confidence=face_presence_confidence,
                min_tracking_confidence=face_tracking_confidence,
            )
            self._face_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _rotation_angle_diff_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
        rot_rel = rot_a @ rot_b.T
        cos_angle = np.clip((np.trace(rot_rel) - 1.0) / 2.0, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def _reset_pose_hold(self):
        self._last_rot_mat = None
        self._last_rvec = None
        self._last_accept_time = None
        self._reject_since = None

    def process(self, frame_bgr: np.ndarray) -> Optional[HeadPoseResult]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._use_solutions:
            rgb.flags.writeable = False
            result = self._face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                self._reset_pose_hold()
                return None
            lm = result.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._face_landmarker.detect(mp_image)
            if not result.face_landmarks:
                self._reset_pose_hold()
                return None
            lm = result.face_landmarks[0]

        def px(idx):
            p = lm[idx]
            return (p.x * w, p.y * h)

        image_points = np.array([
            px(_LANDMARK_IDX["nose_tip"]),
            px(_LANDMARK_IDX["chin"]),
            px(_LANDMARK_IDX["left_eye_outer"]),
            px(_LANDMARK_IDX["right_eye_outer"]),
            px(_LANDMARK_IDX["mouth_left"]),
            px(_LANDMARK_IDX["mouth_right"]),
        ], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self._reset_pose_hold()
            return None

        rot_mat, _ = cv2.Rodrigues(rvec)

        now = time.monotonic()
        use_rot_mat, use_rvec = rot_mat, rvec
        if self._last_rot_mat is not None:
            angle_diff = self._rotation_angle_diff_deg(rot_mat, self._last_rot_mat)
            elapsed = now - self._last_accept_time
            max_allowed_deg = self.max_angular_velocity_deg_s * max(elapsed, 1.0 / 120.0)
            if angle_diff > max_allowed_deg:
                if self._reject_since is None:
                    self._reject_since = now
                if now - self._reject_since < self.pose_hold_s:
                    use_rot_mat, use_rvec = self._last_rot_mat, self._last_rvec
                else:
                    self._reject_since = None
            else:
                self._reject_since = None

        pose_mat = cv2.hconcat([use_rot_mat, tvec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [float(a) for a in euler.flatten()]

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        pitch = -pitch

        nose_forward_3d = np.array([[0.0, 0.0, 100.0]])
        nose_end_2d, _ = cv2.projectPoints(
            nose_forward_3d, use_rvec, tvec, self.camera_matrix, self.dist_coeffs)
        nose_end = (int(nose_end_2d[0, 0, 0]), int(nose_end_2d[0, 0, 1]))
        nose_start = (int(image_points[0][0]), int(image_points[0][1]))

        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        cx_norm = (min(xs) + max(xs)) / 2.0
        cy_norm = (min(ys) + max(ys)) / 2.0
        frustum_x = cx_norm * 2.0 - 1.0
        frustum_y = cy_norm * 2.0 - 1.0
        bbox = (int(min(xs) * w), int(min(ys) * h),
                int(max(xs) * w), int(max(ys) * h))

        if (bbox[2] - bbox[0]) < self.min_face_px:
            self._reset_pose_hold()
            return None

        lx, ly = px(_LANDMARK_IDX["left_eye_outer"])
        rx, ry = px(_LANDMARK_IDX["right_eye_outer"])
        pixel_interocular = math.hypot(rx - lx, ry - ly)
        fx = self.camera_matrix[0, 0]
        approx_distance_mm = (
            (self.REAL_INTEROCULAR_MM * fx) / pixel_interocular
            if pixel_interocular > 1e-3 else float("inf")
        )

        self._last_rot_mat = use_rot_mat
        self._last_rvec = use_rvec
        self._last_accept_time = now

        return HeadPoseResult(
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            frustum_x=frustum_x, frustum_y=frustum_y,
            approx_distance_mm=approx_distance_mm, bbox=bbox,
            keypoints=tuple((int(x), int(y)) for x, y in image_points),
            nose_line=(nose_start, nose_end),
        )

    def close(self):
        if self._use_solutions:
            self._face_mesh.close()
        else:
            self._face_landmarker.close()


# ==========================================================================
# Body (torso) fallback -- see face_orientation_tracker.py's BODY (CHEST)
# FALLBACK docstring section for the full rationale (torso-height distance
# estimate for yaw-rotation invariance, per-landmark visibility gating,
# face-height approximation for frustum_y continuity).
# ==========================================================================
@dataclass
class BodyPoseResult:
    frustum_x: float
    frustum_y: float
    approx_distance_mm: float
    distance_source: str
    chest_point: tuple
    shoulder_points: Optional[tuple]
    hip_points: Optional[tuple]


class BodyPoseEstimator:
    REAL_SHOULDER_MM = 400.0
    REAL_HIP_MM = 320.0
    REAL_TORSO_MM = 500.0
    HEAD_OFFSET_FROM_SHOULDER_MM = 180.0
    HEAD_OFFSET_FROM_HIP_MM = HEAD_OFFSET_FROM_SHOULDER_MM + REAL_TORSO_MM

    # Hysteresis band on landmark-pair visibility, same idea as
    # GazeTrigger's enter/exit yaw thresholds. A single hard 0.5 cutoff
    # let shoulders_ok/hips_ok flip every frame whenever a visibility
    # score hovered right at the boundary -- and each flip changes BOTH
    # distance_source (a different distance formula entirely) AND the
    # frustum_y anchor (shoulder line vs hip line, a 180mm vs 680mm
    # offset difference), a real discontinuity in the raw signal that a
    # downstream One Euro filter can't hide: its speed-adaptive design
    # trusts fast-changing input MORE, not less, so it tracks a jump like
    # this quickly rather than rejecting it as noise. Needs to rise above
    # visibility_enter to start being trusted, and fall below the lower
    # visibility_exit to stop -- the gap in between is dead zone a
    # borderline score can sit in without flipping anything. Widened past
    # the first pass (was 0.5/0.35, a 0.15 gap) after still seeing
    # shoulders/hips bouncing -- 0.6/0.2 is a 0.4 gap, trading more lag on
    # a genuine visibility change for a lot more resistance to noise.
    DEFAULT_VISIBILITY_ENTER = 0.6
    DEFAULT_VISIBILITY_EXIT = 0.2

    # Stability gate: at long range/turned away, a little landmark noise
    # swings approx_distance_mm a lot (distance is REAL_*_MM * fx /
    # span_px -- as span_px shrinks, sensitivity to noise in it blows
    # up). Worse, MediaPipe occasionally recognizes something that isn't
    # a person at all (an actual bike, in testing) as a full pose --
    # which shows up as BOTH frustum position AND distance jumping to
    # somewhere completely different at once, not just distance alone.
    # Holding the last good value briefly and then giving up if the
    # implausible readings persisted (an earlier version of this gate)
    # was the wrong shape for that: once the hold timer ran out it just
    # accepted whatever chaotic reading was current, which is exactly
    # the whiplash-between-two-wrong-values pattern seen in testing.
    #
    # What actually helps: only trust output once CONSECUTIVE raw
    # readings agree with EACH OTHER (not with some old frozen
    # reference), and fully BLOCK output (report no body) for as long as
    # they don't. A spurious detection tends to be internally unstable
    # frame to frame -- requiring brief self-consistency before trusting
    # anything filters most of it out, rather than ever reporting a
    # single frame of it. Comparing against the immediately preceding
    # raw reading (not a stale trusted one) also means a genuine large
    # repositioning (person actually walks somewhere new while briefly
    # occluded) still re-locks quickly once IT settles down, rather than
    # being judged forever against a now-outdated reference point.
    DEFAULT_MAX_DISTANCE_VELOCITY_MM_S = 4000.0
    DEFAULT_MAX_POSITION_VELOCITY_PER_S = 3.0  # frustum units/s -- full width/height span is 2.0
    DEFAULT_STABILITY_CONFIRM_S = 0.15

    def __init__(self, width: int, height: int, camera_matrix: np.ndarray,
                 model_path: str, min_body_px: int = 40,
                 visibility_enter: float = DEFAULT_VISIBILITY_ENTER,
                 visibility_exit: float = DEFAULT_VISIBILITY_EXIT,
                 max_distance_velocity_mm_s: float = DEFAULT_MAX_DISTANCE_VELOCITY_MM_S,
                 max_position_velocity_per_s: float = DEFAULT_MAX_POSITION_VELOCITY_PER_S,
                 stability_confirm_s: float = DEFAULT_STABILITY_CONFIRM_S):
        self.width, self.height = width, height
        self.camera_matrix = camera_matrix
        self.min_body_px = min_body_px
        self.visibility_enter = visibility_enter
        self.visibility_exit = visibility_exit
        self.max_distance_velocity_mm_s = max_distance_velocity_mm_s
        self.max_position_velocity_per_s = max_position_velocity_per_s
        self.stability_confirm_s = stability_confirm_s
        self._shoulders_trusted = False
        self._hips_trusted = False
        self._prev_frustum_x: Optional[float] = None
        self._prev_frustum_y: Optional[float] = None
        self._prev_distance_mm: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._consistent_since: Optional[float] = None
        self._blocked = False

        base_options = mp_tasks_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
        )
        options = mp_tasks_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = mp_tasks_vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _min_visibility(*landmarks) -> float:
        """Worst-case visibility across a set of landmarks. None (not
        reported by this model/build at all) is excluded, not treated as
        0 -- matches the prior "None means trust it" behavior."""
        vis = [p.visibility for p in landmarks if p.visibility is not None]
        return min(vis) if vis else 1.0

    def _pair_trusted(self, min_vis: float, was_trusted: bool) -> bool:
        threshold = self.visibility_exit if was_trusted else self.visibility_enter
        return min_vis >= threshold

    def process(self, frame_bgr: np.ndarray) -> Optional[BodyPoseResult]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            self._shoulders_trusted = False
            self._hips_trusted = False
            self._prev_frustum_x = None
            self._prev_frustum_y = None
            self._prev_distance_mm = None
            self._prev_time = None
            self._consistent_since = None
            self._blocked = False
            return None

        lm = result.pose_landmarks[0]
        ls_p = lm[_POSE_LANDMARK_IDX["left_shoulder"]]
        rs_p = lm[_POSE_LANDMARK_IDX["right_shoulder"]]
        lh_p = lm[_POSE_LANDMARK_IDX["left_hip"]]
        rh_p = lm[_POSE_LANDMARK_IDX["right_hip"]]

        shoulders_ok = self._pair_trusted(
            self._min_visibility(ls_p, rs_p), self._shoulders_trusted)
        hips_ok = self._pair_trusted(
            self._min_visibility(lh_p, rh_p), self._hips_trusted)
        self._shoulders_trusted = shoulders_ok
        self._hips_trusted = hips_ok

        ls = (ls_p.x * w, ls_p.y * h) if shoulders_ok else None
        rs = (rs_p.x * w, rs_p.y * h) if shoulders_ok else None
        lh = (lh_p.x * w, lh_p.y * h) if hips_ok else None
        rh = (rh_p.x * w, rh_p.y * h) if hips_ok else None

        if not shoulders_ok and not hips_ok:
            return None

        shoulder_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0) if shoulders_ok else None
        hip_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0) if hips_ok else None

        fx = self.camera_matrix[0, 0]

        if shoulders_ok and hips_ok:
            torso_px = math.hypot(hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
            if torso_px < self.min_body_px:
                return None
            distance_source = "torso"
            approx_distance_mm = (self.REAL_TORSO_MM * fx) / torso_px if torso_px > 1e-3 else float("inf")
        elif shoulders_ok:
            shoulder_px = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
            if shoulder_px < self.min_body_px:
                return None
            distance_source = "shoulders"
            approx_distance_mm = (self.REAL_SHOULDER_MM * fx) / shoulder_px if shoulder_px > 1e-3 else float("inf")
        else:
            hip_px = math.hypot(rh[0] - lh[0], rh[1] - lh[1])
            if hip_px < self.min_body_px:
                return None
            distance_source = "hips"
            approx_distance_mm = (self.REAL_HIP_MM * fx) / hip_px if hip_px > 1e-3 else float("inf")

        centers = [c for c in (shoulder_mid, hip_mid) if c is not None]
        cx_px = sum(c[0] for c in centers) / len(centers)
        cy_px = sum(c[1] for c in centers) / len(centers)
        frustum_x = (cx_px / w) * 2.0 - 1.0

        if shoulders_ok:
            anchor_y_px = shoulder_mid[1]
            head_offset_mm = self.HEAD_OFFSET_FROM_SHOULDER_MM
        else:
            anchor_y_px = hip_mid[1]
            head_offset_mm = self.HEAD_OFFSET_FROM_HIP_MM
        head_offset_px = (head_offset_mm * fx) / approx_distance_mm if approx_distance_mm > 1e-6 else 0.0
        face_approx_y_px = anchor_y_px - head_offset_px
        frustum_y = (face_approx_y_px / h) * 2.0 - 1.0

        now = time.monotonic()
        if self._prev_time is not None:
            elapsed = max(now - self._prev_time, 1.0 / 120.0)
            position_velocity = math.hypot(frustum_x - self._prev_frustum_x,
                                            frustum_y - self._prev_frustum_y) / elapsed
            distance_velocity = (
                abs(approx_distance_mm - self._prev_distance_mm) / elapsed
                if math.isfinite(approx_distance_mm) and math.isfinite(self._prev_distance_mm)
                else 0.0
            )
            consistent = (position_velocity <= self.max_position_velocity_per_s
                          and distance_velocity <= self.max_distance_velocity_mm_s)
        else:
            consistent = True  # nothing to compare the first-ever reading against

        self._prev_frustum_x = frustum_x
        self._prev_frustum_y = frustum_y
        self._prev_distance_mm = approx_distance_mm
        self._prev_time = now

        if not consistent:
            if not self._blocked:
                print(f"[body] blocking -- inconsistent frame-to-frame jump "
                      f"over {elapsed*1000:.0f}ms (position {position_velocity:.2f}/s > "
                      f"{self.max_position_velocity_per_s:.2f}/s, or distance "
                      f"{distance_velocity:.0f}mm/s > {self.max_distance_velocity_mm_s:.0f}mm/s)")
            self._blocked = True
            self._consistent_since = None
            return None

        if self._blocked:
            if self._consistent_since is None:
                self._consistent_since = now
            if now - self._consistent_since < self.stability_confirm_s:
                return None  # readings agree with each other, but not for long enough yet
            self._blocked = False
            print("[body] resuming -- readings stable again")

        return BodyPoseResult(
            frustum_x=frustum_x, frustum_y=frustum_y,
            approx_distance_mm=approx_distance_mm,
            distance_source=distance_source,
            chest_point=(int(cx_px), int(cy_px)),
            shoulder_points=((int(ls[0]), int(ls[1])), (int(rs[0]), int(rs[1]))) if shoulders_ok else None,
            hip_points=((int(lh[0]), int(lh[1])), (int(rh[0]), int(rh[1]))) if hips_ok else None,
        )

    def close(self):
        self._landmarker.close()


class BodyPoseSmoother:
    _FIELDS = ("frustum_x", "frustum_y", "approx_distance_mm")

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.07):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, OneEuroFilter] = {}

    def update(self, pose: Optional[BodyPoseResult], t: float) -> Optional[BodyPoseResult]:
        if pose is None:
            self._filters = {}
            return None
        values = {f: getattr(pose, f) for f in self._FIELDS}
        smoothed = {}
        for f, x in values.items():
            filt = self._filters.get(f)
            if filt is None:
                filt = OneEuroFilter(t, x, min_cutoff=self.min_cutoff, beta=self.beta)
                self._filters[f] = filt
                smoothed[f] = x
            else:
                smoothed[f] = filt(t, x)
        return BodyPoseResult(
            frustum_x=smoothed["frustum_x"], frustum_y=smoothed["frustum_y"],
            approx_distance_mm=smoothed["approx_distance_mm"],
            distance_source=pose.distance_source,
            chest_point=pose.chest_point, shoulder_points=pose.shoulder_points,
            hip_points=pose.hip_points,
        )


class BodyPoseWorker:
    """Runs BodyPoseEstimator continuously in its own thread off the same
    always-latest-frame slot CameraStream maintains, so it's warm the
    instant face tracking drops out.

    Deliberately throttled to target_fps rather than running flat-out as
    fast as the hardware allows: BodyPoseEstimator's stability gate
    reasons in per-SECOND terms (frustum units/s, mm/s), so how fast this
    loop actually calls process() matters a lot -- the same absolute
    frame-to-frame jitter implies a much larger "velocity" the smaller
    the elapsed time between calls is (velocity = delta/dt). Left
    unthrottled, that rate varies with whatever hardware this happens to
    run on (a beefy dev Mac vs. the target Pi), silently invalidating
    whatever the gate's thresholds were tuned against. Fixing the rate
    keeps the gate's calibration meaningful across machines, and costs
    less CPU on the Pi besides."""

    def __init__(self, cam: CameraStream, estimator: BodyPoseEstimator,
                 target_fps: float = 15.0):
        self._cam = cam
        self._estimator = estimator
        self._min_interval = 1.0 / target_fps
        self._lock = threading.Lock()
        self._result: Optional[BodyPoseResult] = None
        self._timestamp = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        last_frame_ts = 0.0
        last_process_time = 0.0
        while self._running:
            frame, ts = self._cam.latest()
            now = time.monotonic()
            if (frame is None or ts == last_frame_ts
                    or now - last_process_time < self._min_interval):
                time.sleep(0.001)
                continue
            last_frame_ts = ts
            last_process_time = now
            result = self._estimator.process(frame)
            with self._lock:
                self._result = result
                self._timestamp = time.monotonic()

    def latest(self):
        with self._lock:
            return self._result, self._timestamp

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._estimator.close()


class SourceHysteresis:
    """Debounces which tracking source (face vs body) is active, mirroring
    GazeTrigger's enter/exit-dwell pattern -- see
    face_orientation_tracker.py's comment for the full rationale."""

    def __init__(self, hold_face_s: float = 0.3, confirm_face_s: float = 0.15):
        self.hold_face_s = hold_face_s
        self.confirm_face_s = confirm_face_s
        self._using_face = False
        self._last_good_pose: Optional[HeadPoseResult] = None
        self._face_seen_since: Optional[float] = None
        self._face_lost_at: Optional[float] = None

    def update(self, pose: Optional[HeadPoseResult], now: float) -> Optional[HeadPoseResult]:
        if pose is not None:
            if self._face_seen_since is None:
                self._face_seen_since = now
            self._face_lost_at = None
            self._last_good_pose = pose
            if not self._using_face and now - self._face_seen_since >= self.confirm_face_s:
                self._using_face = True
            return pose if self._using_face else None

        self._face_seen_since = None
        if self._using_face:
            if self._face_lost_at is None:
                self._face_lost_at = now
            if now - self._face_lost_at < self.hold_face_s:
                return self._last_good_pose
            self._using_face = False
            self._last_good_pose = None
        return None


class GazeTrigger:
    def __init__(self,
                 enter_yaw_deg: float = 40.0, exit_yaw_deg: float = 50.0,
                 dwell_s: float = 0.25,
                 on_facing=None, on_not_facing=None):
        self.enter_yaw = enter_yaw_deg
        self.exit_yaw = exit_yaw_deg
        self.dwell_s = dwell_s
        self.on_facing = on_facing or (lambda r: print("[trigger] FACING CAMERA", r))
        self.on_not_facing = on_not_facing or (lambda: print("[trigger] not facing"))
        self._facing_since: Optional[float] = None
        self._is_facing = False

    @property
    def is_facing(self) -> bool:
        return self._is_facing

    def update(self, pose: Optional[HeadPoseResult], now: float):
        if pose is None:
            self._facing_since = None
            if self._is_facing:
                self._is_facing = False
                self.on_not_facing()
            return

        within_enter = abs(pose.yaw_deg) < self.enter_yaw
        within_exit = abs(pose.yaw_deg) < self.exit_yaw

        if not self._is_facing:
            if within_enter:
                if self._facing_since is None:
                    self._facing_since = now
                elif now - self._facing_since >= self.dwell_s:
                    self._is_facing = True
                    self.on_facing(pose)
            else:
                self._facing_since = None
        else:
            if not within_exit:
                self._is_facing = False
                self._facing_since = None
                self.on_not_facing()


def direction_label(yaw_deg: float, pitch_deg: float, deadzone: float = 8.0) -> str:
    parts = []
    if pitch_deg > deadzone:
        parts.append("UP")
    elif pitch_deg < -deadzone:
        parts.append("DOWN")
    if yaw_deg > deadzone:
        parts.append("RIGHT")
    elif yaw_deg < -deadzone:
        parts.append("LEFT")
    return " ".join(parts) if parts else "CENTER"


def draw_debug_overlay(frame: np.ndarray, pose: Optional[HeadPoseResult],
                        body_pose: Optional[BodyPoseResult], mirror: bool) -> np.ndarray:
    """The tracker's original --display overlay, unchanged, factored out
    so it can be rendered only when the debug view is actually active."""
    vis = cv2.flip(frame, 1) if mirror else frame.copy()
    vw = vis.shape[1]

    def mx(x):
        return vw - x if mirror else x

    if pose is not None:
        x0, y0, x1, y1 = pose.bbox
        box_x0, box_x1 = sorted((mx(x0), mx(x1)))
        cv2.rectangle(vis, (box_x0, y0), (box_x1, y1), (0, 255, 0), 2)
        for kx, ky in pose.keypoints:
            cv2.circle(vis, (mx(kx), ky), 4, (0, 255, 255), -1)
        (nsx, nsy), (nex, ney) = pose.nose_line
        cv2.arrowedLine(vis, (mx(nsx), nsy), (mx(nex), ney),
                         (0, 0, 255), 3, tipLength=0.3)
        label = direction_label(pose.yaw_deg, pose.pitch_deg)
        cv2.putText(vis, f"yaw {pose.yaw_deg:+.1f} pitch {pose.pitch_deg:+.1f} -> {label}",
                    (box_x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"dist ~{pose.approx_distance_mm:.0f}mm",
                    (box_x0, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    elif body_pose is not None:
        if body_pose.shoulder_points is not None:
            (lx, ly), (rx, ry) = body_pose.shoulder_points
            cv2.line(vis, (mx(lx), ly), (mx(rx), ry), (255, 128, 0), 2)
        if body_pose.hip_points is not None:
            (lx, ly), (rx, ry) = body_pose.hip_points
            cv2.line(vis, (mx(lx), ly), (mx(rx), ry), (255, 128, 0), 2)
        cx, cy = body_pose.chest_point
        cv2.circle(vis, (mx(cx), cy), 6, (255, 128, 0), -1)
        cv2.putText(vis, f"BODY (no face) dist ~{body_pose.approx_distance_mm:.0f}mm "
                         f"[{body_pose.distance_source}]",
                    (mx(cx) - 80, cy - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 128, 0), 2)
    return vis


# ==========================================================================
# Idle wander -- see eye_display.py's original module docstring for the
# full rationale (a hold/move state machine, not real noise -- cheaper and
# a closer likeness of how eyes actually move at rest).
# ==========================================================================
class SaccadeWander:
    def __init__(self, radius: float = 0.5,
                 hold_range: tuple = (0.6, 2.5),
                 move_range: tuple = (0.08, 0.25),
                 seed: Optional[int] = None,
                 now: Optional[float] = None):
        self.radius = radius
        self.hold_range = hold_range
        self.move_range = move_range
        self._rng = random.Random(seed)

        self._from = (0.0, 0.0)
        self._to = (0.0, 0.0)
        self._state = "hold"
        self._state_start = now if now is not None else time.monotonic()
        self._duration = self._rng.uniform(*hold_range)

    def _new_target(self):
        angle = self._rng.uniform(0.0, 2.0 * math.pi)
        r = self.radius * math.sqrt(self._rng.uniform(0.0, 1.0))
        return (r * math.cos(angle), r * math.sin(angle))

    def update(self, now: float) -> tuple:
        elapsed = now - self._state_start
        if elapsed >= self._duration:
            if self._state == "hold":
                self._from = self._to
                self._to = self._new_target()
                self._state = "move"
                self._duration = self._rng.uniform(*self.move_range)
            else:
                self._state = "hold"
                self._duration = self._rng.uniform(*self.hold_range)
            self._state_start = now
            elapsed = 0.0

        if self._state == "hold":
            return self._to

        t = max(0.0, min(1.0, elapsed / self._duration)) if self._duration > 0 else 1.0
        eased = t * t * (3.0 - 2.0 * t)
        x = self._from[0] + (self._to[0] - self._from[0]) * eased
        y = self._from[1] + (self._to[1] - self._from[1]) * eased
        return (x, y)

    def flee(self, now: float):
        """Force an immediate, fast move to a fresh random target --
        used for averting, where eyes need to dart away the instant eye
        contact is made."""
        self._from = self._to
        self._to = self._new_target()
        self._state = "move"
        self._duration = self._rng.uniform(*self.move_range)
        self._state_start = now


class ValueDebouncer:
    """Requires the same value to be reported min_count consecutive times
    in a row before actually committing to it -- a discrete, sample-count
    debounce, distinct from the time-based hold/confirm patterns used
    upstream (GazeTrigger, SourceHysteresis, BodyPoseEstimator's
    stability gate). Those all reduce flicker in the TRACKING signal
    itself, but a single still-noisy frame can slip through all of them
    and reach the eyes as a one-frame flip -- which is exactly what a
    visible twitch looks like. This is the last layer, right at the
    point the eyes decide their mode (tracking / idle wander /
    averting), so nothing downstream of it ever sees a single-sample
    blip."""

    def __init__(self, min_count: int = 3, initial=False):
        self.min_count = min_count
        self._committed = initial
        self._candidate = initial
        self._streak = min_count  # initial value starts already-confirmed

    def update(self, value) -> object:
        if value == self._candidate:
            self._streak += 1
        else:
            self._candidate = value
            self._streak = 1
        if self._streak >= self.min_count:
            self._committed = self._candidate
        return self._committed


class BlinkGenerator:
    """One instance per PAIR -- both eyes in a pair blink together, each
    pair on its own independent random schedule."""

    def __init__(self, interval_range: tuple = (2.0, 6.0),
                 duration_range: tuple = (0.12, 0.25),
                 seed: Optional[int] = None,
                 now: Optional[float] = None):
        self._interval_range = interval_range
        self._duration_range = duration_range
        self._rng = random.Random(seed)
        now = now if now is not None else time.monotonic()
        self._next_blink_at = now + self._rng.uniform(*interval_range)
        self._blink_start: Optional[float] = None
        self._blink_duration = 0.0

    def update(self, now: float) -> float:
        if self._blink_start is None:
            if now < self._next_blink_at:
                return 1.0
            self._blink_start = now
            self._blink_duration = self._rng.uniform(*self._duration_range)

        t = (now - self._blink_start) / self._blink_duration if self._blink_duration > 0 else 1.0
        if t >= 1.0:
            self._blink_start = None
            self._next_blink_at = now + self._rng.uniform(*self._interval_range)
            return 1.0

        return 1.0 - math.sin(math.pi * max(0.0, min(1.0, t)))


# ==========================================================================
# Eye renderer -- ported from the jit.gen shader. See eye_display.py's
# original module docstring (SHADER PORT NOTES / AVERTING) for the full
# design rationale; unchanged here.
# ==========================================================================
def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class EyeRenderer:
    IRIS_COLOR_BGR = np.array([0.2, 0.4, 0.1])

    def __init__(self, width: int, height: int, grid: int = 3, eye_gap: float = 0.45):
        self.width, self.height = width, height
        self.grid = grid
        self.num_pairs = grid * grid
        self.eye_gap = eye_gap

        ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)

        col_width = width / grid
        row_height = height / grid

        col_idx = np.minimum((xs / col_width).astype(np.int64), grid - 1)
        row_idx = np.minimum((ys / row_height).astype(np.int64), grid - 1)
        self._pair_idx = row_idx * grid + col_idx

        pair_center_x = (col_idx + 0.5) * col_width
        pair_center_y = (row_idx + 0.5) * row_height

        self._u = (xs - pair_center_x) / row_height
        self._v = (ys - pair_center_y) / row_height

    def _eye_layer(self, u: np.ndarray, v: np.ndarray, stare_x, stare_y, openness):
        eye_d = np.hypot(u * 12.0, v * (12.0 * 1.1))
        eye_mask = _smoothstep(-5.0, 0.3, 1.0 - eye_d)

        dx = u * (32.0 * 1.333) + stare_x
        dy = v * 32.0 + stare_y
        d2 = np.hypot(dx, dy)
        iris_soft = _smoothstep(-9.0, -7.0, 1.0 - d2)
        pupil_mask = _smoothstep(-4.0, -2.0, 1.0 - d2)
        iris_bin = iris_soft > 0.5

        base_gray = eye_mask - pupil_mask
        base_rgb = np.repeat(base_gray[..., None], 3, axis=2)
        iri_rgb = (iris_soft * eye_mask)[..., None] * self.IRIS_COLOR_BGR[None, None, :]

        combined = np.where(iris_bin[..., None], iri_rgb, base_rgb)
        result = combined - pupil_mask[..., None]
        return result * openness[..., None]

    def render(self, pair_targets, pair_openness, look_gain: float = 16.0) -> np.ndarray:
        targets = np.asarray(pair_targets, dtype=np.float64)
        stare = targets * look_gain
        stare_x_map = stare[:, 0][self._pair_idx]
        stare_y_map = stare[:, 1][self._pair_idx]

        openness_map = np.asarray(pair_openness, dtype=np.float64)[self._pair_idx]

        left = self._eye_layer(self._u - self.eye_gap, self._v, stare_x_map, stare_y_map, openness_map)
        right = self._eye_layer(self._u + self.eye_gap, self._v, stare_x_map, stare_y_map, openness_map)

        final = np.clip(left + right, 0.0, 1.0)
        return (final * 255).astype(np.uint8)


def check_gl_available():
    if glfw is None or moderngl is None:
        print(
            "ERROR: glfw and/or moderngl are not installed.\n"
            "  pip install glfw moderngl"
        )
        sys.exit(1)


# ==========================================================================
# GPU-rendered eyes -- replaces EyeRenderer's CPU numpy math with a real
# GLSL fragment shader. EyeRenderer measured ~7fps on a Pi 5 even after
# shrinking the render resolution 6x (a stopgap tried before this); plain
# numpy has no hardware acceleration behind it on that platform the way it does
# via Apple's Accelerate framework on the macOS dev machine, and this is
# exactly the kind of massively-parallel per-pixel math a GPU is built
# for and a CPU isn't -- fitting, since this whole renderer started life
# as an actual fragment shader (a Max/MSP jit.gen patch) before being
# ported to numpy in the first place. This ports it back, with the exact
# same per-eye math (same constants: 12, 1.1, 32, 1.33, the smoothstep
# bands) -- verified pixel-for-pixel against EyeRenderer offscreen before
# this was wired in; the only real difference is GLSL's y=0-at-bottom
# framebuffer convention, flipped once in the shared vertex shader so
# everything downstream (grid bucketing, the eye math's up/down sense)
# agrees with numpy's y=0-at-top convention.
#
# One shared GLFW window/moderngl context serves BOTH views: the eyes
# shader renders directly (no CPU round-trip at all -- compute and
# display both happen on the GPU), and the debug view uploads the
# already-drawn camera+overlay frame (built by draw_debug_overlay,
# unchanged, still cheap CPU work -- that view was never the bottleneck)
# as a plain texture and draws it as a textured quad. This is what keeps
# the eyes/debug toggle simple: same window, same context, just a
# different draw call.
# ==========================================================================
# Two GLSL dialects to support: desktop GL 3.3 core (confirmed working on
# the macOS dev machine) and GLSL ES 3.00 (what the Pi 5's Mesa V3D driver
# actually provides via EGL -- desktop GL context creation there fails
# outright, "EGL: Failed to create context", confirmed in testing). The
# shader BODIES below are written to be valid under either dialect (same
# in/out qualifiers, same explicit fragColor output, same builtins) --
# only the required header differs (GLES needs an explicit float
# precision declaration desktop GL doesn't), so GLDisplay picks the
# header to prepend based on which context it actually managed to create.
_GLSL_HEADER_DESKTOP = "#version 330\n"
_GLSL_HEADER_GLES = "#version 300 es\nprecision mediump float;\nprecision mediump int;\n"

_GL_VERTEX_SHADER = """
in vec2 in_position;
out vec2 v_uv;
void main() {
    // 0..1, y-flipped so v_uv.y=0 is the TOP of the image, matching
    // numpy's row-0-is-top convention (GL's own is bottom-up).
    v_uv = vec2(in_position.x * 0.5 + 0.5, 1.0 - (in_position.y * 0.5 + 0.5));
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_EYE_FRAGMENT_SHADER = """
in vec2 v_uv;
out vec4 fragColor;

uniform vec2 u_canvas;
uniform int u_grid;
uniform float u_eye_gap;
uniform float u_look_gain;
uniform vec2 u_targets[9];
uniform float u_openness[9];
uniform vec3 u_iris_color;

vec3 eyeLayer(vec2 u, vec2 stare, float openness) {
    float eye_d = length(vec2(u.x * 12.0, u.y * 13.2));
    float eye_mask = smoothstep(-5.0, 0.3, 1.0 - eye_d);

    vec2 dd = vec2(u.x * 32.0 * 1.333 + stare.x, u.y * 32.0 + stare.y);
    float d2 = length(dd);
    float iris_soft = smoothstep(-9.0, -7.0, 1.0 - d2);
    float pupil_mask = smoothstep(-4.0, -2.0, 1.0 - d2);
    bool iris_bin = iris_soft > 0.5;

    float base_gray = eye_mask - pupil_mask;
    vec3 base_rgb = vec3(base_gray);
    vec3 iri_rgb = (iris_soft * eye_mask) * u_iris_color;

    vec3 combined = iris_bin ? iri_rgb : base_rgb;
    return (combined - pupil_mask) * openness;
}

void main() {
    vec2 px = v_uv * u_canvas;
    float col_width = u_canvas.x / float(u_grid);
    float row_height = u_canvas.y / float(u_grid);

    int col = int(min(floor(px.x / col_width), float(u_grid - 1)));
    int row = int(min(floor(px.y / row_height), float(u_grid - 1)));
    int pairIdx = row * u_grid + col;

    vec2 pairCenter = vec2((float(col) + 0.5) * col_width, (float(row) + 0.5) * row_height);
    vec2 u = (px - pairCenter) / row_height;

    vec2 stare = u_targets[pairIdx] * u_look_gain;
    float openness = u_openness[pairIdx];

    vec3 left = eyeLayer(vec2(u.x - u_eye_gap, u.y), stare, openness);
    vec3 right = eyeLayer(vec2(u.x + u_eye_gap, u.y), stare, openness);

    fragColor = vec4(clamp(left + right, 0.0, 1.0), 1.0);
}
"""

_DEBUG_FRAGMENT_SHADER = """
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
    // Texture data is uploaded BGR (matches OpenCV's native order, no
    // CPU-side channel-swap needed) -- swizzle it back to RGB here for
    // correct on-screen color.
    fragColor = vec4(texture(u_tex, v_uv).bgr, 1.0);
}
"""


class GLDisplay:
    def __init__(self, width: int, height: int, title: str = "eye rig"):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        self._window = self._create_window_desktop_or_gles(width, height, title)
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)  # vsync -- avoids burning CPU rendering faster than the display can show

        # moderngl.create_context() defaults its internal min-version
        # check to 330 even when require=None is passed -- fine for the
        # desktop path (that's what we requested), but a GLES context
        # reports its own version_code scheme (e.g. GLES 3.1 -> 310,
        # confirmed on the Pi -- the driver gave a newer version than the
        # 3.0 minimum requested, which is normal), and 310 < 330 would
        # otherwise raise even though the context is perfectly usable.
        self.ctx = moderngl.create_context(require=300 if self._using_gles else 330)
        header = _GLSL_HEADER_GLES if self._using_gles else _GLSL_HEADER_DESKTOP
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._quad_vbo = self.ctx.buffer(quad.tobytes())

        self._eye_prog = self.ctx.program(vertex_shader=header + _GL_VERTEX_SHADER,
                                           fragment_shader=header + _EYE_FRAGMENT_SHADER)
        self._eye_vao = self.ctx.vertex_array(self._eye_prog, [(self._quad_vbo, "2f", "in_position")])

        self._debug_prog = self.ctx.program(vertex_shader=header + _GL_VERTEX_SHADER,
                                             fragment_shader=header + _DEBUG_FRAGMENT_SHADER)
        self._debug_vao = self.ctx.vertex_array(self._debug_prog, [(self._quad_vbo, "2f", "in_position")])
        self._debug_tex = None
        self._debug_tex_size = None

        self._pending_key: Optional[str] = None
        glfw.set_key_callback(self._window, self._on_key)
        self._current_size = (width, height)

    def _create_window_desktop_or_gles(self, width: int, height: int, title: str):
        """Tries desktop OpenGL 3.3 core first (confirmed working via
        moderngl on macOS); if that fails, resets hints and retries as
        OpenGL ES 3.0 (what the Pi 5's Mesa V3D driver actually provides
        -- desktop GL context creation there fails with "EGL: Failed to
        create context", confirmed in testing). Sets self._using_gles so
        callers know which GLSL header to compile shaders with."""
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        window = glfw.create_window(width, height, title, None, None)
        if window:
            self._using_gles = False
            return window

        glfw.default_window_hints()
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        window = glfw.create_window(width, height, title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed under both desktop GL and GLES")
        self._using_gles = True
        return window

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_Q:
            self._pending_key = "q"
        elif key == glfw.KEY_D:
            self._pending_key = "d"

    def poll(self) -> Optional[str]:
        """Polls window events and returns the most recent key pressed
        since the last call ('d', 'q', or None) -- mirrors the role
        cv2.waitKey played in the old loop."""
        glfw.poll_events()
        key = self._pending_key
        self._pending_key = None
        return key

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self._window))

    def _ensure_size(self, width: int, height: int):
        if (width, height) != self._current_size:
            glfw.set_window_size(self._window, width, height)
            self._current_size = (width, height)

    def _use_full_viewport(self):
        # The window's actual framebuffer can be larger than the logical
        # size we asked for (HiDPI/Retina displays commonly render at 2x)
        # -- using the requested width/height for the viewport instead of
        # the real framebuffer size would only fill a corner of the
        # window. u_canvas (the shader's own notion of width/height,
        # which drives its internal proportions) is unaffected -- this
        # only changes how many actual pixels get rasterized into it.
        fb_w, fb_h = glfw.get_framebuffer_size(self._window)
        self.ctx.viewport = (0, 0, fb_w, fb_h)

    def render_eyes(self, width: int, height: int, grid: int, eye_gap: float,
                     look_gain: float, targets, openness):
        self._ensure_size(width, height)
        self.ctx.screen.use()
        self._use_full_viewport()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._eye_prog["u_canvas"].value = (float(width), float(height))
        self._eye_prog["u_grid"].value = grid
        self._eye_prog["u_eye_gap"].value = eye_gap
        self._eye_prog["u_look_gain"].value = look_gain
        self._eye_prog["u_targets"].value = [tuple(t) for t in targets]
        self._eye_prog["u_openness"].value = list(openness)
        # True RGB (not BGR) -- this renders straight to the screen, no
        # OpenCV BGR convention involved. Matches the original shader's
        # literal vec(.1, .4, .2).
        self._eye_prog["u_iris_color"].value = (0.1, 0.4, 0.2)
        self._eye_vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(self._window)

    def render_debug(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        self._ensure_size(w, h)
        if self._debug_tex is None or self._debug_tex_size != (w, h):
            if self._debug_tex is not None:
                self._debug_tex.release()
            self._debug_tex = self.ctx.texture((w, h), 3)
            self._debug_tex_size = (w, h)
        self._debug_tex.write(np.ascontiguousarray(frame_bgr).tobytes())
        self.ctx.screen.use()
        self._use_full_viewport()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._debug_tex.use(location=0)
        self._debug_prog["u_tex"].value = 0
        self._debug_vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(self._window)

    def close(self):
        glfw.terminate()


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    # --- camera / tracking ---
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280,
                     help="Camera capture width (see --eye-width for the "
                          "eyes window instead).")
    ap.add_argument("--height", type=int, default=720,
                     help="Camera capture height (see --eye-height for the "
                          "eyes window instead).")
    ap.add_argument("--no-auto-brighten", dest="auto_brighten",
                     action="store_false",
                     help="Disable the software auto-brightness (on by "
                          "default). Bipolar -- brightens dark scenes AND "
                          "dims overexposed ones, slewed to avoid flicker "
                          "-- see AutoBrighten.")
    ap.add_argument("--target-brightness", type=float, default=120.0,
                     help="Auto-brighten target mean brightness (0-255).")
    ap.add_argument("--min-gain", type=float, default=0.25,
                     help="Auto-brighten gain floor -- how much it can dim "
                          "an overexposed frame.")
    ap.add_argument("--max-gain", type=float, default=4.0,
                     help="Auto-brighten gain ceiling -- how much it can "
                          "brighten a dark frame.")
    ap.add_argument("--brighten-slew", type=float, default=0.5,
                     help="Max gain change per second -- how fast the "
                          "APPLIED gain can ramp toward whatever the "
                          "current frame implies. Lower = smoother/slower "
                          "response to a real lighting change but less "
                          "flicker from ordinary per-frame noise; this is "
                          "the knob for 'the whole point is no flicker.'")
    ap.add_argument("--fps", type=int, default=90,
                     help="Requested camera fps -- see face_orientation_"
                          "tracker.py's FRAME RATE note.")
    ap.add_argument("--min-cutoff", type=float, default=0.8,
                     help="One Euro filter (tracking): cutoff freq when "
                          "signal is still.")
    ap.add_argument("--beta", type=float, default=0.07,
                     help="One Euro filter (tracking): how much speed "
                          "reduces smoothing.")
    ap.add_argument("--no-smoothing", action="store_true",
                     help="Use raw per-frame pose estimates, no filtering.")
    ap.add_argument("--hfov", type=float, default=112.0,
                     help="Lens horizontal FOV in degrees.")
    ap.add_argument("--face-model", type=str, default="face_landmarker.task",
                     help="Path to the MediaPipe Tasks FaceLandmarker model, "
                          "used only as a fallback when mp.solutions.face_mesh "
                          "isn't available in this mediapipe install (dropped "
                          "entirely as of mediapipe 1.0.0 -- the only version "
                          "with wheels on some platforms, e.g. Raspberry Pi).")
    ap.add_argument("--face-detect-confidence", type=float, default=0.5,
                     help="Tasks API FaceLandmarker fallback only: minimum "
                          "confidence to detect a face at all. The Tasks API's "
                          "detector behaves differently from mp.solutions."
                          "face_mesh's at the same threshold -- if face "
                          "tracking now needs you much closer to the camera "
                          "than it used to, try lowering this.")
    ap.add_argument("--face-presence-confidence", type=float, default=0.5,
                     help="Tasks API FaceLandmarker fallback only: minimum "
                          "confidence that a previously-detected face is "
                          "still present.")
    ap.add_argument("--face-track-confidence", type=float, default=0.5,
                     help="Tasks API FaceLandmarker fallback only: minimum "
                          "confidence to keep tracking an already-detected "
                          "face frame to frame.")
    ap.add_argument("--min-face-px", type=int, default=60,
                     help="Reject face detections narrower than this many "
                          "pixels.")
    ap.add_argument("--max-angular-velocity", type=float, default=500.0,
                     help="Degrees/second -- reject an implausible solvePnP "
                          "rotation jump (steep-angle pose ambiguity).")
    ap.add_argument("--pose-hold", type=float, default=0.2,
                     help="Seconds to hold the last known-good rotation "
                          "when a solvePnP jump is rejected.")
    ap.add_argument("--no-body-fallback", dest="body_fallback",
                     action="store_false",
                     help="Disable the chest/shoulder body-position "
                          "fallback.")
    ap.add_argument("--pose-model", type=str, default="pose_landmarker.task",
                     help="Path to the MediaPipe Tasks PoseLandmarker model.")
    ap.add_argument("--min-body-px", type=int, default=40,
                     help="Reject body detections narrower than this many "
                          "pixels.")
    ap.add_argument("--body-fps", type=float, default=15.0,
                     help="Throttle body-fallback inference to this rate, "
                          "rather than running as fast as the hardware "
                          "allows -- the stability gate's velocity "
                          "thresholds (mm/s, frustum units/s) are only "
                          "meaningful relative to a known, fixed sampling "
                          "rate; unthrottled, the same jitter reads as a "
                          "much bigger 'velocity' on faster hardware, "
                          "silently invalidating the gate's calibration.")
    ap.add_argument("--visibility-enter", type=float,
                     default=BodyPoseEstimator.DEFAULT_VISIBILITY_ENTER,
                     help="Shoulder/hip pair visibility must rise above "
                          "this to START being trusted (see BodyPoseEstimator's "
                          "hysteresis comment -- prevents shoulders_ok/"
                          "hips_ok flapping on a noisy visibility score).")
    ap.add_argument("--visibility-exit", type=float,
                     default=BodyPoseEstimator.DEFAULT_VISIBILITY_EXIT,
                     help="Shoulder/hip pair visibility must fall below "
                          "this to STOP being trusted. Should be well below "
                          "--visibility-enter -- the gap is the hysteresis "
                          "band. Widen the gap (raise enter, lower exit) if "
                          "shoulders/hips are still bouncing.")
    ap.add_argument("--max-distance-velocity", type=float,
                     default=BodyPoseEstimator.DEFAULT_MAX_DISTANCE_VELOCITY_MM_S,
                     help="mm/s -- body tracking BLOCKS output (reports no "
                          "body) whenever a new distance reading implies "
                          "moving faster than this relative to the "
                          "previous one (at long range/turned away, a "
                          "little landmark noise swings approx_distance_mm "
                          "a lot; a completely wrong detection -- e.g. an "
                          "object mistaken for a person -- swings it even "
                          "more). Watch the console for '[body] blocking' "
                          "lines (they print the implied mm/s) to tune "
                          "this from real numbers rather than guessing.")
    ap.add_argument("--max-position-velocity", type=float,
                     default=BodyPoseEstimator.DEFAULT_MAX_POSITION_VELOCITY_PER_S,
                     help="Frustum units/second (the -1..1 span is 2.0 "
                          "wide) -- body tracking also BLOCKS output "
                          "whenever frustum position jumps faster than "
                          "this. This is what catches a wrong detection "
                          "appearing somewhere else in frame entirely, not "
                          "just a distance swing on the same real target.")
    ap.add_argument("--stability-confirm", type=float,
                     default=BodyPoseEstimator.DEFAULT_STABILITY_CONFIRM_S,
                     help="Seconds consecutive readings must agree with "
                          "each other before output resumes after being "
                          "blocked -- requiring brief self-consistency "
                          "before trusting anything is what filters out a "
                          "spurious, internally-unstable detection rather "
                          "than eventually accepting whatever it settles "
                          "on.")
    ap.add_argument("--body-min-cutoff", type=float, default=0.3,
                     help="One Euro filter (body position/distance): "
                          "cutoff freq when signal is still. Separate from "
                          "--min-cutoff (face) and defaulted lower/heavier "
                          "-- the body fallback's job is stability over "
                          "responsiveness, and its signal is inherently "
                          "noisier than the face's.")
    ap.add_argument("--body-beta", type=float, default=0.01,
                     help="One Euro filter (body): how much speed reduces "
                          "smoothing. Separate from --beta (face) and "
                          "defaulted lower -- a fast-changing body reading "
                          "is more often noise than genuine fast motion, "
                          "unlike the face's yaw/pitch.")
    ap.add_argument("--enter-yaw", type=float, default=40.0,
                     help="Yaw threshold (degrees) to START being 'facing'.")
    ap.add_argument("--exit-yaw", type=float, default=50.0,
                     help="Yaw threshold (degrees) to STOP being 'facing'.")
    ap.add_argument("--dwell", type=float, default=0.25,
                     help="Seconds within the enter threshold before the "
                          "gaze trigger fires.")
    ap.add_argument("--face-hold", type=float, default=0.3,
                     help="Seconds to hold the last known face pose after "
                          "it's lost before handing off to body.")
    ap.add_argument("--face-confirm", type=float, default=0.15,
                     help="Seconds face must stay present before being "
                          "trusted as active again.")

    # --- OSC (optional -- for the physical rig / Max / TouchDesigner) ---
    ap.add_argument("--no-osc", dest="osc", action="store_false",
                     help="Don't send OSC at all (the on-screen eyes don't "
                          "need it -- this is only for external consumers "
                          "like a physical animatronic rig).")
    ap.add_argument("--osc-host", type=str, default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=9000)

    # --- window / view toggle ---
    ap.add_argument("--no-window", action="store_true",
                     help="Don't show any window at all (true headless -- "
                          "e.g. unattended Pi deployment driving only OSC "
                          "output). Overrides --debug.")
    ap.add_argument("--debug", action="store_true",
                     help="Start in the debug (tracker overlay) view "
                          "instead of the eyes view. Press 'd' to toggle "
                          "either way at runtime.")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false",
                     help="Disable selfie-style mirroring of the debug "
                          "view only -- doesn't affect tracking math.")

    # --- eyes ---
    ap.add_argument("--eye-width", type=int, default=1200,
                     help="Eyes window width. More width = more room "
                          "BETWEEN pair columns (independent of --eye-gap, "
                          "which is the within-a-pair spacing).")
    ap.add_argument("--eye-height", type=int, default=480,
                     help="Eyes window height.")
    ap.add_argument("--grid", type=int, default=3,
                     help="Grid is NxN pairs of eyes (default 3 -> nine "
                          "pairs, matching the original Max patch).")
    ap.add_argument("--eye-gap", type=float, default=0.45,
                     help="Horizontal offset of each eye from its pair's "
                          "center -- the within-a-pair spacing knob.")
    ap.add_argument("--look-gain", type=float, default=16.0,
                     help="How far the iris/pupil shifts per unit of "
                          "frustum position.")
    ap.add_argument("--eye-min-cutoff", type=float, default=1.0,
                     help="One Euro filter (eyes): cutoff freq when signal "
                          "is still.")
    ap.add_argument("--eye-beta", type=float, default=0.02,
                     help="One Euro filter (eyes): how much speed reduces "
                          "smoothing.")
    ap.add_argument("--no-wander", action="store_true",
                     help="Disable idle saccade wander -- rest dead-center "
                          "when nobody's tracked.")
    ap.add_argument("--wander-radius", type=float, default=0.5)
    ap.add_argument("--wander-hold-min", type=float, default=0.6)
    ap.add_argument("--wander-hold-max", type=float, default=2.5)
    ap.add_argument("--wander-move-min", type=float, default=0.08)
    ap.add_argument("--wander-move-max", type=float, default=0.25)
    ap.add_argument("--eye-mode-confirm", type=int, default=3,
                     help="Consecutive frames is_facing (or 'nobody "
                          "tracked') must agree before the eyes actually "
                          "switch mode (tracking/idle wander/averting) -- "
                          "a count-based debounce right at the eyes, on "
                          "top of the tracker's own time-based debouncing "
                          "(GazeTrigger, SourceHysteresis). Raise this if "
                          "the eyes still twitch on a single noisy frame; "
                          "1 disables it (react immediately).")
    ap.add_argument("--no-avert", action="store_true",
                     help="Disable averting on direct eye contact.")
    ap.add_argument("--avert-radius", type=float, default=0.85)
    ap.add_argument("--avert-hold-min", type=float, default=0.3)
    ap.add_argument("--avert-hold-max", type=float, default=1.0)
    ap.add_argument("--avert-move-min", type=float, default=0.05)
    ap.add_argument("--avert-move-max", type=float, default=0.15)
    ap.add_argument("--no-blink", action="store_true")
    ap.add_argument("--blink-interval-min", type=float, default=2.0)
    ap.add_argument("--blink-interval-max", type=float, default=6.0)
    ap.add_argument("--blink-duration-min", type=float, default=0.12)
    ap.add_argument("--blink-duration-max", type=float, default=0.25)
    args = ap.parse_args()

    osc_client = None
    if args.osc:
        if SimpleUDPClient is None:
            print("[osc] python-osc not installed -- continuing without "
                  "OSC output. (pip install python-osc, or pass --no-osc "
                  "to silence this.)")
        else:
            osc_client = SimpleUDPClient(args.osc_host, args.osc_port)
    _osc_error_printed = False

    def send_osc(is_facing: bool, source: int, fx: float, fy: float,
                 dist: float, yaw: float, pitch: float):
        nonlocal _osc_error_printed
        if osc_client is None:
            return
        try:
            osc_client.send_message(
                "/looking", [1 if is_facing else 0, fx, fy, dist, yaw, pitch, source])
            _osc_error_printed = False
        except OSError as e:
            if not _osc_error_printed:
                print(f"[osc] failed to send: {e}")
                _osc_error_printed = True

    cam = CameraStream(args.camera, args.width, args.height, args.fps,
                        auto_brighten_enabled=args.auto_brighten,
                        target_brightness=args.target_brightness,
                        min_gain=args.min_gain, max_gain=args.max_gain,
                        brighten_slew=args.brighten_slew).start()
    estimator = HeadPoseEstimator(cam.width, cam.height, args.hfov,
                                   min_face_px=args.min_face_px,
                                   max_angular_velocity_deg_s=args.max_angular_velocity,
                                   pose_hold_s=args.pose_hold,
                                   face_model_path=args.face_model,
                                   face_detection_confidence=args.face_detect_confidence,
                                   face_presence_confidence=args.face_presence_confidence,
                                   face_tracking_confidence=args.face_track_confidence)

    body_worker = None
    if args.body_fallback:
        if mp_tasks_python is None or mp_tasks_vision is None:
            print("[body] mediapipe.tasks not available -- continuing "
                  "face-only.")
        elif not os.path.exists(args.pose_model):
            print(f"[body] pose model not found at '{args.pose_model}' -- "
                  "continuing face-only.")
        else:
            try:
                body_estimator = BodyPoseEstimator(
                    cam.width, cam.height, estimator.camera_matrix,
                    args.pose_model, min_body_px=args.min_body_px,
                    visibility_enter=args.visibility_enter,
                    visibility_exit=args.visibility_exit,
                    max_distance_velocity_mm_s=args.max_distance_velocity,
                    max_position_velocity_per_s=args.max_position_velocity,
                    stability_confirm_s=args.stability_confirm)
                body_worker = BodyPoseWorker(cam, body_estimator, target_fps=args.body_fps).start()
            except Exception as e:
                print(f"[body] failed to start PoseLandmarker ({e}) -- "
                      "continuing face-only.")

    def handle_facing(pose: HeadPoseResult):
        print(f"[trigger] eye contact: yaw={pose.yaw_deg:+.1f} "
              f"pitch={pose.pitch_deg:+.1f} dist={pose.approx_distance_mm:.0f}mm "
              f"frustum=({pose.frustum_x:+.2f},{pose.frustum_y:+.2f})")

    def handle_not_facing():
        print("[trigger] gaze released")

    trigger = GazeTrigger(
        enter_yaw_deg=args.enter_yaw, exit_yaw_deg=args.exit_yaw,
        dwell_s=args.dwell,
        on_facing=handle_facing, on_not_facing=handle_not_facing)
    source_hysteresis = SourceHysteresis(
        hold_face_s=args.face_hold, confirm_face_s=args.face_confirm)
    smoother = None if args.no_smoothing else OneEuroSmoother(
        min_cutoff=args.min_cutoff, beta=args.beta)
    body_smoother = None if args.no_smoothing else BodyPoseSmoother(
        min_cutoff=args.body_min_cutoff, beta=args.body_beta)

    # Render at a smaller internal resolution and let cv2.resize (a fast,
    # Eyes render on the GPU now (see GLDisplay) -- a real fragment
    # shader doing this per-pixel math has no trouble at full display
    # resolution, so there's no need for EyeRenderer's CPU-side
    # downscale-then-upscale trick anymore.
    n = args.grid * args.grid
    idle_wanders = [
        SaccadeWander(radius=args.wander_radius,
                      hold_range=(args.wander_hold_min, args.wander_hold_max),
                      move_range=(args.wander_move_min, args.wander_move_max))
        for _ in range(n)
    ]
    avert_wanders = [
        SaccadeWander(radius=args.avert_radius,
                      hold_range=(args.avert_hold_min, args.avert_hold_max),
                      move_range=(args.avert_move_min, args.avert_move_max))
        for _ in range(n)
    ]
    blinkers = [
        BlinkGenerator(interval_range=(args.blink_interval_min, args.blink_interval_max),
                       duration_range=(args.blink_duration_min, args.blink_duration_max))
        for _ in range(n)
    ]
    eye_filt_x: list = [None] * n
    eye_filt_y: list = [None] * n
    was_facing = False
    # Count-based debounce right at the point the eyes decide their mode
    # (tracking / idle wander / averting) -- see ValueDebouncer. Separate
    # from, and downstream of, GazeTrigger's/SourceHysteresis's own
    # time-based debouncing.
    facing_debounce = ValueDebouncer(min_count=args.eye_mode_confirm, initial=False)
    no_track_debounce = ValueDebouncer(min_count=args.eye_mode_confirm, initial=True)

    view_mode = "debug" if args.debug else "eyes"
    show_window = not args.no_window
    gl: Optional[GLDisplay] = None
    if show_window:
        check_gl_available()
        gl = GLDisplay(args.eye_width, args.eye_height, title="eye rig")
        print(f"[view] showing {view_mode} -- press 'd' to toggle, 'q' to quit")

    last_frame_ts = 0.0
    window_start_count, window_start_time = 0, time.monotonic()
    proc_count = 0
    last_report = time.monotonic()
    last_direction_print = time.monotonic()

    try:
        while True:
            frame, ts = cam.latest()
            if frame is None or ts == last_frame_ts:
                time.sleep(0.001)
                continue
            last_frame_ts = ts

            pose_raw = estimator.process(frame)
            now = time.monotonic()
            pose = source_hysteresis.update(pose_raw, now)
            body_pose, _ = body_worker.latest() if body_worker is not None else (None, 0.0)
            if smoother is not None:
                pose = smoother.update(pose, now)
            if body_smoother is not None:
                body_pose = body_smoother.update(body_pose, now)
            trigger.update(pose, now)
            proc_count += 1

            # Single source-of-truth for what's currently tracked --
            # drives BOTH the OSC payload and the in-process eye targeting
            # below (no OSC round-trip needed for the latter anymore).
            if pose is not None:
                source = 1
                yaw, pitch = pose.yaw_deg, pose.pitch_deg
                lx, ly, dist = pose.frustum_x, pose.frustum_y, pose.approx_distance_mm
            elif body_pose is not None:
                source = 2
                yaw, pitch = 0.0, 0.0
                lx, ly, dist = body_pose.frustum_x, body_pose.frustum_y, body_pose.approx_distance_mm
            else:
                source = 0
                yaw = pitch = lx = ly = dist = 0.0

            send_osc(trigger.is_facing, source, lx, ly, dist, yaw, pitch)

            # --- eye targeting (ported from eye_display.py's main loop) ---
            # Debounced by sample count (see ValueDebouncer) right here,
            # at the point the eyes actually decide their mode -- a
            # single still-noisy frame reaching this far used to flip
            # the eyes for one frame and read as a visible twitch, even
            # with all the upstream time-based debouncing already in
            # place.
            is_facing = facing_debounce.update(trigger.is_facing)
            no_track = no_track_debounce.update(source == 0)

            if is_facing and not was_facing and not args.no_avert:
                for w in avert_wanders:
                    w.flee(now)
            was_facing = is_facing

            if is_facing and not args.no_avert:
                raw_targets = [w.update(now) for w in avert_wanders]
            elif no_track and not args.no_wander:
                raw_targets = [w.update(now) for w in idle_wanders]
            else:
                # frustum_y is image-space (-1 top .. +1 bottom), but the
                # shader's stare math shifts the iris toward the OPPOSITE
                # side of whatever offset it's given -- negate here (see
                # eye_display.py's original comment for how this was
                # confirmed).
                raw_targets = [(lx, -ly)] * n

            if args.no_smoothing:
                eye_smoothed = raw_targets
            else:
                eye_smoothed = []
                for i in range(n):
                    tx, ty = raw_targets[i]
                    if eye_filt_x[i] is None:
                        eye_filt_x[i] = OneEuroFilter(now, tx, min_cutoff=args.eye_min_cutoff, beta=args.eye_beta)
                        eye_filt_y[i] = OneEuroFilter(now, ty, min_cutoff=args.eye_min_cutoff, beta=args.eye_beta)
                        eye_smoothed.append((tx, ty))
                    else:
                        eye_smoothed.append((eye_filt_x[i](now, tx), eye_filt_y[i](now, ty)))

            openness = [1.0] * n if args.no_blink else [b.update(now) for b in blinkers]

            if show_window:
                if view_mode == "eyes":
                    gl.render_eyes(args.eye_width, args.eye_height, args.grid,
                                    args.eye_gap, args.look_gain, eye_smoothed, openness)
                else:
                    vis = draw_debug_overlay(frame, pose, body_pose, args.mirror)
                    gl.render_debug(vis)
                key = gl.poll()
                if gl.should_close() or key == 'q':
                    break
                elif key == 'd':
                    view_mode = "debug" if view_mode == "eyes" else "eyes"
                    print(f"[view] switched to {view_mode}")

            now = time.monotonic()
            if now - last_direction_print >= 0.5:
                if pose is not None:
                    label = direction_label(pose.yaw_deg, pose.pitch_deg)
                    print(f"[pose] yaw={pose.yaw_deg:+.1f} "
                          f"pitch={pose.pitch_deg:+.1f} -> {label}")
                elif body_pose is not None:
                    print(f"[pose] no face -- body tracked at "
                          f"frustum=({body_pose.frustum_x:+.2f},{body_pose.frustum_y:+.2f}) "
                          f"dist~{body_pose.approx_distance_mm:.0f}mm "
                          f"[{body_pose.distance_source}]")
                else:
                    print("[pose] no face or body detected")
                last_direction_print = now

            if now - last_report >= 2.0:
                cap_fps = cam.fps_actual(window_start_count, window_start_time)
                proc_fps = proc_count / (now - last_report)
                print(f"[stats] capture ~{cap_fps:.0f}fps | "
                      f"processing ~{proc_fps:.1f}fps")
                proc_count = 0
                last_report = now

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        estimator.close()
        if body_worker is not None:
            body_worker.stop()
        if gl is not None:
            gl.close()


if __name__ == "__main__":
    main()
