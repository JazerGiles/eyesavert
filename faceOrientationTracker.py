#!/usr/bin/env python3
"""
face_orientation_tracker.py

Reads the ELP AR0234 global-shutter USB3 camera at high frame rate, tracks a
person's face, computes:
  1. Where they are in the camera frustum (normalized x, y, and rough z/depth)
  2. Their head orientation (yaw, pitch, roll)
and fires a callback with hysteresis + dwell time when the person is facing
the camera directly ("eye contact" trigger) -- intended to drive the
animatronic eye's AVERTING state transition.

--------------------------------------------------------------------------
IMPORTANT CAMERA NOTES (ELP AR0234, USB3.0 global shutter)
--------------------------------------------------------------------------
- This sensor does NOT expose raw Bayer data over UVC. Available formats at
  high frame rate are MJPEG or YUY2, both at 1280x720:
      MJPEG @ 1280x720 : up to 200fps
      YUY2  @ 1280x720 : up to 200fps (but ~368MB/s -- right at USB3's
                          practical ceiling; frequently drops frames)
  MJPEG is the reliable choice for sustained 200fps. This script uses MJPEG.

- Before running, verify the negotiated mode on your system:
      v4l2-ctl --list-devices
      v4l2-ctl -d /dev/videoX --list-formats-ext
  OpenCV/V4L2 sometimes silently falls back to a lower fps if the requested
  mode isn't supported by the exact backend -- always check
  cap.get(cv2.CAP_PROP_FPS) after opening (this script prints it).

--------------------------------------------------------------------------
FRAME RATE
--------------------------------------------------------------------------
Reliable tracking matters more than raw fps here. 200fps at 720p is right
at the edge of sustained USB3 bandwidth/driver headroom -- that's exactly
where you start seeing intermittent dropped frames, which hurts reliability
more than it helps. Default here is 90fps: comfortably inside the camera's
and USB stack's margin, still far above what MediaPipe can process, and
noticeably steadier in practice. Bump --fps back up later only if you have
a specific reason to (e.g. capturing fast motion for something other than
tracking).

If the preview looks flickery (brightness pulsing, not stutter), that's
very likely AC mains lighting interacting with the camera's exposure
timing, not a bug here -- common with high frame rates that aren't a
clean multiple of the mains frequency (60Hz in the US). Quick diagnostic:
try --fps 60 or --fps 30, and/or point the camera at daylight/a window
instead of indoor lighting -- if the flicker changes or disappears, that
confirms it's lighting, not code. It's mostly cosmetic for the debug
window; MediaPipe reads each frame's raw pixel data regardless of visible
flicker, so it usually doesn't hurt detection reliability.

--------------------------------------------------------------------------
IMPORTANT PROCESSING NOTE
--------------------------------------------------------------------------
The camera can deliver 200fps; a Raspberry Pi running MediaPipe Face
Landmarker + solvePnP cannot process anywhere near that fast (realistically
15-40fps on a Pi 4/5 depending on model complexity and resolution). This
script therefore runs capture and processing as independent threads:
  - The capture thread always overwrites a single shared "latest frame"
    slot -- it never blocks and never queues, so processing always works
    on the most recent frame available (no growing latency backlog).
  - The processing thread runs Face Landmarker on whatever frame is
    current each time it finishes the previous one.

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
Linux / Raspberry Pi:
    pip install opencv-python mediapipe --break-system-packages

macOS (Mac Mini):
    # Python 3.13 + recent mediapipe releases have been unreliable for
    # this project -- mp.solutions has been inconsistently shipped in
    # mediapipe 0.10.2x+, and the newer Tasks API crashed on this machine
    # regardless of delegate settings. mediapipe==0.10.14 on Python 3.11
    # is confirmed working (tested directly), so pin both:
    python3.11 -m venv venv && source venv/bin/activate
    pip install opencv-python "mediapipe==0.10.14" python-osc
    # The first run will prompt for Camera permission (System Settings ->
    # Privacy & Security -> Camera) -- grant it to Terminal/your IDE, not
    # just the script, or capture will silently return empty frames.

No separate model download needed -- mp.solutions.face_mesh ships its
model inside the mediapipe pip package.

--------------------------------------------------------------------------
BODY (CHEST) FALLBACK
--------------------------------------------------------------------------
Face landmarks disappear as soon as someone turns away or backs far
enough off that facial features are too small/occluded to track -- but a
person's torso is often still plainly visible in frame at that distance.
Rather than only reacting to that after the fact (which would mean a
cold start on body tracking right when it's needed, and a beat of dead
air in between), MediaPipe's Tasks-API PoseLandmarker runs continuously
in its own background thread the whole time face tracking is running --
tracking shoulders (11/12) and hips (23/24), a stand-in for "torso" --
off the same always-latest-frame slot the camera capture thread
maintains. So its result is already warm and current the instant the
face is lost, and there's no gap between "tracking face" and "tracking
body": whichever one currently has data is used, frame to frame, with no
dropout in between.

Tracking both shoulders and hips (not just shoulders) specifically
targets turning away from the camera: yaw rotation foreshortens the
shoulder-to-shoulder span a lot (a profile view shrinks it well below its
face-on width), which would otherwise make the distance estimate swing
with rotation, not just with actual distance. Shoulder-mid-to-hip-mid
"torso height" runs along the body's long axis instead, roughly parallel
to that rotation, so it stays far more stable as someone turns in place;
it's used for distance whenever both pairs are visible, with shoulder-
or hip-width as fallbacks when only one pair is. Per-landmark visibility
scores also gate which points are trusted, and the tracked position
averages whichever points currently qualify -- so one shoulder dropping
out of view mid-turn shifts the estimate gradually instead of jumping.

This is still position-only: yaw/pitch/roll are not derivable from
shoulder/hip geometry, so facing/eye-contact (is_facing / dwell /
hysteresis) stays gated on real face orientation and goes false the
instant the face is lost, same as before -- only frustum position +
rough distance keep flowing from the body so whatever's driving the eye
rig can keep visually tracking the person's position instead of snapping
back to center. Running a second model continuously costs real CPU (see
--no-body-fallback below to disable it if that's too much for a given
Pi/model combination).

Requires pose_landmarker.task in the working directory (or --pose-model):
    curl -L -o pose_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
This is optional -- if the model file is missing or the Tasks API isn't
available, the script prints a warning and continues face-only (graceful
degradation, not a hard failure). Disable explicitly with
--no-body-fallback.

Run:
    python3 face_orientation_tracker.py --camera 0 --display

Trigger output: OSC message "/looking <1|0> <frustum_x> <frustum_y>
<approx_distance_mm> <yaw_deg> <pitch_deg> <tracking_source>" streamed
every processed frame (not just on facing/not-facing transitions) to
127.0.0.1:9000 by default (--osc-host / --osc-port to change). The first
value reflects current facing state; tracking_source is 0 (nothing
tracked), 1 (face), or 2 (body/chest fallback only). When tracking_source
is 0, all other values are sent as 0; when it's 2, yaw_deg/pitch_deg are
sent as 0 (unknown) but frustum_x/frustum_y/approx_distance_mm still
reflect the tracked chest position. Point your OSC receiver
(TouchDesigner, Max/MSP, a small custom listener, whatever's driving the
PCA9685) at that address and port.

"""

import argparse
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

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


def check_osc_available():
    if SimpleUDPClient is None:
        print(
            "ERROR: python-osc is not installed.\n"
            "  pip install python-osc"
        )
        sys.exit(1)


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
    sensor doesn't do meaningful auto-exposure/gain adjustment on its own
    (fine for its intended fast-shutter use case, not for a dim room), so
    rather than fight that at the UVC control level (inconsistently
    exposed across backends -- see _default_backend's AVFoundation note,
    and unverified for this sensor on V4L2/the Pi either), this is a
    simple, cheap, cross-platform stand-in.

    The gain actually APPLIED doesn't jump straight to that ideal value
    though -- it slews toward it at a bounded rate (max_gain_change_per_s)
    instead, same reason a real camera's AGC ramps gain/exposure smoothly
    rather than snapping: applying whatever gain a single frame's mean
    brightness implies would flicker visibly on ordinary per-frame noise
    (a shadow passing through the shot, a moment of motion blur), not
    just on a genuine lighting change. A longer slew trades slower
    response to a real lighting change for a lot less flicker -- default
    here (0.5 gain/second) means swinging the full 0.25-4.0 range takes
    several seconds.

    Applied in CameraStream, upstream of everything (display and both CV
    pipelines) -- MediaPipe needs to actually see the face/body in a dim
    room just as much as a human watching the preview would."""

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
# Generic average adult face geometry (arbitrary units, mm-ish). This is the
# standard 6-point model used in most solvePnP-based head pose tutorials.
# It's approximate -- fine for a facing/not-facing gaze trigger, not for
# precision measurement.
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye, outer corner
    (225.0, 170.0, -135.0),   # right eye, outer corner
    (-150.0, -150.0, -125.0), # mouth, left corner
    (150.0, -150.0, -125.0),  # mouth, right corner
], dtype=np.float64)

# Corresponding MediaPipe FaceMesh/FaceLandmarker landmark indices
# (468/478-point mesh).
_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    # extra points used for depth/frustum estimation
    "left_eye_inner": 133,
    "right_eye_inner": 362,
}

# BlazePose (MediaPipe PoseLandmarker) 33-point landmark indices used for
# the chest/body fallback below.
_POSE_LANDMARK_IDX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}


def build_camera_matrix(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Approximate intrinsics from the lens's stated horizontal FOV.
    Good enough for head-pose gating; run a proper cv2.calibrateCamera
    checkerboard calibration if you need precise angles."""
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = fx  # assume square pixels
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
    bbox: tuple        # (x0, y0, x1, y1) in pixels
    keypoints: tuple    # the 6 (x,y) pixel points used for solvePnP
    nose_line: tuple    # ((x0,y0), (x1,y1)) -- nose tip to a point
                         # projected 100mm out along the face's forward
                         # axis, i.e. an arrow showing which way it's
                         # pointing. This is the clearest single thing to
                         # look at to confirm direction is being tracked.


class OneEuroFilter:
    """One Euro Filter (Casiez, Roussel, Vogel 2012). Adapts its cutoff
    frequency to signal speed: near-stationary input gets heavily smoothed
    (kills jitter), fast movement gets smoothed less (kills lag) -- unlike a
    fixed-alpha EMA, which has to pick one tradeoff for both cases.

    min_cutoff: cutoff frequency when the signal is still. Lower = smoother
                at rest, but a bit more lag on slow movements.
    beta:       how much speed increases the cutoff. Higher = snappier
                response to fast head turns, at the cost of more jitter
                during them.
    """

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
    """Applies an independent OneEuroFilter per pose channel (each field
    has its own dynamics -- yaw whips around faster than distance does)."""

    _FIELDS = ("yaw_deg", "pitch_deg", "roll_deg",
               "frustum_x", "frustum_y", "approx_distance_mm")

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.07):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, OneEuroFilter] = {}

    def update(self, pose: Optional[HeadPoseResult], t: float) -> Optional[HeadPoseResult]:
        if pose is None:
            self._filters = {}  # face lost -- don't smooth across the gap
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
            # These stay raw -- only used for the debug overlay, not the
            # trigger logic, so no need to smooth them.
            bbox=pose.bbox, keypoints=pose.keypoints, nose_line=pose.nose_line,
        )


class HeadPoseEstimator:
    REAL_INTEROCULAR_MM = 63.0  # average adult interpupillary distance

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
        self.dist_coeffs = np.zeros((4, 1))  # assume "no distortion" lens
        # Reject detections smaller than this (in pixels, face bbox width).
        # A tiny/far face gives noisy landmark positions, which produces
        # noisy/wrong solvePnP angles -- better to report "no face" than a
        # confident-looking but garbage yaw/pitch.
        self.min_face_px = min_face_px

        # solvePnP with only 6 points has a known pose-ambiguity problem:
        # at steep angles (near-profile), the 2D landmarks can be nearly
        # as well explained by a wildly different 3D rotation, and
        # solvePnP can flip to it frame-to-frame ("which way is the nose
        # pointed" going haywire) even though the face itself is turning
        # smoothly. Gate on angular velocity: reject a new rotation that
        # implies turning faster than a head plausibly can, and hold the
        # last known-good rotation briefly instead (position/distance
        # below are NOT gated -- they come straight from 2D landmarks,
        # not solvePnP, so they're not subject to this ambiguity and stay
        # fresh every frame even while rotation is being held).
        # pose_hold_s mirrors SourceHysteresis.hold_face_s -- if rejections
        # persist past it, give up holding and accept whatever solvePnP is
        # producing rather than freezing the rotation forever.
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
        # delegate explicitly set. That's why solutions was used at all
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
        """Geodesic angular distance between two rotation matrices, in
        degrees -- the true angle you'd have to rotate by to go from one
        orientation to the other, regardless of axis."""
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
            rgb.flags.writeable = False  # perf hint, mediapipe convention
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
            # Floor elapsed so back-to-back calls (elapsed near 0) don't
            # make the allowed angle collapse to ~0 and reject everything.
            max_allowed_deg = self.max_angular_velocity_deg_s * max(elapsed, 1.0 / 120.0)
            if angle_diff > max_allowed_deg:
                if self._reject_since is None:
                    self._reject_since = now
                if now - self._reject_since < self.pose_hold_s:
                    use_rot_mat, use_rvec = self._last_rot_mat, self._last_rvec
                else:
                    self._reject_since = None  # held too long -- accept reality
            else:
                self._reject_since = None

        pose_mat = cv2.hconcat([use_rot_mat, tvec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [float(a) for a in euler.flatten()]

        # cv2.decomposeProjectionMatrix has a well-known quirk (common
        # across solvePnP-based head-pose code): with this model's
        # convention -- nose tip at Z=0, other features set back at
        # negative Z -- "facing the camera" decomposes to pitch near
        # +-180 degrees instead of the intuitive 0. Unwrap it back into a
        # centered range so 0 = facing forward, matching yaw's convention.
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        # Sign flip: this model/solvePnP combination produces pitch
        # negative when tilting the head up and positive when tilting
        # down -- backwards from the conventional pitch sign (up =
        # positive, down = negative, matching direction_label() below and
        # typical aerospace/robotics convention). Fixing the sign here, at
        # the source, rather than in the label mapping, so every consumer
        # of pitch_deg (display, trigger, UDP payload) agrees.
        pitch = -pitch

        # A point 100mm in front of the nose along the model's forward
        # (+Z) axis, projected into the image -- this is what draws the
        # direction arrow. Nose tip is Z=0 and the other features (chin,
        # eyes, mouth) sit at negative Z (set back into the face), so +Z
        # is "outward, away from the face" -- the direction someone's
        # actually facing.
        nose_forward_3d = np.array([[0.0, 0.0, 100.0]])
        nose_end_2d, _ = cv2.projectPoints(
            nose_forward_3d, use_rvec, tvec, self.camera_matrix, self.dist_coeffs)
        nose_end = (int(nose_end_2d[0, 0, 0]), int(nose_end_2d[0, 0, 1]))
        nose_start = (int(image_points[0][0]), int(image_points[0][1]))

        # Frustum position: normalized center of all face landmarks
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
            return None  # too small/far to trust -- see min_face_px note above

        # Rough distance from interocular pixel distance (similar-triangles).
        lx, ly = px(_LANDMARK_IDX["left_eye_outer"])
        rx, ry = px(_LANDMARK_IDX["right_eye_outer"])
        pixel_interocular = math.hypot(rx - lx, ry - ly)
        fx = self.camera_matrix[0, 0]
        approx_distance_mm = (
            (self.REAL_INTEROCULAR_MM * fx) / pixel_interocular
            if pixel_interocular > 1e-3 else float("inf")
        )

        # Commit the accepted rotation only now that the frame has fully
        # passed all rejection checks (min_face_px above included) --
        # don't let a frame we're about to discard anyway pollute the
        # held state.
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
# Body (torso) fallback -- position-only tracking via shoulder + hip
# landmarks, used to keep reporting where someone is once they're too far/
# turned away for face tracking to work. See BODY (CHEST) FALLBACK in the
# module docstring for the rationale behind running this continuously in
# its own thread rather than only starting it once the face is lost.
#
# Turning away from the camera is exactly the case shoulders-only handles
# worst: yaw rotation (spinning left/right) foreshortens the *horizontal*
# shoulder-to-shoulder span a lot (a profile view can shrink it to a
# fraction of its face-on width), which used to make the shoulder-width
# distance estimate swing wildly with rotation, not just with actual
# distance. Two things fix that:
#   1. Track hips (23/24) as well as shoulders (11/12), and prefer the
#      *vertical* shoulder-mid-to-hip-mid span (torso height) for the
#      distance estimate whenever both are visible -- that span runs along
#      the body's long axis, roughly parallel to the camera's yaw rotation
#      axis, so it barely foreshortens as someone turns in place. Shoulder
#      width (and a hip-width fallback) are kept as fallbacks for when only
#      one pair is visible.
#   2. Use MediaPipe's per-landmark visibility score to only trust points
#      that are actually likely visible, and average whichever of the 4
#      torso points currently qualify (weighted by visibility) for the
#      position estimate -- so losing one shoulder mid-turn (occluded by
#      the body itself) doesn't make the tracked position jump or drop out.
# ==========================================================================
@dataclass
class BodyPoseResult:
    frustum_x: float          # -1 (left edge) .. +1 (right edge)
    frustum_y: float          # -1 (top edge)  .. +1 (bottom edge)
    approx_distance_mm: float
    distance_source: str      # "torso" | "shoulders" | "hips" -- which pair
                               # of points the distance estimate came from,
                               # mainly for the debug overlay/logs.
    chest_point: tuple        # (x, y) in pixels -- visibility-weighted
                               # centroid of whichever torso points are
                               # currently trusted (not just shoulders)
    shoulder_points: Optional[tuple]  # ((lx,ly), (rx,ry)) px, or None if
                                        # either shoulder isn't visible enough
    hip_points: Optional[tuple]        # ((lx,ly), (rx,ry)) px, or None if
                                        # either hip isn't visible enough


class BodyPoseEstimator:
    # Average adult biacromial (shoulder-to-shoulder) breadth and hip
    # breadth, and shoulder-mid-to-hip-mid torso height. All three are
    # cruder than the face estimator's interocular distance (varies more
    # with build/clothing/posture), but this only ever runs as a position
    # fallback, never for anything precision-sensitive. Torso height is
    # preferred when available -- see the module-level comment above.
    REAL_SHOULDER_MM = 400.0
    REAL_HIP_MM = 320.0
    REAL_TORSO_MM = 500.0

    # Rough neck+lower-face height above the shoulder line -- used to
    # estimate where the FACE would be from shoulder/hip position, rather
    # than reporting the torso centroid's own (much lower) height. Without
    # this, frustum_y jumps downward the instant tracking hands off from
    # face to body (the chest is genuinely well below the face), which
    # reads as a sudden vertical snap for whatever's consuming frustum_y
    # (e.g. eye_display.py). HEAD_OFFSET_FROM_HIP_MM only applies when
    # shoulders aren't visible and hips are the anchor instead.
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

        # Tasks API (not legacy mp.solutions.pose) with delegate=CPU --
        # mirrors camera_body_test.py, which confirmed this combination
        # works reliably on macOS; the legacy solutions module is the one
        # that's been inconsistently shipped in recent mediapipe releases,
        # and the Tasks API's GPU-service crash (see HeadPoseEstimator's
        # comment on why face uses the legacy API) is specifically a GPU
        # delegate issue -- CPU delegate sidesteps it.
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
            return None  # need at least one full pair to measure anything

        shoulder_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0) if shoulders_ok else None
        hip_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0) if hips_ok else None

        fx = self.camera_matrix[0, 0]

        # Prefer torso height (yaw-rotation-invariant) whenever both pairs
        # are visible; otherwise fall back to whichever single pair we have.
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

        # Position: average whichever of shoulder-mid/hip-mid are available,
        # so losing one pair mid-turn shifts the centroid gradually rather
        # than snapping or dropping out.
        centers = [c for c in (shoulder_mid, hip_mid) if c is not None]
        cx_px = sum(c[0] for c in centers) / len(centers)
        cy_px = sum(c[1] for c in centers) / len(centers)
        frustum_x = (cx_px / w) * 2.0 - 1.0

        # frustum_y specifically approximates FACE height, not the torso
        # centroid's height -- see HEAD_OFFSET_FROM_SHOULDER_MM above.
        # Anchored on the shoulder line when visible (closer to the head,
        # so the estimate is less sensitive to the offset being only
        # approximate); falls back to the hip line with a larger offset.
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
    """Same OneEuroFilter-per-channel approach as OneEuroSmoother, applied
    to the body/chest fallback's (smaller) field set. Kept as a separate
    class rather than generalizing OneEuroSmoother -- these track two
    different dataclasses with different fields, and there are only ever
    these two concrete uses."""

    _FIELDS = ("frustum_x", "frustum_y", "approx_distance_mm")

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.07):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, OneEuroFilter] = {}

    def update(self, pose: Optional[BodyPoseResult], t: float) -> Optional[BodyPoseResult]:
        if pose is None:
            self._filters = {}  # body lost -- don't smooth across the gap
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
    """Runs a BodyPoseEstimator continuously in its own thread, reading off
    the same always-latest-frame slot the camera capture thread maintains
    (see CameraStream) -- so body tracking is already warm and current the
    instant face tracking drops out, instead of starting cold only once the
    face is lost. Mirrors CameraStream's never-blocks/never-queues pattern:
    the main thread always reads whatever the most recent processed result
    is, with no backlog.

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
        """Returns (BodyPoseResult | None, timestamp) for the most recently
        processed body pose."""
        with self._lock:
            return self._result, self._timestamp

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._estimator.close()


# ==========================================================================
# Face/body source hysteresis -- same enter/exit-dwell pattern as
# GazeTrigger below, applied to which tracking source is treated as
# active. Without this, a single missed frame (motion blur, a fast head
# turn, brief occlusion) right at the edge of face detectability flips
# tracking_source to body and back on consecutive frames -- each flip is
# itself a small position jump (see BodyPoseEstimator's face-height
# approximation: close, but not pixel-identical to the real face
# position), so rapid flicker there reads as jitter downstream even
# though each individual position estimate is fine on its own.
# ==========================================================================
class SourceHysteresis:
    def __init__(self, hold_face_s: float = 0.3, confirm_face_s: float = 0.15):
        # hold_face_s: once face has been active, keep reporting the last
        # known face pose for this long after it drops out before
        # actually handing off to body -- rides out brief dropouts.
        # confirm_face_s: once face reappears while body/none is active,
        # require it to stay present this long before trusting it as
        # active again -- avoids flickering back on a single spurious hit.
        self.hold_face_s = hold_face_s
        self.confirm_face_s = confirm_face_s
        self._using_face = False
        self._last_good_pose: Optional[HeadPoseResult] = None
        self._face_seen_since: Optional[float] = None
        self._face_lost_at: Optional[float] = None

    def update(self, pose: Optional[HeadPoseResult], now: float) -> Optional[HeadPoseResult]:
        """pose: this frame's raw face read (None if no face detected).
        Returns the HeadPoseResult to treat as the active face pose this
        frame (may be held over from a prior frame), or None if face
        should be considered unavailable right now."""
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
                return self._last_good_pose  # brief hold-over, riding out the gap
            self._using_face = False
            self._last_good_pose = None
        return None


# ==========================================================================
# Gaze trigger: hysteresis + dwell time, mirrors the eye rig's state-machine
# pattern (avoid flicker at the threshold boundary, don't fire on a single
# noisy frame).
# ==========================================================================
class GazeTrigger:
    def __init__(self,
                 enter_yaw_deg: float = 40.0, exit_yaw_deg: float = 50.0,
                 dwell_s: float = 0.25,
                 on_facing=None, on_not_facing=None):
        # enter_yaw is the tighter threshold to START being "facing";
        # exit_yaw is the looser threshold to STOP -- this gap is the
        # hysteresis band that prevents flicker right at the boundary.
        # Only yaw gates the trigger -- pitch is still computed and shown
        # in the display/logs, it just doesn't affect facing/not-facing.
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
            # Already facing: only drop out once we exceed the wider exit
            # band (hysteresis), not the instant we cross enter threshold.
            if not within_exit:
                self._is_facing = False
                self._facing_since = None
                self.on_not_facing()


# ==========================================================================
# Human-readable direction label. Signs confirmed against real hardware:
# yaw matches LEFT/RIGHT as-is; pitch_deg is corrected to the conventional
# up=positive/down=negative sign at the source (see the sign flip in
# HeadPoseEstimator.process()), so this mapping needs no compensation.
# ==========================================================================
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


# ==========================================================================
# Main
# ==========================================================================
def main():
    check_osc_available()

    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-auto-brighten", dest="auto_brighten",
                     action="store_false",
                     help="Disable the software auto-brightness (on by "
                          "default -- the ELP AR0234 doesn't meaningfully "
                          "auto-expose/gain-adjust on its own in low "
                          "light). Bipolar -- brightens dark scenes AND "
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
                     help="Lower is more reliable (less USB/driver headroom "
                          "pressure); 90 comfortably outpaces what MediaPipe "
                          "can process anyway. See FRAME RATE note above.")
    ap.add_argument("--min-cutoff", type=float, default=0.8,
                     help="One Euro filter: cutoff freq when signal is still. "
                          "Lower = smoother at rest, slightly more lag.")
    ap.add_argument("--beta", type=float, default=0.07,
                     help="One Euro filter: how much speed reduces smoothing. "
                          "Higher = snappier during fast head turns, more "
                          "jitter during them.")
    ap.add_argument("--no-smoothing", action="store_true",
                     help="Use raw per-frame pose estimates, no filtering.")
    ap.add_argument("--hfov", type=float, default=112.0,
                     help="Lens horizontal FOV in degrees (110/112 per ELP spec)")
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
                          "pixels (too small/far to trust reliably).")
    ap.add_argument("--max-angular-velocity", type=float, default=500.0,
                     help="Degrees/second -- reject a solvePnP rotation "
                          "that implies turning faster than this (a real "
                          "head can't), holding the last known-good "
                          "rotation briefly instead. Targets solvePnP's "
                          "known 6-point pose ambiguity at steep angles, "
                          "where it can flip to a wildly different but "
                          "similarly-plausible orientation frame to "
                          "frame. Position/distance aren't affected -- "
                          "only yaw/pitch/roll and the nose-direction "
                          "arrow are gated.")
    ap.add_argument("--pose-hold", type=float, default=0.2,
                     help="Seconds to hold the last known-good rotation "
                          "when solvePnP's output is rejected as an "
                          "implausible jump, before giving up and "
                          "accepting whatever it's producing anyway "
                          "(mirrors --face-hold's give-up-eventually "
                          "philosophy).")
    ap.add_argument("--no-body-fallback", dest="body_fallback",
                     action="store_false",
                     help="Disable the chest/shoulder body-position fallback "
                          "(runs continuously alongside face tracking by "
                          "default -- see BODY (CHEST) FALLBACK above). "
                          "Disable if PoseLandmarker's extra CPU load hurts "
                          "framerate on a given Pi/model combination.")
    ap.add_argument("--pose-model", type=str, default="pose_landmarker.task",
                     help="Path to the MediaPipe Tasks PoseLandmarker model "
                          "file used for the body fallback.")
    ap.add_argument("--min-body-px", type=int, default=40,
                     help="Reject body detections whose measured span "
                          "(torso height, or shoulder/hip width as a "
                          "fallback -- whichever the distance estimate is "
                          "using) is narrower than this many pixels (too "
                          "small/far/edge-on to trust reliably).")
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
                          "this to START being trusted (see "
                          "BodyPoseEstimator's hysteresis comment -- "
                          "prevents shoulders_ok/hips_ok flapping on a "
                          "noisy visibility score).")
    ap.add_argument("--visibility-exit", type=float,
                     default=BodyPoseEstimator.DEFAULT_VISIBILITY_EXIT,
                     help="Shoulder/hip pair visibility must fall below "
                          "this to STOP being trusted. Should be well "
                          "below --visibility-enter -- the gap is the "
                          "hysteresis band. Widen the gap (raise enter, "
                          "lower exit) if shoulders/hips are still "
                          "bouncing.")
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
                     help="Yaw threshold (degrees) to START being 'facing'. "
                          "Only yaw gates the trigger -- pitch is shown "
                          "but not used for facing/not-facing.")
    ap.add_argument("--exit-yaw", type=float, default=50.0,
                     help="Yaw threshold (degrees) to STOP being 'facing'. "
                          "Should be looser than --enter-yaw -- the gap is "
                          "the hysteresis band that prevents flicker.")
    ap.add_argument("--dwell", type=float, default=0.25,
                     help="Seconds within the enter threshold before the "
                          "trigger actually fires (debounces a single "
                          "noisy frame).")
    ap.add_argument("--face-hold", type=float, default=0.3,
                     help="Seconds to keep reporting the last known face "
                          "pose after face is lost before actually handing "
                          "off to the body/chest fallback -- rides out a "
                          "brief dropout (motion blur, fast turn) instead "
                          "of flickering tracking_source back and forth.")
    ap.add_argument("--face-confirm", type=float, default=0.15,
                     help="Seconds face must stay continuously present "
                          "before being trusted as active again once "
                          "body/none was active -- avoids flickering back "
                          "on a single spurious detection.")
    ap.add_argument("--osc-host", type=str, default="127.0.0.1",
                     help="OSC destination host. Defaults to localhost -- "
                          "change this if the OSC receiver moves to a "
                          "separate device on the network.")
    ap.add_argument("--osc-port", type=int, default=9000,
                     help="OSC destination port. Match whatever your "
                          "receiver (TouchDesigner, Max/MSP, a custom "
                          "script, etc.) is listening on.")
    ap.add_argument("--display", action="store_true",
                     help="Show a debug window with landmarks + pose overlay")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false",
                     help="Disable selfie-style mirroring of the debug "
                          "display (mirrored by default). Only affects "
                          "the preview window, not the underlying "
                          "yaw/pitch/frustum math used for triggering.")
    args = ap.parse_args()

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
            print("[body] mediapipe.tasks not available in this install -- "
                  "continuing face-only. (pip install/upgrade mediapipe, or "
                  "pass --no-body-fallback to silence this.)")
        elif not os.path.exists(args.pose_model):
            print(f"[body] pose model not found at '{args.pose_model}' -- "
                  "continuing face-only. See BODY (CHEST) FALLBACK in the "
                  "module docstring for the download command, or pass "
                  "--no-body-fallback to silence this.")
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

    # --- TRANSPORT TO THE SERVO CONTROLLER: OSC over localhost ---
    # /looking <1|0> <frustum_x> <frustum_y> <approx_distance_mm> <yaw_deg> <pitch_deg> <tracking_source>
    # Sent every processed frame (streaming), not just on facing/not-facing
    # transitions -- whatever's driving the eye rig (TouchDesigner, Max/MSP,
    # a custom OSC receiver feeding the PCA9685, etc.) gets a live value
    # every frame, with the first element reflecting current facing state.
    # tracking_source: 0 = nothing tracked, 1 = face, 2 = body/chest fallback.
    # Sent to 127.0.0.1 by default; change --osc-host if that ever moves to
    # a separate device on the network.
    osc_client = SimpleUDPClient(args.osc_host, args.osc_port)
    _osc_error_printed = False

    def stream_osc(is_facing: bool, pose: Optional[HeadPoseResult],
                    body_pose: Optional[BodyPoseResult]):
        # Synchronous, not threaded -- localhost sends are effectively
        # instant (no DNS involved, unlike the earlier .local hostname
        # issue), and at this call rate (once per processed frame) a
        # thread per call would just be wasted overhead.
        nonlocal _osc_error_printed
        if pose is not None:
            source = 1
            yaw, pitch = pose.yaw_deg, pose.pitch_deg
            fx, fy, dist = pose.frustum_x, pose.frustum_y, pose.approx_distance_mm
        elif body_pose is not None:
            # Position only -- yaw/pitch aren't derivable from shoulder
            # geometry, so they're reported as 0 (unknown), not a guess.
            source = 2
            yaw, pitch = 0.0, 0.0
            fx, fy, dist = (body_pose.frustum_x, body_pose.frustum_y,
                             body_pose.approx_distance_mm)
        else:
            source = 0
            yaw = pitch = fx = fy = dist = 0.0
        try:
            osc_client.send_message(
                "/looking", [1 if is_facing else 0, fx, fy, dist, yaw, pitch, source])
            _osc_error_printed = False
        except OSError as e:
            if not _osc_error_printed:  # don't spam the console every frame
                print(f"[trigger] failed to send OSC message: {e}")
                _osc_error_printed = True

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
            # Debounce which source is active (see SourceHysteresis) --
            # rides out a brief face dropout instead of immediately
            # handing off to body and back on consecutive frames.
            pose = source_hysteresis.update(pose_raw, now)
            # Body tracking runs continuously in its own thread (see
            # BodyPoseWorker) -- read whatever its latest result is, every
            # frame, regardless of whether the face was just found. That's
            # what makes the handoff gapless: it's already warm, not
            # started reactively once the face disappears.
            body_pose, _ = body_worker.latest() if body_worker is not None else (None, 0.0)
            if smoother is not None:
                pose = smoother.update(pose, now)
            if body_smoother is not None:
                body_pose = body_smoother.update(body_pose, now)
            trigger.update(pose, now)
            stream_osc(trigger.is_facing, pose, body_pose)
            proc_count += 1

            if args.display:
                # Mirror for display only (selfie-style), so it feels
                # natural to look at -- the underlying yaw/pitch/frustum
                # math above stays in true (unmirrored) camera space, since
                # that's what should drive the physical eye rig.
                vis = cv2.flip(frame, 1) if args.mirror else frame.copy()
                vw = vis.shape[1]

                def mx(x):
                    return vw - x if args.mirror else x

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
                    # No face -- draw the body/torso fallback instead, in a
                    # different color so it's obvious at a glance which
                    # source is currently driving the OSC stream. Either
                    # point pair may be absent (occluded/turned away), so
                    # only draw whichever is currently available.
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
                cv2.imshow("face orientation tracker", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()