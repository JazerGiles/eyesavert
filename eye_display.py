#!/usr/bin/env python3
"""
eye_display.py

Standalone eye-graphic renderer, ported from the Max/MSP jit.gen shader
that used to drive this. Runs as its own process/window, decoupled from
face_orientation_tracker.py via OSC -- it listens for the "/looking"
messages that script already streams and points the eye at wherever the
tracked person currently is.

Deliberately has NO mediapipe/cv2-camera dependency -- just an OSC
listener + a numpy-rendered window -- so it stays light on a Pi 5's 4GB of
RAM regardless of what the tracker process is doing.

--------------------------------------------------------------------------
WHAT DRIVES THE EYE
--------------------------------------------------------------------------
"/looking" carries: <is_facing> <frustum_x> <frustum_y> <approx_distance_mm>
<yaw_deg> <pitch_deg> <tracking_source>, where tracking_source is 0 (no
one tracked), 1 (face), or 2 (body/chest fallback, position-only -- see
BODY (CHEST) FALLBACK in face_orientation_tracker.py).

frustum_x/frustum_y (where the person is in the camera's frame, -1..+1)
is what points the eyes here, not yaw/pitch -- that's the one signal
that's available continuously from BOTH tracking sources, so the eyes
keep tracking the person's on-screen position through the face/body
handoff with no gap, matching the tracker's own no-dropout design. When
tracking_source is 0 (nobody tracked, or the tracker's gone quiet), the
eyes switch to an idle saccade wander instead of freezing -- see
SaccadeWander below. --no-wander disables that and just rests at center.

--------------------------------------------------------------------------
AVERTING
--------------------------------------------------------------------------
is_facing is the tracker's own hysteresis+dwell-debounced "direct eye
contact" signal (see GazeTrigger in face_orientation_tracker.py) -- the
moment it goes true, every pair of eyes darts away to its OWN
independent random spot (SaccadeWander.flee(), tuned faster/wider than
idle wander via --avert-*), instead of all nine pairs fleeing to the
same place. While eye contact holds, each pair keeps re-darting
independently on its own schedule, so the group reads as scattering
rather than moving as one. The instant is_facing goes false again, all
pairs resume tracking the same shared position -- each pair's own
smoothing filter eases it back independently, so they visibly regroup
rather than snapping into sync. --no-avert disables this and pairs just
keep tracking through direct eye contact.

Blinking is independent of tracking/averting entirely -- each pair
blinks on its own random schedule (both eyes in a pair always blink
together) the whole time, via BlinkGenerator. --no-blink disables it.

--------------------------------------------------------------------------
SHADER PORT NOTES
--------------------------------------------------------------------------
The original Max jit.gen shader tiled a grid of eye pairs (zoom=9, nine
pairs) that either wandered independently (driven by a noise texture
sampled per tile, "which"=0) or all snapped to a single manually-supplied
"look" vector ("which"=1), and blinked off that same noise texture's
alpha channel (nnz.w < -.4). This port doesn't have an equivalent input
for that noise texture, so both the wander and the blink are modeled
directly here instead of sourced from noise -- SaccadeWander and
BlinkGenerator above, not a port of the shader's noise sampling.
"which" itself isn't a fixed mode here either: every pair tracks the
shared "look" position (which=1 behavior) UNTIL is_facing goes true, at
which point that pair switches to its own independent wander target
(which=0-like, but each pair's own target instead of one shared one) --
see AVERTING above.

The eye math itself (sclera/iris/pupil as nested smoothstep distance
fields, iris+pupil offset by "stare" to fake looking in a direction) is
ported directly from the shader's arithmetic. One spot is a guess rather
than a certainty: `gtp(iris, .5)` in the original -- Jitter's "greater
than pass" op, approximated here as a hard threshold (iris_soft > 0.5)
before compositing the iris color, which is consistent with how the
result is used immediately afterward (`mix(..., iris>.5)`). If the iris
edge looks different from the Max version, that's the spot to revisit.

Run (with face_orientation_tracker.py already running and streaming to
the same host/port):
    python3 eye_display.py --port 9000
"""

import argparse
import math
import random
import threading
import time
from typing import Optional

import cv2
import numpy as np

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


# ==========================================================================
# Minimal One Euro Filter -- duplicated (not imported) from
# face_orientation_tracker.py deliberately: this script intentionally
# avoids importing that module so it never pulls in mediapipe/cv2-capture
# as a dependency just for a ~15-line filter. See Casiez, Roussel, Vogel
# 2012; same rationale as the tracker's copy (smooths jitter at rest,
# tracks fast motion with less lag than a fixed-alpha EMA).
# ==========================================================================
class OneEuroFilter:
    def __init__(self, t0: float, x0: float, min_cutoff: float = 1.0,
                 beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
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


# ==========================================================================
# OSC receiver -- mirrors the tracker's CameraStream pattern: a background
# thread always overwrites a single "latest" slot, the render loop always
# reads whatever's most current, never blocks/queues.
# ==========================================================================
class LookReceiver:
    def __init__(self, host: str, port: int):
        self._lock = threading.Lock()
        self._is_facing = False
        self._frustum_x = 0.0
        self._frustum_y = 0.0
        self._tracking_source = 0
        self._last_update = 0.0

        dispatcher = Dispatcher()
        dispatcher.map("/looking", self._on_looking)
        self._server = BlockingOSCUDPServer((host, port), dispatcher)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _on_looking(self, address, *args):
        if len(args) < 7:
            return
        is_facing, fx, fy, _dist, _yaw, _pitch, source = args[:7]
        with self._lock:
            self._is_facing = bool(is_facing)
            self._frustum_x = float(fx)
            self._frustum_y = float(fy)
            self._tracking_source = int(source)
            self._last_update = time.monotonic()

    def start(self):
        self._thread.start()
        return self

    def latest(self, stale_after_s: float = 1.0):
        """Returns (frustum_x, frustum_y, tracking_source, is_facing).
        Falls back to (0, 0, 0, False) if nothing tracked yet or the
        tracker's gone quiet (e.g. the process died) -- rest/idle rather
        than freeze on a stale position or get stuck averting forever."""
        with self._lock:
            if time.monotonic() - self._last_update > stale_after_s:
                return 0.0, 0.0, 0, False
            return self._frustum_x, self._frustum_y, self._tracking_source, self._is_facing

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=1.0)


# ==========================================================================
# Idle wander -- stand-in for the shader's noise-texture-driven "which=0"
# branch (see SHADER PORT NOTES), used whenever nobody's tracked. Real
# noise (Perlin/Simplex) would mean per-frame gradient hashing for not
# much payoff here; eyes at rest don't actually drift continuously, they
# fixate on a spot and then saccade to a new one -- so this is just a
# tiny state machine (hold / move, a couple of numbers and a timer) that's
# cheaper than any noise function AND a closer likeness of the real thing.
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
        # sqrt(uniform) so points are uniform over the disk's *area*, not
        # bunched up near the center.
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
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep ease -- fast in, soft landing
        x = self._from[0] + (self._to[0] - self._from[0]) * eased
        y = self._from[1] + (self._to[1] - self._from[1]) * eased
        return (x, y)

    def flee(self, now: float):
        """Force an immediate, fast move to a fresh random target,
        overriding whatever hold/move state this instance was in --
        used for averting (see AVERTING below), where the eyes need to
        dart away the instant eye contact is made, not whenever this
        particular instance's own timer next happens to expire."""
        self._from = self._to
        self._to = self._new_target()
        self._state = "move"
        self._duration = self._rng.uniform(*self.move_range)
        self._state_start = now


# ==========================================================================
# Blink -- stand-in for the shader's other noise-texture-driven bit
# (blink was `nnz.w < -.4` off that same missing "in2" texture). Same
# philosophy as SaccadeWander: don't fake a noise signal, just model the
# actual behavior directly -- eyes rest open, then briefly and quickly
# close and reopen at random intervals. One instance per PAIR (both eyes
# in a pair always blink together, matching a real pair of eyes; each
# pair blinks on its own independent schedule).
# ==========================================================================
class BlinkGenerator:
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
        """Returns eyelid openness in [0, 1] -- 1 = fully open, 0 = fully
        closed."""
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

        # A single smooth close-then-open dip across the blink window,
        # bottoming out at 0 at the midpoint -- cheap (one sin, only while
        # actually blinking) and reads as a natural blink rather than a
        # linear/mechanical one.
        return 1.0 - math.sin(math.pi * max(0.0, min(1.0, t)))


# ==========================================================================
# Eye renderer -- ported from the jit.gen shader (see module docstring).
# Vectorized with numpy over the whole canvas; the parts that don't depend
# on "stare" (the sclera shape) are precomputed once in __init__.
# ==========================================================================
def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class EyeRenderer:
    """Renders a grid x grid tiling of eye PAIRS (grid=3 -> nine pairs,
    matching the original Max patch's zoom(9)). Each pair has its own
    independent gaze target and blink state (both eyes within a pair
    always share both -- a pair looks/blinks as one unit -- but different
    pairs can look different directions and blink on different
    schedules, which is what averting needs).

    This does not replicate the original's wrap()-based tiling algebra
    (that also relied on a mirrored-alternate-tile trick to fake pairs
    from a singly-repeating pattern, which needs Jitter's exact wrap/gtp
    semantics to get pixel-right, and isn't something worth reverse-
    engineering blind). Instead it tiles the canvas into a grid x grid
    array of cells via straightforward arithmetic on the pixel
    coordinates, and renders two eyes per cell directly -- same visual
    result, simpler to reason about and to retune (--grid, --eye-gap).

    width/height need not be equal. Each eye's own shape is scaled
    relative to the cell HEIGHT only (row_height = height/grid), not the
    column width -- so a wider-than-tall canvas adds horizontal room
    *between* pair columns (letting --eye-gap grow without the two eyes
    in a pair, or neighboring pairs, overlapping) without stretching the
    eyes themselves into ellipses. A too-narrow canvas is exactly what
    caused eyes to overlap in the first place: a square canvas makes each
    column only as wide as a row is tall, which isn't enough horizontal
    room for two side-by-side eyes plus a gap.
    """

    # BGR (OpenCV channel order) -- vec(.1, .4, .2) in the original (RGB).
    IRIS_COLOR_BGR = np.array([0.2, 0.4, 0.1])

    def __init__(self, width: int, height: int, grid: int = 3, eye_gap: float = 0.35):
        self.width, self.height = width, height
        self.grid = grid
        self.num_pairs = grid * grid
        self.eye_gap = eye_gap

        ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)

        col_width = width / grid
        row_height = height / grid  # also the fixed pixels-per-unit scale for u AND v

        col_idx = np.minimum((xs / col_width).astype(np.int64), grid - 1)
        row_idx = np.minimum((ys / row_height).astype(np.int64), grid - 1)
        self._pair_idx = row_idx * grid + col_idx

        pair_center_x = (col_idx + 0.5) * col_width
        pair_center_y = (row_idx + 0.5) * row_height

        # Cell-local coordinates, both scaled by row_height so eye shape
        # stays consistent regardless of column width -- v ranges roughly
        # [-0.5, 0.5] within a row same as before; u ranges roughly
        # [-width/height/(2*grid)*grid, ...] i.e. wider than [-0.5, 0.5]
        # whenever width > height, which is exactly the extra room a wide
        # canvas is for.
        self._u = (xs - pair_center_x) / row_height
        self._v = (ys - pair_center_y) / row_height

    def _eye_layer(self, u: np.ndarray, v: np.ndarray, stare_x, stare_y, openness):
        """One eye's sclera/iris/pupil composite (not yet clipped/scaled to
        0-255) at cell-local coordinates (u, v) -- ported directly from the
        shader's per-eye arithmetic (see SHADER PORT NOTES). stare_x/y and
        openness may be scalars or full (height, width) per-pixel maps."""
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
        """pair_targets: array-like, shape (num_pairs, 2) -- (x, y) gaze
        target per pair, same -1..1 units as tracked frustum positions.
        pair_openness: array-like, shape (num_pairs,) -- 1=open, 0=closed.
        """
        targets = np.asarray(pair_targets, dtype=np.float64)
        stare = targets * look_gain
        stare_x_map = stare[:, 0][self._pair_idx]
        stare_y_map = stare[:, 1][self._pair_idx]

        openness_map = np.asarray(pair_openness, dtype=np.float64)[self._pair_idx]

        left = self._eye_layer(self._u - self.eye_gap, self._v, stare_x_map, stare_y_map, openness_map)
        right = self._eye_layer(self._u + self.eye_gap, self._v, stare_x_map, stare_y_map, openness_map)

        # Eyes within a pair don't overlap (eye_gap keeps them apart), and
        # neither does one cell's pair reach into the next cell's, so a
        # plain sum is a safe union composite -- no need for a max/over op.
        final = np.clip(left + right, 0.0, 1.0)
        return (final * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1",
                     help="OSC listen host -- match --osc-host on the "
                          "tracker (defaults to localhost on both).")
    ap.add_argument("--port", type=int, default=9000,
                     help="OSC listen port -- match --osc-port on the "
                          "tracker.")
    ap.add_argument("--width", type=int, default=1200,
                     help="Window width in pixels. Wider-than-tall gives "
                          "MORE horizontal room BETWEEN pair columns (see "
                          "EyeRenderer) -- this is the 'distance between "
                          "pairs' knob. It's independent of --eye-gap, "
                          "which is the 'distance between the two eyes "
                          "within a pair' knob.")
    ap.add_argument("--height", type=int, default=480,
                     help="Window height in pixels.")
    ap.add_argument("--fps", type=int, default=30,
                     help="Render loop rate. Independent of the tracker's "
                          "own frame rate -- this just redraws at a fixed "
                          "clock, reading whatever the latest OSC value is.")
    ap.add_argument("--min-cutoff", type=float, default=1.0,
                     help="One Euro filter: cutoff freq when signal is "
                          "still. Lower = smoother at rest, more lag.")
    ap.add_argument("--beta", type=float, default=0.02,
                     help="One Euro filter: how much speed reduces "
                          "smoothing. Higher = snappier, more jitter.")
    ap.add_argument("--no-smoothing", action="store_true",
                     help="Use the raw per-packet look position, no "
                          "filtering.")
    ap.add_argument("--stale-after", type=float, default=1.0,
                     help="Seconds without an OSC update before resting "
                          "the eye at center (e.g. tracker not running).")
    ap.add_argument("--look-gain", type=float, default=16.0,
                     help="How far the iris/pupil shifts per unit of "
                          "frustum position (matches the original shader's "
                          "literal, but that was tuned for a different "
                          "rendering context -- see SHADER PORT NOTES. "
                          "Lower this if the iris pushes out of the "
                          "sclera entirely as the tracked person nears "
                          "the edge of frame.")
    ap.add_argument("--no-wander", action="store_true",
                     help="Disable idle saccade wander -- just rest the "
                          "eye dead-center when nobody's tracked.")
    ap.add_argument("--wander-radius", type=float, default=0.5,
                     help="Max distance (in the same -1..1 frustum units "
                          "as tracked positions) an idle fixation point "
                          "can land from center.")
    ap.add_argument("--wander-hold-min", type=float, default=0.6,
                     help="Shortest idle fixation pause, in seconds.")
    ap.add_argument("--wander-hold-max", type=float, default=2.5,
                     help="Longest idle fixation pause, in seconds.")
    ap.add_argument("--wander-move-min", type=float, default=0.08,
                     help="Shortest idle saccade (move-to-new-spot) "
                          "duration, in seconds.")
    ap.add_argument("--wander-move-max", type=float, default=0.25,
                     help="Longest idle saccade duration, in seconds.")
    ap.add_argument("--grid", type=int, default=3,
                     help="Grid is NxN pairs of eyes (default 3 -> nine "
                          "pairs, matching the original Max patch's "
                          "zoom(9)). All pairs track the same tracked "
                          "position in sync; they only diverge while "
                          "averting (see --avert-* below).")
    ap.add_argument("--eye-gap", type=float, default=0.45,
                     help="Horizontal offset of each eye from its pair's "
                          "center, in cell-local units -- how close "
                          "together the two eyes in a pair sit (the "
                          "'within a pair' knob -- see --width for the "
                          "'between pairs' one). 0.35 gives a bit of soft "
                          "glow overlap between the two eyes (measured "
                          "~14%) so they read as a pair rather than two "
                          "separate eyes; 0.50 was the exact zero-overlap "
                          "point at the default --width, if you want them "
                          "fully distinct instead.")
    ap.add_argument("--no-avert", action="store_true",
                     help="Disable averting -- pairs keep tracking "
                          "normally even on direct eye contact.")
    ap.add_argument("--avert-radius", type=float, default=0.85,
                     help="Max distance a pair's flee target can land "
                          "from center while averting -- deliberately "
                          "wider than --wander-radius so looking away "
                          "reads as an obvious dodge, not idle drift.")
    ap.add_argument("--avert-hold-min", type=float, default=0.3,
                     help="Shortest pause between darts while averting, "
                          "in seconds.")
    ap.add_argument("--avert-hold-max", type=float, default=1.0,
                     help="Longest pause between darts while averting, "
                          "in seconds.")
    ap.add_argument("--avert-move-min", type=float, default=0.05,
                     help="Shortest dart duration while averting, in "
                          "seconds -- the 'quickly' in 'look away "
                          "quickly.'")
    ap.add_argument("--avert-move-max", type=float, default=0.15,
                     help="Longest dart duration while averting, in "
                          "seconds.")
    ap.add_argument("--no-blink", action="store_true",
                     help="Disable blinking.")
    ap.add_argument("--blink-interval-min", type=float, default=2.0,
                     help="Shortest gap between a pair's blinks, in "
                          "seconds.")
    ap.add_argument("--blink-interval-max", type=float, default=6.0,
                     help="Longest gap between a pair's blinks, in "
                          "seconds.")
    ap.add_argument("--blink-duration-min", type=float, default=0.12,
                     help="Shortest blink duration, in seconds.")
    ap.add_argument("--blink-duration-max", type=float, default=0.25,
                     help="Longest blink duration, in seconds.")
    args = ap.parse_args()

    receiver = LookReceiver(args.host, args.port).start()
    renderer = EyeRenderer(args.width, args.height, grid=args.grid, eye_gap=args.eye_gap)
    n = renderer.num_pairs

    # One independent instance per pair for both idle wander and averting
    # -- each gets its own random seed from OS entropy (seed=None), so
    # pairs diverge from each other on their own without any explicit
    # coordination. (Only the real tracked position is ever shared across
    # all pairs -- that's the "track the same thing" case.)
    idle_wanders = [
        SaccadeWander(
            radius=args.wander_radius,
            hold_range=(args.wander_hold_min, args.wander_hold_max),
            move_range=(args.wander_move_min, args.wander_move_max),
        )
        for _ in range(n)
    ]
    avert_wanders = [
        SaccadeWander(
            radius=args.avert_radius,
            hold_range=(args.avert_hold_min, args.avert_hold_max),
            move_range=(args.avert_move_min, args.avert_move_max),
        )
        for _ in range(n)
    ]
    blinkers = [
        BlinkGenerator(
            interval_range=(args.blink_interval_min, args.blink_interval_max),
            duration_range=(args.blink_duration_min, args.blink_duration_max),
        )
        for _ in range(n)
    ]
    print(f"[eye] listening for /looking on {args.host}:{args.port} -- {n} pairs")

    # Persistent per-pair, per-axis smoothing filters -- always active
    # regardless of mode (tracking/idle/averting). Letting them run
    # continuously (rather than resetting on mode changes) is deliberate:
    # when averting ends, each pair's filter eases from wherever that pair
    # scattered to back toward the shared target independently, which
    # reads as the group cautiously regrouping rather than snapping back
    # in unison.
    filt_x: list = [None] * n
    filt_y: list = [None] * n
    was_facing = False
    frame_interval = 1.0 / args.fps

    try:
        while True:
            loop_start = time.monotonic()
            now = loop_start
            lx, ly, source, is_facing = receiver.latest(stale_after_s=args.stale_after)

            if is_facing and not was_facing and not args.no_avert:
                for w in avert_wanders:
                    w.flee(now)
            was_facing = is_facing

            if is_facing and not args.no_avert:
                raw_targets = [w.update(now) for w in avert_wanders]
            elif source == 0 and not args.no_wander:
                raw_targets = [w.update(now) for w in idle_wanders]
            else:
                # frustum_y follows the tracker's documented image-space
                # convention (-1 top .. +1 bottom), but the shader's stare
                # math shifts the iris toward the OPPOSITE side of whatever
                # offset it's given (adding +Y to the argument of a
                # minimum-distance formula moves that minimum toward -Y) --
                # confirmed by rendering look_y=+0.8 (person low in frame)
                # and seeing the iris land at the TOP instead of the
                # bottom. Negate here, at the one place real tracked
                # position enters "look", rather than in the tracker or in
                # the shader math itself.
                raw_targets = [(lx, -ly)] * n

            if args.no_smoothing:
                smoothed = raw_targets
            else:
                smoothed = []
                for i in range(n):
                    tx, ty = raw_targets[i]
                    if filt_x[i] is None:
                        filt_x[i] = OneEuroFilter(now, tx, min_cutoff=args.min_cutoff, beta=args.beta)
                        filt_y[i] = OneEuroFilter(now, ty, min_cutoff=args.min_cutoff, beta=args.beta)
                        smoothed.append((tx, ty))
                    else:
                        smoothed.append((filt_x[i](now, tx), filt_y[i](now, ty)))

            if args.no_blink:
                openness = [1.0] * n
            else:
                openness = [b.update(now) for b in blinkers]

            img = renderer.render(smoothed, openness, look_gain=args.look_gain)
            cv2.imshow("eye display", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
