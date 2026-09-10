#!/usr/bin/env python3
"""
camera_body_test.py (Tasks API version)

Minimal starting point: open the camera, run MediaPipe's Tasks API
PoseLandmarker on each frame, draw the detected body landmarks as dots.
No threading, no smoothing, no frustum/trigger logic -- just confirming
camera capture + body detection work together reliably.

Uses the Tasks API (not the legacy solutions.pose) because the pip
mediapipe package has been inconsistently shipping the legacy solutions
module across recent versions. delegate=CPU is set explicitly since
leaving it unset previously triggered a macOS GPU-service crash with the
FaceLandmarker task -- worth confirming whether PoseLandmarker has the
same issue or not.

Setup:
    pip install opencv-python mediapipe

    # Download the pose landmarker model once (lite = fastest/smallest,
    # good starting point -- switch to _full or _heavy later if accuracy
    # needs improving):
    curl -L -o pose_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

Run:
    python3 camera_body_test.py --camera 0
"""

import argparse
import platform

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    system = platform.system()
    backend = (cv2.CAP_AVFOUNDATION if system == "Darwin" else
               cv2.CAP_V4L2 if system == "Linux" else cv2.CAP_ANY)

    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. On macOS, check System "
            f"Settings -> Privacy & Security -> Camera has granted access "
            f"to your terminal/IDE."
        )

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[camera] requested {width}x{height}@{fps} -> "
          f"negotiated {actual_w}x{actual_h}@{actual_fps:.1f}")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--model", type=str, default="pose_landmarker.task")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.fps)

    base_options = mp_python.BaseOptions(
        model_asset_path=args.model,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    print("Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] frame read failed, retrying...")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            h, w = frame.shape[:2]
            if result.pose_landmarks:
                for landmarks in result.pose_landmarks:
                    for lm in landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                                   4, (0, 255, 0), -1)
                status = "body detected"
            else:
                status = "no body detected"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.imshow("body detection test (Tasks API)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()