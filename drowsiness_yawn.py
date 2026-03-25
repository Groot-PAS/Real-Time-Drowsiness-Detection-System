"""
python drowsiness_yawn.py --webcam 0 --alarm Alert.wav

Modernized for Python 3.12+:
- Replaced dlib/scipy with MediaPipe Tasks FaceLandmarker
- Removed playsound dependency (uses built-in winsound on Windows)
"""

from __future__ import annotations

import argparse
import os
import time
from threading import Thread

import cv2
import imutils
import numpy as np

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: mediapipe. Run: pip install -r requirements.txt\n"
        f"Import error: {e}"
    )


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _play_wav_windows(path: str) -> None:
    if os.name != "nt":
        return
    try:
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


alarm_status = False
alarm_status2 = False
saying = False


def sound_alarm(path: str) -> None:
    global alarm_status, alarm_status2, saying

    while alarm_status:
        _play_wav_windows(path)
        time.sleep(0.5)

    if alarm_status2:
        saying = True
        _play_wav_windows(path)
        time.sleep(0.5)
        saying = False


# MediaPipe FaceMesh landmark indices (commonly used sets)
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # p1,p2,p3,p4,p5,p6
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = (13, 14)  # inner upper/lower lip
FACE_SCALE = (33, 263)  # eye outer corners for normalization


def eye_aspect_ratio(eye6: np.ndarray) -> float:
    a = _euclid(eye6[1], eye6[5])
    b = _euclid(eye6[2], eye6[4])
    c = _euclid(eye6[0], eye6[3])
    if c <= 1e-6:
        return 0.0
    return (a + b) / (2.0 * c)


def final_ear_from_facemesh(landmarks_px: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    left = landmarks_px[np.array(LEFT_EYE, dtype=np.int32)]
    right = landmarks_px[np.array(RIGHT_EYE, dtype=np.int32)]
    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
    return ear, left, right


def yawn_ratio_from_facemesh(landmarks_px: np.ndarray) -> float:
    top = landmarks_px[MOUTH_INNER[0]]
    bottom = landmarks_px[MOUTH_INNER[1]]
    mouth_open = _euclid(top, bottom)

    s1 = landmarks_px[FACE_SCALE[0]]
    s2 = landmarks_px[FACE_SCALE[1]]
    scale = _euclid(s1, s2)
    if scale <= 1e-6:
        return 0.0
    return mouth_open / scale


def main() -> int:
    global alarm_status, alarm_status2

    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
    ap.add_argument(
        "-a",
        "--alarm",
        type=str,
        default="Alert.wav",
        help="path to alarm WAV file (Windows recommended)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "face_landmarker.task"),
        help="path to MediaPipe face landmarker model (.task)",
    )
    ap.add_argument(
        "--list-cams",
        action="store_true",
        help="probe camera indices (0..10) and exit",
    )
    ap.add_argument("--ear-thresh", type=float, default=0.25, help="eye aspect ratio threshold")
    ap.add_argument("--ear-frames", type=int, default=30, help="consecutive frames below threshold")
    ap.add_argument(
        "--yawn-thresh",
        type=float,
        default=0.045,
        help="normalized mouth opening threshold (tune if too sensitive)",
    )
    args = vars(ap.parse_args())

    eye_ar_thresh = float(args["ear_thresh"])
    eye_ar_consec_frames = int(args["ear_frames"])
    yawn_thresh = float(args["yawn_thresh"])

    alarm_path = str(args["alarm"])
    if alarm_path and not os.path.isabs(alarm_path):
        alarm_path = os.path.join(os.getcwd(), alarm_path)

    def _probe_cameras(max_index: int = 10) -> list[int]:
        available: list[int] = []
        for i in range(max_index + 1):
            cap_i = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap_i is not None and cap_i.isOpened():
                available.append(i)
                cap_i.release()
            else:
                try:
                    cap_i.release()
                except Exception:
                    pass
        return available

    if bool(args["list_cams"]):
        cams = _probe_cameras(10)
        print("Available camera indices (0..10):", cams)
        return 0

    # Ensure the FaceLandmarker model exists (auto-download if missing)
    model_path = str(args["model"])
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    def _ensure_model(path: str) -> None:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"-> Downloading model to: {path}")
        print(f"-> From: {url}")
        try:
            import urllib.request

            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise SystemExit(
                "Could not download the MediaPipe model automatically.\n"
                f"- URL: {url}\n"
                f"- Target: {path}\n"
                f"Error: {e}"
            )

    _ensure_model(model_path)

    # Import Tasks API (this is what your mediapipe build provides on Windows)
    try:
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Your MediaPipe installation doesn't include the Tasks API.\n"
            "Try reinstalling: pip install --upgrade mediapipe\n"
            f"Import error: {e}"
        )

    print("-> Starting Video Stream")
    # Phone Link / virtual webcams often work best via DirectShow backend
    cap = cv2.VideoCapture(int(args["webcam"]), cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(int(args["webcam"]))
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open webcam index {args['webcam']}.\n"
            "Run: python drowsiness_yawn.py --list-cams\n"
            "Then retry with -w <index>.\n"
            "If you are using Phone Link, ensure it is exposing a webcam/virtual camera device."
        )

    options = FaceLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    counter = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = imutils.resize(frame, width=640)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                landmarks_px = np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)

                ear, left_eye, right_eye = final_ear_from_facemesh(landmarks_px)
                yawn_ratio = yawn_ratio_from_facemesh(landmarks_px)

                cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

                if ear < eye_ar_thresh:
                    counter += 1
                    if counter >= eye_ar_consec_frames:
                        if alarm_status is False:
                            alarm_status = True
                            if alarm_path:
                                t = Thread(target=sound_alarm, args=(alarm_path,))
                                t.daemon = True
                                t.start()

                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                else:
                    counter = 0
                    alarm_status = False

                if yawn_ratio > yawn_thresh:
                    cv2.putText(
                        frame,
                        "YAWN ALERT!",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    if alarm_status2 is False and saying is False:
                        alarm_status2 = True
                        if alarm_path:
                            t = Thread(target=sound_alarm, args=(alarm_path,))
                            t.daemon = True
                            t.start()
                else:
                    alarm_status2 = False

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"YAWN: {yawn_ratio:.3f}",
                    (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                counter = 0
                alarm_status = False
                alarm_status2 = False
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
