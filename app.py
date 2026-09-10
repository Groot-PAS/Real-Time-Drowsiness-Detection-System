import cv2
import gradio as gr
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import urllib.request
import os
import time

# --- Download the MediaPipe Model ---
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# --- Initialize MediaPipe Face Landmarker ---
options = FaceLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.IMAGE, # Use IMAGE mode for Gradio frame-by-frame processing
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(options)

# --- Logic from original script ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = (13, 14)
FACE_SCALE = (33, 263)

def _euclid(a, b):
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(eye6):
    a = _euclid(eye6[1], eye6[5])
    b = _euclid(eye6[2], eye6[4])
    c = _euclid(eye6[0], eye6[3])
    if c <= 1e-6: return 0.0
    return (a + b) / (2.0 * c)

def final_ear_from_facemesh(landmarks_px):
    left = landmarks_px[np.array(LEFT_EYE, dtype=np.int32)]
    right = landmarks_px[np.array(RIGHT_EYE, dtype=np.int32)]
    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
    return ear, left, right

def yawn_ratio_from_facemesh(landmarks_px):
    top = landmarks_px[MOUTH_INNER[0]]
    bottom = landmarks_px[MOUTH_INNER[1]]
    mouth_open = _euclid(top, bottom)
    s1 = landmarks_px[FACE_SCALE[0]]
    s2 = landmarks_px[FACE_SCALE[1]]
    scale = _euclid(s1, s2)
    if scale <= 1e-6: return 0.0
    return mouth_open / scale

# Track consecutive frames for drowsiness
counter = 0

def process_frame(frame):
    global counter
    
    eye_ar_thresh = 0.25
    eye_ar_consec_frames = 15 # Lowered for web streaming
    yawn_thresh = 0.045
    
    if frame is None:
        return None
        
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = landmarker.detect(mp_image)
    
    if result.face_landmarks:
        lm = result.face_landmarks[0]
        landmarks_px = np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)
        
        ear, left_eye, right_eye = final_ear_from_facemesh(landmarks_px)
        yawn_ratio = yawn_ratio_from_facemesh(landmarks_px)
        
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
        
        # Check Drowsiness
        if ear < eye_ar_thresh:
            counter += 1
            if counter >= eye_ar_consec_frames:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            counter = 0
            
        # Check Yawn
        if yawn_ratio > yawn_thresh:
            cv2.putText(frame, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        cv2.putText(frame, f"EAR: {ear:.2f}", (int(w-150), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"YAWN: {yawn_ratio:.3f}", (int(w-150), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        counter = 0
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

# --- Gradio UI ---
with gr.Blocks(title="Real-Time Drowsiness Detection") as demo:
    gr.Markdown("# 💤 Real-Time Drowsiness Detection")
    gr.Markdown("This app analyzes your webcam feed to detect if you are closing your eyes (drowsiness) or yawning.")
    
    with gr.Row():
        # WebRTC component for webcam streaming
        image_input = gr.Image(sources=["webcam"], streaming=True)
        image_output = gr.Image(streaming=True)
        
    image_input.stream(fn=process_frame, inputs=image_input, outputs=image_output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
