import streamlit as st
import cv2
import time
import numpy as np
import torch
import cvzone
from ultralytics import YOLO

# Load YOLO model
MODEL_PATH = "models/m_version_1_136.pt"  # Adjust path if needed
model = YOLO(MODEL_PATH)

classNames = ["fake", "real"]
confidence_threshold = 0.8

# ------------------------- STREAMLIT UI -------------------------

# Apply custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://shuftipro.com/wp-content/uploads/verification_featured.jpg");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat; 
        }
        .stButton > button {
        width: 250px;
        height: 70px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 25px;
        border: none;
        background: white;
        color: black;
        margin: 10px;
        transition: 0.3s;
        }
        .stButton > button:hover {
            background: #FFD700;
            color: black;
        }
        .team {
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            margin-top: 20px;
        }
        .face-count {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<p style='font-size:55px; color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); font-weight:bold; text-align:center;'>Face Anti-Spoofing Detection</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:23px; color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); text-align:center;'>Click 'Start' to open the camera and classify in real-time.</p>", unsafe_allow_html=True)


# Buttons
st.markdown("<div class='buttons'>", unsafe_allow_html=True)
start_detection = st.button("Start")
stop_detection = st.button("Stop")
st.markdown("</div>", unsafe_allow_html=True)

# State for camera control
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if start_detection:
    st.session_state.camera_active = True
if stop_detection:
    st.session_state.camera_active = False

frame_placeholder = st.empty()

# Start camera and detection
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    prev_frame_time = 0

    while cap.isOpened():
        new_frame_time = time.time()
        success, img = cap.read()
        if not success:
            st.write("Failed to capture image.")
            break

        results = model(img, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                if conf > confidence_threshold:
                    color = (0, 255, 0) if classNames[cls] == "real" else (0, 0, 255)

                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(
                        img,
                        f"{classNames[cls].upper()} {int(conf * 100)}%",
                        (max(0, x1), max(35, y1)),
                        scale=2,
                        thickness=4,
                        colorR=color,
                        colorB=color,
                    )

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Convert frame to RGB for Streamlit
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img, channels="RGB")

        if not st.session_state.camera_active:
            break

    cap.release()
    cv2.destroyAllWindows()
