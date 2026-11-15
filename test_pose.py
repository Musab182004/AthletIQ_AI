import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# Model download
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        model_path
    )

# Setup for VIDEO mode
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp_ms = int((frame_count / fps) * 1000)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # VIDEO mode ke liye timestamp zaruri hai
    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        h, w, _ = frame.shape

        # Dots draw karo
        for landmark in landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Knee angle calculate karo
        # Left side — hip(23), knee(25), ankle(27)
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]

        hx, hy = int(hip.x * w), int(hip.y * h)
        kx, ky = int(knee.x * w), int(knee.y * h)
        ax, ay = int(ankle.x * w), int(ankle.y * h)

        import math
        angle = math.degrees(math.atan2(ay - ky, ax - kx) - 
                            math.atan2(hy - ky, hx - kx))
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle

        # Angle dikhao screen pe
        cv2.putText(frame, f"Knee Angle: {int(angle)}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Done!")