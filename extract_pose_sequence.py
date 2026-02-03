import cv2
import mediapipe as mp
import numpy as np


def extract_pose_sequence(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    cap = cv2.VideoCapture(video_path)
    frame_features = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Flatten all landmark coordinates into a 1D vector
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        else:
            keypoints = np.zeros(33 * 3)  # empty pose

        frame_features.append(keypoints)

    cap.release()
    return np.array(frame_features)


import numpy as np


