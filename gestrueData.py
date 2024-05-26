import cv2
import os
import mediapipe as mp
import numpy as np

# Setup Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create directory to save images if it doesn't exist
data_dir = 'sign_language_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Specify the sign language gestures you want to capture
gestures = ['A', 'B', 'C']  # Extend this list with more gestures as needed

# Initialize webcam
cap = cv2.VideoCapture(0)

for gesture in gestures:
    gesture_dir = os.path.join(data_dir, gesture)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    print(f'Collecting images for gesture: {gesture}')
    for img_num in range(100):  # Capture 100 images per gesture
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                landmarks = landmarks.flatten()
                np.save(os.path.join(gesture_dir, f'{gesture}_{img_num}.npy'), landmarks)
        
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Image: {img_num}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Collecting Sign Language Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()