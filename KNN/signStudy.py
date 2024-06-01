import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
import pandas as pd

max_num_hands = 1
gesture = {0: 'ㄱ', 1: 'ㄲ', 2:'ㄴ', 3:'ㄷ', 4:'ㄸ', 5:'ㄹ', 6:'ㅁ', 7:'ㅂ', 8:'ㅃ', 9:'ㅅ', 10:'ㅇ', 11:'ㅈ', 12:'ㅉ', 13:'ㅊ', 14:'ㅋ', 15:'ㅌ',16:'ㅍ',17:'ㅎ',
           18:'ㅏ', 19:'ㅑ', 20:'ㅓ', 21:'ㅕ', 22:'ㅗ', 23:'ㅛ', 24:'ㅜ', 25:'ㅠ', 26:'ㅡ', 27:'ㅣ',
           28:'ㅐ', 29:'ㅒ', 30:'ㅔ', 31:'ㅖ', 32:' ㅚ', 33:'ㅟ', 34:'ㅢ', 35:'clear', 36:'space'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# f= open("test.txt", 'w')
f=open("KNN/test.txt", 'w')
cap = cv2.VideoCapture(0)
startTime = time.time()
prev_index = 0
sentence = ''

recognizeDelay = 1

while True:
    ret, img = cap.read()
    if not ret:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))

            angle = np.degrees(angle)
            if keyboard.is_pressed('a'):
                for num in angle:
                    num=round(num,6)
                    f.write(str(num))
                    f.write(',')
                f.write("36.000000") #데이터를 저장할 gesture의 label번호
                f.write('\n')
                print("next")

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("HandTracking", img)
    if cv2.waitKey(1) == ord('b'):
        break

f.close()

# test.txt 파일을 읽습니다.
df_test = pd.read_csv('KNN/test.txt', header=None)

# 임의의 20개 행을 선택합니다.
df_sample = df_test.sample(20)

# dataset.csv 파일이 이미 존재하는 경우, 선택한 행을 추가합니다.
# 파일이 존재하지 않는 경우, 새 파일을 생성합니다.
df_sample.to_csv('KNN/dataset.csv', mode='a', header=False, index=False)