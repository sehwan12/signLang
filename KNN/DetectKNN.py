import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
from PIL import Image, ImageFont, ImageDraw
chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
jung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
jong_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def putText(img, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)


def combine_hangul(chosung, jungsung, jongsung):
    # 초성과 중성만 있는 경우
    if jongsung == 100:
        return chr(0xAC00 + chosung_list.index(chosung) * 21 * 28 + jung_list.index(jungsung) * 28)
    # 모든 글자가 있는 경우
    return chr(0xAC00 + (chosung_list.index(chosung) * 21 + jung_list.index(jungsung)) * 28 + jong_list.index(jongsung))

max_num_hands = 1
gesture = {0: 'ㄱ', 1: 'ㄲ', 2:'ㄴ', 3:'ㄷ', 4:'ㄸ', 5:'ㄹ', 6:'ㅁ', 7:'ㅂ', 8:'ㅃ', 9:'ㅅ', 10:'ㅇ', 11:'ㅈ', 12:'ㅉ', 13:'ㅊ', 14:'ㅋ', 15:'ㅌ',16:'ㅍ',17:'ㅎ',
           18:'ㅏ', 19:'ㅑ', 20:'ㅓ', 21:'ㅕ', 22:'ㅗ', 23:'ㅛ', 24:'ㅜ', 25:'ㅠ', 26:'ㅡ', 27:'ㅣ',
           28:'ㅐ', 29:'ㅒ', 30:'ㅔ', 31:'ㅖ', 32:'ㅚ', 33:'ㅟ', 34:'ㅢ', 35:'clear', 36:'space'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


file = np.genfromtxt('KNN/dataset.csv', delimiter=',', ndmin=2)
angleFile = file[:, :-1]
label = file[:, -1]
angle = angleFile.astype(np.float32)
label = label.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
startTime = time.time()
prev_index = 0
sentence = ''
fullsentence = ''
# 한글 조합을 위한 변수
current_chosung = ''
current_jungsung = ''
current_jongsung = ''
combined_char = ''
# 초성, 중성, 종성을 추적하는 변수를 추가합니다.
input_stage = "chosung"

recognizeDelay = 5
last_input_time=time.time()-0.5
input_delay=0.1
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

                compareV1 = v[[0, 1, 2, 4, 5, 6, 7,  8,  9, 10, 12, 13, 14, 16, 17], :]
                compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))

                angle = np.degrees(angle)

                data = np.array([angle], dtype=np.float32)

                # knn.findNearest를 호출하기 전에 dataset을 필터링합니다.
                if input_stage == "chosung" or input_stage == "jongsung":
                    filtered_dataset = [row for row in data if 0 <= row[-1] <= 17]
                elif input_stage == "jungsung":
                    filtered_dataset = [row for row in data if 18 <= row[-1] <= 34]
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])
                
                if current_chosung is '':
                    if 0<=idx<=17 :
                        current_chosung = gesture[idx]
                        input_stage = "jungsung"
                    elif idx == 35:
                        fullsentence = ''
                    elif idx == 36:
                        fullsentence += ' '
                    else:
                        pass    
                elif current_chosung and current_jungsung is '':
                    if 18<=idx<=34 :
                        current_jungsung = gesture[idx]
                        input_stage = "jongsung"
                    elif idx == 35:
                        fullsentence = ''
                    elif idx == 36:
                        fullsentence += ' '
                    else:
                        pass
                else:
                    if 0<=idx<=17 :
                        current_jongsung = gesture[idx]
                        input_stage = "chosung"
                    elif idx == 35:
                        fullsentence = ''
                    elif idx == 36:
                        fullsentence += ' '
                    else:
                        pass
                # 한글 문자 조합 및 화면에 표시
                if current_chosung and current_jungsung :
                    combined_char = combine_hangul(current_chosung,current_jungsung ,current_jongsung )
                    if current_jongsung:
                        sentence += combined_char
                        combined_char=''
                        current_chosung = ''
                        current_jungsung = ''
                        current_jongsung = ''
                        input_stage = "chosung"
                        
                print(f"Current chosung: {current_chosung}, Current jungsung: {current_jungsung}, Current jongsung: {current_jongsung}")
                print(f"Sentence: {sentence}, Combined Char: {combined_char}")
                img=putText(img, gesture[idx].upper(), (int(res.landmark[0].x * img.shape[1]) - 10, int(res.landmark[0].y * img.shape[0]) + 40), 'KNN/NanumGothic.ttf', 30, (255, 255, 255))
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                
    if time.time() - last_input_time >= input_delay:  
        if(sentence != ''):
            print(sentence)
            fullsentence += sentence
            img = putText(img, fullsentence, (20, 400), 'KNN/NanumGothic.ttf', 40, (255, 255, 255))
            last_input_time = time.time()
            sentence = ''
        elif(combined_char != ''):
            print(combined_char)
            fullsentence += combined_char
            img = putText(img, fullsentence, (20, 400), 'KNN/NanumGothic.ttf', 40, (255, 255, 255))
            last_input_time = time.time()
            fullsentence=fullsentence[:-1]
        else:
            print(current_chosung)
            fullsentence += current_chosung
            img = putText(img, fullsentence, (20, 400), 'KNN/NanumGothic.ttf', 40, (255, 255, 255))
            last_input_time = time.time()
            fullsentence=fullsentence[:-1]       
        
        cv2.imshow("HandTracking", img)
        key = cv2.waitKey(1)
        if key == ord('b'):
            break
        elif key == ord('c'):
            current_chosung = ''
            current_jungsung = ''
            current_jongsung = ''
            sentence = ''
            fullsentence = ''
            combined_char = ''
            img = putText(img, fullsentence, (20, 400), 'KNN/NanumGothic.ttf', 40, (255, 255, 255))
            print("Clear")

