def combine_hangul(chosung_idx, jungsung_idx, jongsung_idx):
    CHOSUNG_BASE = ['ㄱ', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    JUNGSUNG_BASE = ['ㅏ', 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    JONGSUNG_BASE = [' ', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    # 매핑된 인덱스로부터 초성, 중성, 종성의 유니코드 인덱스를 가져온다.
    chosung = CHOSUNG_BASE[chosung_idx]
    jungsung = JUNGSUNG_BASE[jungsung_idx]
    jongsung = JONGSUNG_BASE[jongsung_idx]  # 종성 인덱스가 35인 경우는 종성이 없음을 의미 (35는 'clear'의 인덱스)

    # 유니코드 포인트 계산
    unicode = 0xAC00 + (chr(chosung) * 21 + chr(jungsung)) * 28 + chr(jongsung)
    #unicode=chosung+jungsung+jongsung
    # 
    return chr(unicode)

# 초성, 중성, 종성 인덱스 예제 (ㄱ, ㅏ, 없음)
print(combine_hangul(0, 0, 0))  # '가'