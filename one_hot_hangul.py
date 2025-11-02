import numpy as np


# 초성, 중성, 종성 리스트
CHOSUNG_LIST = [
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]
JUNGSUNG_LIST = [
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]
JONGSUNG_LIST = [
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]


def one_hot_hangul(char):
    """
    한글 한 글자를 초성, 중성, 종성으로 분해하여 리스트로 반환합니다.
    """
    # 한글 음절의 유니코드 시작점과 끝점
    HANGUL_START = 0xAC00
    HANGUL_END = 0xD7A3

    # 입력된 문자의 유니코드 값
    char_code = ord(char)

    # 한글 음절 범위에 있는지 확인
    if HANGUL_START <= char_code <= HANGUL_END:
        # 초성, 중성, 종성 인덱스 계산 (중성 개수 = 21, 종성 개수 = 28)
        relative_code = char_code - HANGUL_START
        chosung_index = relative_code // (21 * 28)
        jungsung_index = (relative_code % (21 * 28)) // 28
        jongsung_index = relative_code % 28

        # 원-핫 인코딩 [19개 초성 | 21개 중성 | 28개 종성]으로 구성된 벡터 만들기
        # 인코딩의 각 초성 중성 종성의 위치는 위 CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST와 같음
        encoding = np.zeros((68,), np.bool_)

        # 각  1로 바꾸기
        encoding[chosung_index] = 1
        encoding[19 + jungsung_index] = 1
        encoding[19 + 21 + jongsung_index] = 1

        return encoding
    else:
        # 한글 음절이 아니면 오류
        raise ValueError("입력이 한글이 아닙니다!")


# 테스트
if __name__ == "__main__":
    print(one_hot_hangul("각"))
    print(one_hot_hangul("가"))
