import numpy as np
from PIL import Image
from one_hot_hangul import CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST
from random import randint
import unicodedata


# !!!!! 테스트할 데이터들의 PATH를 잘 입력하세요


# 훈련 세트 전용 검사기 (이미지 출력)
def test_dataset_tensor(
    path: str, how_many: int = 3, starting_from: int = 0, reversed: bool = False
) -> None:
    loaded = np.load(path)
    length = loaded.shape[0]

    print("텐서 모양: ", loaded.shape)

    n_show = min(how_many, length - starting_from)

    if not reversed:
        start = starting_from
        end = starting_from + n_show - 1
    else:
        start = length - 1 - starting_from
        end = length - starting_from - n_show

    print(
        start,
        "번째 데이터부터 ",
        end,
        "번째 데이터까지 확인합니다.",
    )

    reversed_int = 1 if not reversed else -1

    for i in range(start, end + reversed_int, reversed_int):
        array = loaded[i]
        array = array.astype(np.uint8)
        img = Image.fromarray(array, mode="L")
        img.show(str(i) + ".png")


# 유니코드 이미지 전용 검사기 (이미지 출력)
def test_unicode_tensor(
    path: str, how_many: int = 3, starting_from: int = 0, reversed: bool = False
) -> None:
    loaded = np.load(path)
    length = loaded.shape[0]

    print("텐서 모양: ", loaded.shape)

    n_show = min(how_many, length - starting_from)

    if not reversed:
        start = starting_from
        end = starting_from + n_show - 1
    else:
        start = length - 1 - starting_from
        end = length - starting_from - n_show

    print(
        start,
        "번째 데이터부터 ",
        end,
        "번째 데이터까지 확인합니다.",
    )

    reversed_int = 1 if not reversed else -1

    for i in range(start, end + reversed_int, reversed_int):
        array = loaded[i]
        array = np.squeeze(array)  # (64, 64, 1) -> (64, 64)로 차원 축소
        array *= 255
        array = array.astype(
            np.uint8
        )  # array를 0-255 범위의 부호 없는 8비트 정수로 변환
        img = Image.fromarray(array, mode="L")
        img.show(str(i) + ".png")


# 훈련 세트 레이블 전용 검사기 (인코딩 -> 레이블 출력)
def test_dataset_label(
    path: str, how_many: int = 3, starting_from: int = 0, reversed: bool = False
) -> None:
    loaded = np.load(path)
    length = loaded.shape[0]

    print("텐서 모양: ", loaded.shape)

    n_show = min(how_many, length - starting_from)

    if not reversed:
        start = starting_from
        end = starting_from + n_show - 1
    else:
        start = length - 1 - starting_from
        end = length - starting_from - n_show

    print(
        start,
        "번째 데이터부터 ",
        end,
        "번째 데이터까지 확인합니다.",
    )

    reversed_int = 1 if not reversed else -1

    for i in range(start, end + reversed_int, reversed_int):
        encoding = loaded[i]
        for i, value in enumerate(encoding):
            if value == 0:
                continue

            if i < 19:
                print(CHOSUNG_LIST[i], end="")
            elif i < 40:
                print(JUNGSUNG_LIST[i - 19], end=" ")
            elif i == 40:
                print("빈칸")
            else:
                print(JONGSUNG_LIST[i - 40])


# 유니코드 레이블 전용 검사기 (레이블, 레이블의 ord(), 레이블의 이름 출력)
def test_unicode_label(
    path: str, how_many: int = 3, starting_from: int = 0, reversed: bool = False
) -> None:
    loaded = np.load(path, allow_pickle=True)
    length = loaded.shape[0]

    print("텐서 모양: ", loaded.shape)

    n_show = min(how_many, length - starting_from)

    if not reversed:
        start = starting_from
        end = starting_from + n_show - 1
    else:
        start = length - 1 - starting_from
        end = length - starting_from - n_show

    print(
        start,
        "번째 데이터부터 ",
        end,
        "번째 데이터까지 확인합니다.",
    )

    reversed_int = 1 if not reversed else -1

    for i in range(start, end + reversed_int, reversed_int):
        label = loaded[i]
        if isinstance(label, (np.integer, int)):
            code_point = label
            char = chr(code_point)
            try:
                name = unicodedata.name(char)
            except ValueError:
                name = "NAME NOT FOUND"
            print(f"{char} {hex(code_point)} {name}")
        elif isinstance(label, str):
            char = label
            try:
                names = " / ".join(unicodedata.name(c) for c in char)
                hexes = " + ".join(hex(ord(c)) for c in char)
                print(f"{char} ({hexes}) {names}")
            except ValueError:
                print(f"{char} (NAME NOT FOUND)")


INDEX = 4000

test_unicode_tensor(
    path="combined_unicode_tensors_0.npy",
    how_many=5,
    starting_from=INDEX,
    reversed=False,
)

"ᐚ ᐛ"

test_unicode_label(
    path="combined_unicode_labels_0.npy",
    how_many=5,
    starting_from=INDEX,
    reversed=False,
)
