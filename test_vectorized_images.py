import numpy as np
from PIL import Image
from one_hot_hangul import CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST


# !!!!! 테스트할 데이터들의 PATH를 잘 입력하세요

# k = 0 # k번째 텐서와 레이블을 같이 테스팅할 수 있도록 하는 변수
# PATH_TENSOR = f"Hangul-to-Unicode-Obfuscation-Project/unicode_tensors_{k}.npy"
# PATH_LABEL = f"obfuscation_labels/labels_handwriting_{k}.npy"
j = 0
PATH_UNICODE_TENSOR = f"unicode_tensors_{j}.npy"


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


# 레이블 전용 검사기 (인코딩 -> 레이블 출력)
def test_label(
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


# 64번 부터 순방향 5개 유니코드를 검사
test_unicode_tensor(
    path=PATH_UNICODE_TENSOR, how_many=5, starting_from=64, reversed=False
)

# test_label(path=PATH_LABEL, how_many=5, starting_from=20000, reversed=False)
