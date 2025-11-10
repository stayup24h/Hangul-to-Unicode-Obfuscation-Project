import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from tqdm import tqdm

from search_fonts import NOTO_FONTS_NAME

# 스크립트 파일의 위치
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH: str = os.path.join(
    SCRIPT_DIR, "..", r"notofonts\fonts\{0}\full\ttf\{0}-Regular.ttf"
)

# --- 설정 ---
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
OUTPUT_FILENAME_BASE = (
    "unicode_tensors"  # unicode_tensors_0.npy, unicode_tensors_1.npy 등으로 저장됩니다.
)
UNRENDERED_LOG_FILENAME = "unrendered_characters.txt"
FONT_SIZE = 48  # 이미지 크기에 맞게 조절
# 유니코드 범위를 나눌 크기 (65536개씩). 약 1GB의 메모리를 사용하며 총 17개의 파일이 생성될 수 있습니다.
CHUNK_SIZE = 0x08000

# --- 유니코드 제외 범위 ---
# C0/C1 제어 문자, 서로게이트, 사설 영역 등은 렌더링할 수 없거나 의미가 없습니다.
EXCLUDE_RANGES = [
    (0x0000, 0x001F),  # C0 Controls
    (0x007F, 0x009F),  # C1 Controls
    (0xD800, 0xDFFF),  # High and Low Surrogates
    (0xE000, 0xF8FF),  # Private Use Area
    (0xFFF0, 0xFFFF),  # Specials
    (0xF0000, 0xFFFFD),  # Supplementary Private Use Area-A
    (0x100000, 0x10FFFD),  # Supplementary Private Use Area-B
]


def is_excluded(code_point):
    """주어진 유니코드 코드 포인트가 제외 범위에 속하는지 확인합니다."""
    for start, end in EXCLUDE_RANGES:
        if start <= code_point <= end:
            return True
    return False


# 폰트 지원 여부를 빠르게 확인하기 위한 캐시
font_cache = {}


def is_char_in_font(char, font_path):
    """주어진 폰트 파일이 특정 문자를 지원하는지 확인합니다."""
    if font_path not in font_cache:
        try:
            # TTFont 객체를 캐시에 저장하여 반복적인 파일 로딩 방지
            font_cache[font_path] = TTFont(font_path, lazy=True)
        except Exception:
            font_cache[font_path] = None
            return False

    font = font_cache[font_path]
    if font is None:
        return False

    # 폰트의 cmap(character-to-glyph-map)을 확인하여 문자 지원 여부 판단
    for cmap in font.get("cmap").tables:
        if ord(char) in cmap.cmap:
            return True
    return False


def render_char_to_tensor(char, font_path):
    """문자를 이미지로 렌더링하고 Numpy 텐서로 변환합니다."""
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except IOError:
        # Pillow가 폰트 파일을 열지 못하는 경우
        return None

    image = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    # 문자를 이미지 중앙에 정렬하기 위해 바운딩 박스 계산
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = (
            (IMAGE_WIDTH - text_width) / 2 - bbox[0],
            (IMAGE_HEIGHT - text_height) / 2 - bbox[1],
        )
    except Exception:
        # 일부 특수 문자는 bbox 계산에 실패할 수 있음
        position = (5, 5)

    draw.text(position, char, font=font, fill="black")

    # 이미지를 Numpy 배열로 변환하고 0~1 사이 값으로 정규화
    tensor = np.array(image, dtype=np.float32) / 255.0

    # reshape()의 첫 번째 차원은 제거합니다. 배열에 직접 할당할 것이기 때문입니다.
    return tensor.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 1))


def main():
    """메인 실행 함수"""
    print("Starting Unicode to Tensor conversion...")

    # 실제로 존재하는 폰트 파일 경로 목록 생성
    available_font_paths = []
    print("Checking available fonts...")
    for font_name in tqdm(NOTO_FONTS_NAME, desc="Checking Fonts"):
        path = FONT_PATH.format(font_name)
        if os.path.exists(path):
            available_font_paths.append(path)
    print(f"Found {len(available_font_paths)} available Noto font files.")

    unrendered_chars = []  # 렌더링하지 못한 문자를 저장할 리스트
    rendered_count = 0  # 렌더링에 성공한 문자 수

    # 전체 유니코드 범위를 CHUNK_SIZE에 따라 순회합니다.
    for chunk_index, start_codepoint in enumerate(range(0, 0x110000, CHUNK_SIZE)):
        end_codepoint = min(start_codepoint + CHUNK_SIZE, 0x110000)

        # 청크의 모든 코드 포인트가 제외 대상인지 확인합니다.
        is_fully_excluded = all(
            is_excluded(cp) for cp in range(start_codepoint, end_codepoint)
        )

        if is_fully_excluded:
            print(
                f"\nSkipping chunk {chunk_index} (U+{start_codepoint:04X}-U+{end_codepoint-1:04X}) as it falls entirely within excluded ranges."
            )
            continue

        print(
            f"\nProcessing chunk {chunk_index}: U+{start_codepoint:04X} to U+{end_codepoint-1:04X}"
        )

        # 현재 청크에 대한 텐서를 할당합니다.
        current_chunk_size = end_codepoint - start_codepoint
        chunk_tensor = np.ones(
            (current_chunk_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype=np.float32
        )

        # 현재 청크의 유니코드 범위를 순회합니다.
        for i, code_point in enumerate(
            tqdm(range(start_codepoint, end_codepoint), desc=f"Chunk {chunk_index}")
        ):
            if is_excluded(code_point):
                continue

            char = chr(code_point)

            # 이 문자를 지원하는 폰트 찾기
            for font_path in available_font_paths:
                if is_char_in_font(char, font_path):
                    tensor = render_char_to_tensor(char, font_path)
                    if tensor is not None:
                        # 렌더링된 텐서를 청크 내의 상대적 인덱스에 할당합니다.
                        chunk_tensor[i] = tensor
                        rendered_count += 1
                    break
            else:
                unrendered_chars.append(char)

        # 현재 청크의 결과 텐서를 .npy 파일로 저장합니다.
        output_filename = f"{OUTPUT_FILENAME_BASE}_{chunk_index}.npy"
        print(f"Saving chunk {chunk_index} to {output_filename}...")
        np.save(output_filename, chunk_tensor)
        del chunk_tensor  # 메모리 해제

    print(f"\nSuccessfully rendered {rendered_count} characters in total.")

    # 렌더링하지 못한 문자들을 파일에 저장합니다.
    if unrendered_chars:
        print(
            f"Could not find fonts for {len(unrendered_chars)} characters. "
            f"Saving them to {UNRENDERED_LOG_FILENAME}."
        )
        with open(UNRENDERED_LOG_FILENAME, "w", encoding="utf-8") as f:
            for char in unrendered_chars:
                f.write(f"{char} (U+{ord(char):04X})\n")

    print("Done.")


if __name__ == "__main__":
    main()
