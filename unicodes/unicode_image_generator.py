import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from tqdm import tqdm

from search_fonts import NOTO_FONTS_NAME

# 현재 이 파일의 위치
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ! 폰트 파일 위치 (설정 필요)
FONT_PATH: str = os.path.join(
    SCRIPT_DIR, "..", "..", r"notofonts\fonts\{0}\full\ttf\{0}-Regular.ttf"
)

# --- 설정 ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
FONT_SIZE = 96  # 이미지 크기에 맞게 조절 필요
OUTPUT_FILENAME_BASE = "unicode_tensors"  # unicode_tensors_{k}.npy
OUTPUT_LABEL_BASE = "unicode_labels"  # unicode_labels_{k}.npy
UNRENDERED_LOG_FILENAME = (
    "unrendered_characters.txt"  # 렌더 안 되는 빈 유니코드 목록 저장
)

# 텐서 하나로 만들 유니코드 범위 크기
# 0x10000 (65536개씩) 약 1GB, 4개 파일
# 0x04000 (16384개씩) 약 256MB, 16개 파일
# 0x01000 (4096개씩) 약 64MB, 64개 파일
CHUNK_SIZE = 0x04000
# 중간에 중단해도, 이 값을 수정하여 START_INDEX번째 파일부터 돌릴 수 있음
START_INDEX = 0

# --- 유니코드 제외 범위 ---
# C0/C1 제어 문자, 서로게이트, 사설 영역 등은 렌더링할 수 없거나 의미가 없습니다.
EXCLUDE_RANGES = [
    (0x0000, 0x001F),  # C0 Controls
    (0x007F, 0x009F),  # C1 Controls
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    (0xD800, 0xDFFF),  # High and Low Surrogates
    (0xE000, 0xF8FF),  # Private Use Area
    (0xFFF0, 0xFFFF),  # Specials
    # 애초에 x40000까지만 의미있어서 여기까진 순회하면 됨
    # (0xF0000, 0xFFFFD),  # Supplementary Private Use Area-A
    # (0x100000, 0x10FFFD),  # Supplementary Private Use Area-B
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
    print("Starting Unicode to Tensor conversion...")

    # 실제로 존재하는 폰트 파일 경로 목록 생성
    available_font_paths = []
    unavailable_font_names = []
    print("Checking available fonts...")
    for font_name in tqdm(NOTO_FONTS_NAME, desc="Checking Fonts"):
        path = FONT_PATH.format(font_name)
        if os.path.exists(path):
            available_font_paths.append(path)
        else:
            unavailable_font_names.append(font_name)
    print(f"Found {len(available_font_paths)} available Noto font files.")
    print("Unavailable fonts: ", unavailable_font_names)
    total_rendered_count = 0  # 전체 렌더링에 성공한 문자 수
    unrendered_chars = []  # 렌더링하지 못한 문자를 저장할 리스트

    # 전체 유니코드 범위를 순회
    for chunk_index, start_codepoint in enumerate(
        range(START_INDEX, 0x40000, CHUNK_SIZE)
    ):
        end_codepoint = min(start_codepoint + CHUNK_SIZE, 0x40000)

        # 청크의 모든 코드 포인트가 제외 대상인지 확인
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

        # 현재 청크에 대한 유니코드 이미지 텐서와 그 레이블을 저장할 리스트
        chunk_tensors = []
        chunk_labels = []
        chunk_rendered_count = 0

        # 현재 청크의 유니코드 범위를 순회합니다.
        for code_point in tqdm(
            range(start_codepoint, end_codepoint), desc=f"Chunk {chunk_index}"
        ):
            if is_excluded(code_point):
                continue

            char = chr(code_point)

            # 이 문자를 지원하는 폰트 찾기
            for font_path in available_font_paths:
                if is_char_in_font(char, font_path):
                    tensor = render_char_to_tensor(char, font_path)
                    if tensor is not None:
                        # 투명한 문자가 아닐 경우에만
                        if not np.all(tensor == 1.0):
                            chunk_tensors.append(tensor)
                            chunk_labels.append(code_point)
                            chunk_rendered_count += 1
                        else:
                            print(
                                f"Skipping transparent character U+{code_point:04X} from {font_path}"
                            )
                    break
            else:
                unrendered_chars.append(char)

        # 현재 청크의 결과를 저장
        if chunk_tensors:  # 렌더링된 문자가 있는 경우에만 저장
            chunk_tensor = np.array(chunk_tensors, dtype=np.float32)
            chunk_label = np.array(chunk_labels, dtype=np.uint32)

            output_filename_tensor = f"{OUTPUT_FILENAME_BASE}_{chunk_index}.npy"
            output_filename_label = f"{OUTPUT_LABEL_BASE}_{chunk_index}.npy"
            print(
                f"Saving chunk {chunk_index} ({chunk_rendered_count} characters) to {output_filename_tensor} and {output_filename_label}..."
            )
            np.save(output_filename_tensor, chunk_tensor)
            np.save(output_filename_label, chunk_label)
            total_rendered_count += chunk_rendered_count
        else:
            print(f"No characters rendered in chunk {chunk_index}. Skipping file save.")

    print(f"\nSuccessfully rendered {total_rendered_count} characters in total.")

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
