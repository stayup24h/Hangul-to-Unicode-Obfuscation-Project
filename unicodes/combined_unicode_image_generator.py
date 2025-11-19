import os

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, "lib")

# lib 폴더를 DLL 검색 경로에 추가
if os.path.exists(lib_path):
    print(f"DLL 경로 추가: {lib_path}")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib_path)
    os.environ["PATH"] = lib_path + os.pathsep + os.environ["PATH"]
else:
    print("Warning: lib 폴더를 찾을 수 없습니다.")

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from fontTools.ttLib import TTFont
from tqdm import tqdm

# 기존 combination.py의 클래스
from combination import UniversalScriptCombiner
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
OUTPUT_FILENAME_BASE = "combined_unicode_tensors"
OUTPUT_LABEL_BASE = "combined_unicode_labels"
UNRENDERED_LOG_FILENAME = "unrendered_combined_characters.txt"

# 하나의 파일로 저장할 이미지 개수
CHUNK_SIZE = 4096

# --- 유니코드 제외 범위 (Base 문자에 대해 적용) ---
# C0/C1 제어 문자, 한글 자모/음절, 사설 영역 등
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
            font_cache[font_path] = TTFont(font_path, lazy=True)
        except Exception:
            font_cache[font_path] = None
            return False

    font = font_cache[font_path]
    if font is None:
        return False

    for cmap in font.get("cmap").tables:
        if ord(char) in cmap.cmap:
            return True
    return False


def render_char_to_tensor(char, font_path):
    """문자를 이미지로 렌더링하고 Numpy 텐서로 변환합니다. (잘림 방지 기능 추가)"""
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except IOError:
        return None

    # 1. 문자를 충분히 큰 임시 캔버스에 렌더링합니다.
    # Pillow의 textbbox는 때때로 부정확하므로, 여유 공간을 둡니다.
    temp_size = (IMAGE_WIDTH * 3, IMAGE_HEIGHT * 3)
    temp_image = Image.new("L", temp_size, "white")
    temp_draw = ImageDraw.Draw(temp_image)

    # 중앙에 대략적으로 렌더링
    temp_draw.text((IMAGE_WIDTH, IMAGE_HEIGHT), char, font=font, fill="black")

    # 2. 실제 그려진 글자의 경계 상자(bounding box)를 찾습니다.
    # 이미지를 numpy 배열로 변환하여 글자가 그려진 부분(흰색이 아닌 부분)을 찾습니다.
    temp_array = np.array(temp_image)
    non_white_pixels = np.where(temp_array < 255)

    if non_white_pixels[0].size == 0:  # 그려진 내용이 없으면 (투명 문자 등)
        return None

    top, bottom = np.min(non_white_pixels[0]), np.max(non_white_pixels[0])
    left, right = np.min(non_white_pixels[1]), np.max(non_white_pixels[1])

    # 3. 경계 상자를 이용해 글자 부분만 잘라냅니다.
    cropped_image = temp_image.crop(
        (int(left), int(top), int(right + 1), int(bottom + 1))
    )

    # 4. 최종 이미지 크기에 맞게 비율을 유지하며 리사이즈합니다.
    final_image = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")

    # 복사본을 만들어 리사이즈 (thumbnail은 원본을 수정함)
    resized_image = cropped_image.copy()
    resized_image.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)

    # 5. 리사이즈된 이미지를 최종 캔버스의 중앙에 붙여넣습니다.
    paste_x = (IMAGE_WIDTH - resized_image.width) // 2
    paste_y = (IMAGE_HEIGHT - resized_image.height) // 2
    final_image.paste(resized_image, (paste_x, paste_y))

    tensor = np.array(final_image, dtype=np.float32) / 255.0
    return tensor.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 1))


def save_chunk(chunk_index, tensors, labels):
    """텐서와 레이블의 청크를 파일로 저장합니다."""
    if not tensors:
        print(f"No characters to save for chunk {chunk_index}.")
        return

    chunk_tensor = np.array(tensors, dtype=np.float32)
    # 레이블은 문자열이므로 dtype=object로 지정
    chunk_label = np.array(labels, dtype=object)

    output_filename_tensor = f"{OUTPUT_FILENAME_BASE}_{chunk_index}.npy"
    output_filename_label = f"{OUTPUT_LABEL_BASE}_{chunk_index}.npy"
    print(
        f"Saving chunk {chunk_index} ({len(tensors)} characters) to {output_filename_tensor} and {output_filename_label}..."
    )
    np.save(output_filename_tensor, chunk_tensor)
    np.save(output_filename_label, chunk_label)


def main():

    isRaqmSupported = features.check_feature(feature="raqm")

    print(f"RAQM 지원: {isRaqmSupported}")
    if not isRaqmSupported:
        return

    # 1. 조합 문자 리스트 생성
    print("Generating combined character list...")
    combiner = UniversalScriptCombiner()
    available_scripts = combiner.list_available_scripts()
    all_combined_array = []
    for script in available_scripts:
        if combined_chars := combiner.get_binary_combinations(script.upper()):
            all_combined_array.extend(combined_chars)
    print(f"Generated {len(all_combined_array)} combined characters in total.")

    # 2. 폰트 파일 준비
    print("\nChecking available fonts...")
    available_font_paths = []
    for font_name in tqdm(NOTO_FONTS_NAME, desc="Checking Fonts"):
        path = FONT_PATH.format(font_name)
        if os.path.exists(path):
            available_font_paths.append(path)
    print(f"Found {len(available_font_paths)} available Noto font files.")

    # 3. 이미지 생성
    print("\nStarting combined Unicode to Tensor conversion...")
    total_rendered_count = 0
    unrendered_chars = []
    chunk_tensors = []
    chunk_labels = []
    chunk_index = 0

    for char in tqdm(all_combined_array, desc="Processing characters"):
        # Base 문자가 제외 대상인지 확인
        if is_excluded(ord(char[0])):
            continue

        rendered_successfully = False
        # 이 문자를 지원하는 폰트 찾기 (Base 문자 기준)
        for font_path in available_font_paths:
            if is_char_in_font(char[0], font_path):
                tensor = render_char_to_tensor(char, font_path)
                if tensor is not None and not np.all(tensor == 1.0):
                    chunk_tensors.append(tensor)
                    chunk_labels.append(char)
                    total_rendered_count += 1
                    rendered_successfully = True
                    break  # 렌더링 성공 시 다음 문자로

        if not rendered_successfully:
            unrendered_chars.append(char)

        # 청크가 다 차면 저장
        if len(chunk_tensors) >= CHUNK_SIZE:
            save_chunk(chunk_index, chunk_tensors, chunk_labels)
            chunk_tensors, chunk_labels = [], []
            chunk_index += 1

    # 마지막 남은 청크 저장
    if chunk_tensors:
        save_chunk(chunk_index, chunk_tensors, chunk_labels)

    print(f"\nSuccessfully rendered {total_rendered_count} characters in total.")

    # 렌더링하지 못한 문자들을 파일에 저장
    if unrendered_chars:
        print(
            f"Could not render {len(unrendered_chars)} characters. "
            f"Saving them to {UNRENDERED_LOG_FILENAME}."
        )
        with open(UNRENDERED_LOG_FILENAME, "w", encoding="utf-8") as f:
            for char in unrendered_chars:
                f.write(f"{char} (U+{ord(char[0]):04X} + U+{ord(char[1]):04X})\n")

    print("Done.")


if __name__ == "__main__":
    main()
