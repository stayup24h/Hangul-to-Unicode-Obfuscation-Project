# 파일 path들 신경써주세요
# 45번 줄 테스트용으로 개수 제한해놨으니 그것도 신경써주세요
# 제미니 덕분에 병렬처리로 for문돌립니다

import json
import numpy as np
from PIL import Image
import os
import concurrent.futures
from functools import partial


def pan_image(img: Image.Image, width: int, height: int):
    # PIL image panning
    # add white pixels up/down evenly and left/right evenly
    img = img.convert("L")  # 흑백으로 변환
    original_width, original_height = img.size
    if width < original_width or height < original_height:
        raise ValueError("Target width/height must be >= original image size")

    left = (width - original_width) // 2
    top = (height - original_height) // 2

    new_image = Image.new(img.mode, (width, height), "white")
    new_image.paste(img, (left, top))

    return new_image


# 고마워 Gemini야
def process_image_path(img_path: str, width: int, height: int):
    # 워커 프로세스에서 실행 — 이미지 열고 pan_image 호출 후 ndarray 반환
    img = Image.open(img_path)
    panned = pan_image(img, width, height)
    return np.asarray(panned, dtype=np.uint8)


def main():
    with open(
        "데이터셋/13.한국어글자체/02.인쇄체_230721_add/printed_data_info.json", "r"
    ) as info:
        annotations = json.loads(info.read())["annotations"]  # 레이블 및 메타데이터

        syllable_annos = [
            # 음절 레이블인지 확인
            a
            for a in annotations[:10000]
            if a["attributes"]["type"] == "글자(음절)"
            # 테스트용 100개만
        ]

        n = len(syllable_annos)
        if n == 0:
            results = np.empty((0, 150, 150), dtype=np.uint8)
        else:
            height, width = 150, 150
            results = np.empty((n, height, width), dtype=np.uint8)

            # 이미지 경로 리스트 생성
            img_paths = []
            for anno in syllable_annos:
                image_name = anno["image_id"]
                img_paths.append(
                    (
                        f"데이터셋/13.한국어글자체/02.인쇄체_230721_add/"
                        f"01_printed_syllable_images/syllables/{image_name}.png"
                    )
                )

            # 병렬 워커 수 설정
            workers = min(8, (os.cpu_count() or 1))

            # ProcessPoolExecutor로 병렬 처리 (IO+CPU 작업에 적합)
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                # ex.map은 입력 순서를 보장하므로 인덱스와 매핑 가능
                for i, arr in enumerate(
                    ex.map(
                        partial(process_image_path, width=width, height=height),
                        img_paths,
                    )
                ):
                    results[i] = arr

    np.save("tensor_printed_1", results)
    print(results.shape)


if __name__ == "__main__":
    main()
