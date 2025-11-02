import json
import numpy as np
from PIL import Image
import os
import concurrent.futures
from functools import partial
from one_hot_hangul import one_hot_hangul


### 데이터셋 어노테이션 파일(printed_data_info.json)과 데이터셋 폴더(이미지들) 아래 입력할것
### PANNING 픽셀 설정 이미지 파일이 panning할 수 없을 정도로 크면 main에서 알아서 무시됨
### 5만 개 파일씩 나눠서 벡터화되며 레이블도 저장됨

### tensor_printed_i.npy(이미지 텐서)를 내놓음
### tensor_printed_i_annotations.json(어노테이션)을 내놓음

LABEL_PATH = "AI_hub/13.한국어글자체/02.인쇄체_230721_add/printed_data_info.json"
IMAGES_PATH = (
    "AI_hub/13.한국어글자체/02.인쇄체_230721_add/01_printed_syllable_images/syllables/"
)
PANNING_WIDTH = 150
PANNING_HEIGHT = 150


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


def process_image_path(img_path: str, width: int, height: int):
    # 워커 프로세스에서 실행 — 이미지 열고 pan_image 호출 후 ndarray 반환
    img = Image.open(img_path)
    panned = pan_image(img, width, height)
    return np.asarray(panned, dtype=np.uint8)


def main():
    with open(LABEL_PATH, "r") as info:
        printed_data_info = json.loads(info.read())
        images = printed_data_info["images"]  # 이미지 width/height 정보
        annotations = printed_data_info["annotations"]  # 레이블 및 메타데이터

    for i in range(0, len(annotations) // 50000):
        valid_annotations = []
        for k in range(50000 * i, 50000 * (i + 1)):
            if (
                annotations[k]["attributes"]["type"]
                == "글자(음절)"  # 한 글자 이미지이며
                and images[k]["width"] <= PANNING_WIDTH  # 너비/높이가 모두 panning 미만
                and images[k]["height"] <= PANNING_HEIGHT
            ):

                valid_annotations.append(annotations[k])

        print("Vectorization Start")
        n = len(valid_annotations)
        if n == 0:
            # results = np.empty((0, PANNING_WIDTH, PANNING_HEIGHT), dtype=np.uint8)
            labels = np.empty((0, 68), dtype=np.bool_)
        else:
            # width, height = PANNING_HEIGHT, PANNING_WIDTH
            # results = np.empty((n, width, height), dtype=np.uint8)
            labels = np.empty((n, 68), dtype=np.bool_)

            # 이미지 경로 리스트 생성
            # img_paths = []
            for p, anno in enumerate(valid_annotations):
                # image_name = anno["image_id"]
                # img_paths.append((IMAGES_PATH + f"{image_name}.png"))
                labels[p] = one_hot_hangul(anno["text"])

            # # 병렬 워커 수 설정
            # workers = min(8, (os.cpu_count() or 1))

            # # ProcessPoolExecutor로 병렬 처리 (IO+CPU 작업에 적합)
            # with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            #     # ex.map은 입력 순서를 보장하므로 인덱스와 매핑 가능
            #     for j, arr in enumerate(
            #         ex.map(
            #             partial(process_image_path, width=width, height=height),
            #             img_paths,
            #         )
            #     ):
            #         if (j + 1) % 1000 == 0:
            #             print(f"{j+1}/50000")
            #         results[j] = arr

        # np.save(f"tensor_printed_{i}", results)
        # print(f"tensor_printed_{i} saved!")
        # print(results.shape)

        np.save(f"labels_printed_{i}", labels)
        print(f"labels_printed_{i} saved!")
        print(labels.shape)

        # with open(
        #     f"tensor_printed_{i}_annotations.json", "w", encoding="utf-8"
        # ) as annos:
        #     annos.write(json.dumps(valid_annotations, ensure_ascii=False))
        #     print(f"tensor_printed_{i}_annotations saved!")


if __name__ == "__main__":
    main()
