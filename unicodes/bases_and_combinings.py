# 각 유니코드 블록마다 Base와 Combining 글자를 분류하여,
# bases_and_combinings.json 파일에 담음

import unicodedata
import numpy as np
from unicode_block import get_unicode_blocks
import json

all_labels = list(np.load("unicodes/all_unicode_labels.npy"))

unicode_blocks = get_unicode_blocks("unicodes/block.txt")

bases_and_combinings_by_blocks = {}

for block_name, _range in unicode_blocks.items():
    print("Processing " + block_name + "...")
    bases = []
    combinings = []
    for code_point in range(_range[0], _range[1] + 1):
        if code_point >= int("40000", 16):
            break
        char = chr(code_point)
        if code_point in all_labels:  # 올바르게 렌더링 되는 이미지 중에 있는가?
            if unicodedata.category(char) in [
                "Lm",
                "Mc",
                "Mn",
                "Me",
                "Pc",
                "Pd",
                "Ps",
                "Pe",
                "Pi",
                "Pf",
                "Po",
                "Sk",
            ]:
                combinings.append(char)
            else:
                bases.append(char)

    if not combinings:  # 조합하는 글자가 없으면 저장 없이 넘어가기
        continue

    bases_and_combinings_by_blocks[block_name] = {
        "bases": bases,
        "combinings": combinings,
    }

with open("unicodes/bases_and_combinings.json", "w") as file:
    file.write(json.dumps(bases_and_combinings_by_blocks))
