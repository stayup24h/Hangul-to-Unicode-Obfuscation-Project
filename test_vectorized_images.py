# 'tensor_printed_1.npy' numpy 오브젝트 가져와서
# 첫 이미지 세 개만 PIL로 띄웁니다

import numpy as np
from PIL import Image

loaded = np.load("tensor_printed_1.npy")
# 로그
print(loaded)
print(loaded.shape)
print(type(loaded))

# 첫 세 개 이미지 PIL로 표시
n_show = min(3, loaded.shape[0])
for i in range(n_show):
    arr = loaded[i]
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.show()
