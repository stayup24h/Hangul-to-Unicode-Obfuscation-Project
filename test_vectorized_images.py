# 'tensor_printed_1.npy' numpy 오브젝트 가져와서
# 첫 이미지 세 개만 PIL로 띄웁니다

import numpy as np
from PIL import Image
import random

loaded = np.load("tensor_printed_2.npy")
# 로그
print(loaded)
print(loaded.shape)
print(type(loaded))

# 첫 세 개 이미지 PIL로 표시
n_show = min(3, loaded.shape[0])
random_num = random.randint(0, 10000)
print(random_num)
for i in range(n_show):
    arr = loaded[random_num + i]
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.show()
