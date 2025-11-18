# 데이터셋 중 모든 데이터가 0인 데이터가 잘못 만들어짐

import numpy as np

for k in range(1000):
    tensor = np.load(f"unicode_tensors_{k}.npy")
    n = 0
    first = 0
    allow = True
    for j, matrix in enumerate(tensor):
        if not matrix.any():
            n += 1
            if allow:
                first = j
                allow = False
    if n:
        print(k, "번째 파일, ", first, "이후 총 ", n, "개의 흰 파일이 있습니다.")
    else:
        print(k, "번째 파일에는 흰 파일이 없습니다.")
    print(tensor.shape)
