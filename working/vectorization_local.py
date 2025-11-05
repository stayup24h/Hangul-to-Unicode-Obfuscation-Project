import os, json, tensorflow as tf
import numpy as np
from one_hot_hangul import one_hot_hangul
from typing import Literal

ROOT_DIR = r"C:\Users\user\Desktop\기학기\val"
OUT_DIR = r"C:\Users\user\Desktop\기학기\ValData"
BATCH_SIZE = 256
IMG_SIZE = 150
SHARD_SIZE = 30000
SAVE_DTYPE: Literal["uint8" "float32"]="uint8"

def read_one_hot_from_json_py(json_path_tensor):
    import os, json
    path = json_path_tensor.numpy().decode("utf-8")
    path = os.path.normpath(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ch = data["text"]["letter"]["value"]
    return one_hot_hangul(ch).astype(np.float32)

def read_one_hot_from_json_tf(json_path: tf.Tensor) -> tf.Tensor:
    vec = tf.py_function(
        func=read_one_hot_from_json_py,
        inp=[json_path],
        Tout=tf.float32
    )
    vec.set_shape((68,))
    return vec

def load_and_preprocess_image(img_path: tf.Tensor) -> tf.Tensor:
    """임의 해상도의 JPG/PNG → [150,150,1] float32[0,1]"""
    bytes_ = tf.io.read_file(img_path)
    # decode_image는 포맷 자동감지(jpeg/png 등), eager/graph 모두 동작
    img = tf.image.decode_image(bytes_, channels=1, expand_animations=False)  # [H,W,1], uint8
    img.set_shape([None, None, 1])
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)   # float32
    img = tf.cast(img, tf.float32) / 255.0                    # [0,1]
    return img

def to_json_path(img_path: tf.Tensor) -> tf.Tensor:
    """
    이미지 경로 → 동일 파일명 .json 경로
    확장자(.jpg/.jpeg/.JPG/.JPEG 등)를 제거하고 .json으로 치환
    """
    # 디렉토리 + 파일명 분리
    p = tf.strings.regex_replace(img_path, r"\\", "/")        # 정규식 표현
    dirname  = tf.strings.regex_replace(p, r"/[^/]+$", "/")   # 
    basename = tf.strings.regex_replace(p, r"^.*/", "")       # 파일명만
    stem     = tf.strings.regex_replace(basename, r"\.[^.]+$", "")  # 확장자 제거
    return tf.strings.join([dirname, stem, ".json"])

def make_pair(img_path: tf.Tensor):
    json_path = to_json_path(img_path)
    img = load_and_preprocess_image(img_path)
    y = read_one_hot_from_json_tf(json_path)   # (68,)
    return img, y

def build_dataset(root_dir: str, batch_size=256, shuffle=True, cache=True) -> tf.data.Dataset:
    # 하위 폴더/*/*.jpg | *.jpeg | (대소문자 포함)
    pats = [
        os.path.join(root_dir, "*", "*.jpg"),
        os.path.join(root_dir, "*", "*.jpeg"),
        os.path.join(root_dir, "*", "*.JPG"),
        os.path.join(root_dir, "*", "*.JPEG"),
    ]
    ds = tf.data.Dataset.list_files(pats, shuffle=shuffle)
    ds = ds.map(make_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:   ds = ds.cache()
    if shuffle: ds = ds.shuffle(4096)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def save_dataset_to_numpy_shards(
    dataset: tf.data.Dataset,
    out_dir: str,
    shard_size: int = 50_000,
    save_dtype: Literal["uint8","float32"] = "uint8",
    verbose: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    H, W, C = IMG_SIZE, IMG_SIZE, 1

    # 버퍼(메모리) 준비
    if save_dtype == "uint8":
        X_buf = np.empty((shard_size, H, W, C), dtype=np.uint8)
    else:
        X_buf = np.empty((shard_size, H, W, C), dtype=np.float32)
    Y_buf = np.empty((shard_size, 68), dtype=np.float32)

    shard_idx = 0
    in_shard = 0
    total = 0

    def flush():
        nonlocal shard_idx, in_shard, X_buf, Y_buf
        np.save(os.path.join(out_dir, f"X_shard_{shard_idx:03d}.npy"), X_buf[:in_shard])
        np.save(os.path.join(out_dir, f"Y_shard_{shard_idx:03d}.npy"), Y_buf[:in_shard])
        if verbose:
            print(f"saved shard {shard_idx:03d}: {in_shard} samples")
        shard_idx += 1
        in_shard = 0
        # 새 버퍼
        if save_dtype == "uint8":
            X_buf = np.empty((shard_size, H, W, C), dtype=np.uint8)
        else:
            X_buf = np.empty((shard_size, H, W, C), dtype=np.float32)
        Y_buf = np.empty((shard_size, 68), dtype=np.float32)

    for bi, (imgs, labels) in enumerate(dataset):
        X = imgs.numpy()      # (B,150,150,1) float32 [0,1]
        Y = labels.numpy()    # (B,68) float32

        if save_dtype == "uint8":
            X = (np.clip(X, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            X = X.astype(np.float32, copy=False)
        Y = Y.astype(np.float32, copy=False)

        B = X.shape[0]
        off = 0
        while off < B:
            take = min(shard_size - in_shard, B - off)
            X_buf[in_shard:in_shard+take] = X[off:off+take]
            Y_buf[in_shard:in_shard+take] = Y[off:off+take]
            in_shard += take
            off += take
            total += take
            if in_shard == shard_size:
                flush()

        if verbose and (bi % 50 == 0):
            print(f"[batch {bi}] total={total}, shard={shard_idx:03d}, in_shard={in_shard}")

    if in_shard > 0:
        flush()

    if verbose:
        print(f"Done. Total samples saved: {total}")
        
if __name__ == "__main__":
    print("→ Building dataset from:", ROOT_DIR)
    ds = build_dataset(ROOT_DIR, batch_size=BATCH_SIZE, shuffle=True, cache=True)

    print("→ Saving shards to:", OUT_DIR)
    save_dataset_to_numpy_shards(
        ds,
        out_dir=OUT_DIR,
        shard_size=SHARD_SIZE,
        save_dtype=SAVE_DTYPE,
        verbose=True,
    )

    print("All done.")