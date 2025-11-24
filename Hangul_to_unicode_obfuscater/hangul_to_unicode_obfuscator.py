import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import Model
from sklearn.model_selection import KFold
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import faiss
from PIL import Image, ImageDraw, ImageFont



class H2UObfuscator:
    def __init__(self):
        # --- 구동에 필요한 파일 위치 ---
        self._font_path = "./NotoSansKR-Medium.ttf"
        self.FAISS_UNICODE_INDEX_FILENAME = "./unicode_faiss_index.faiss"
        self.UNICODE_MAP_FILENAME = "./final_labels.npy"
        self.FAISS_HANGLE_INDEX_FILENAME = "./hangle_faiss_index.faiss"
        self.HANGLE_MAP_FILENAME = "./hangle_labels.npy"
        self.MODEL_PATH = "./final.keras"

        self.unicode_labels = np.load(self.UNICODE_MAP_FILENAME, allow_pickle=True)
        self.hangle_labels = np.load(self.HANGLE_MAP_FILENAME, allow_pickle=True)

        # --- 이미지 생성기 설정 ---
        self.IMAGE_HEIGHT = 128
        self.IMAGE_WIDTH = 128
        self.FONT_SIZE = 96  # 이미지 크기에 맞게 조절 필요

        # --- OCR 모델 설정 ---
        L_SLICE = slice(0, 19)
        V_SLICE = slice(19, 40)
        T_SLICE = slice(40, 68)
        
        # --- 모델 로드 ---
        def __hangul_loss(self, y_true, logits):
            cho  = tf.nn.softmax_cross_entropy_with_logits(labels=y_true[:, L_SLICE], logits=logits[:, L_SLICE])
            jung = tf.nn.softmax_cross_entropy_with_logits(labels=y_true[:, V_SLICE], logits=logits[:, V_SLICE])
            jong = tf.nn.softmax_cross_entropy_with_logits(labels=y_true[:, T_SLICE], logits=logits[:, T_SLICE])
            return cho + jung + jong

        def __acc_first(self, y_true, y_pred):
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true[:, L_SLICE], -1),
                                                tf.argmax(y_pred[:, L_SLICE], -1)), tf.float32))
        def __acc_middle(self, y_true, y_pred):
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true[:, V_SLICE], -1),
                                                tf.argmax(y_pred[:, V_SLICE], -1)), tf.float32))
        def __acc_last(self, y_true, y_pred):
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true[:, T_SLICE], -1),
                                                tf.argmax(y_pred[:, T_SLICE], -1)), tf.float32))
        def __acc_joint(self, y_true, y_pred):
            l_ok = tf.equal(tf.argmax(y_true[:, L_SLICE], -1), tf.argmax(y_pred[:, L_SLICE], -1))
            v_ok = tf.equal(tf.argmax(y_true[:, V_SLICE], -1), tf.argmax(y_pred[:, V_SLICE], -1))
            t_ok = tf.equal(tf.argmax(y_true[:, T_SLICE], -1), tf.argmax(y_pred[:, T_SLICE], -1))
            return tf.reduce_mean(tf.cast(l_ok & v_ok & t_ok, tf.float32))

        self.model = tf.keras.models.load_model(
            self.MODEL_PATH,
            custom_objects={
                "hangul_loss": __hangul_loss,
                "acc_first": __acc_first,
                "acc_middle": __acc_middle,
                "acc_last": __acc_last,
                "acc_joint": __acc_joint,
            }
        )
        

    def render_char_to_tensor(self, char):
        """문자를 이미지로 렌더링하고 Numpy 텐서로 변환합니다."""
        try:
            font = ImageFont.truetype(self._font_path, self.FONT_SIZE)
        except IOError:
            # Pillow가 폰트 파일을 열지 못하는 경우
            return None

        image = Image.new("L", (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(image)

        # 문자를 이미지 중앙에 정렬하기 위해 바운딩 박스 계산
        try:
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = (
                (self.IMAGE_WIDTH - text_width) / 2 - bbox[0],
                (self.IMAGE_HEIGHT - text_height) / 2 - bbox[1],
            )
        except Exception:
            # 일부 특수 문자는 bbox 계산에 실패할 수 있음
            position = (5, 5)

        draw.text(position, char, font=font, fill="black")

        # 이미지를 Numpy 배열로 변환하고 0~1 사이 값으로 정규화
        tensor = np.array(image, dtype=np.float32) / 255.0

        # reshape()의 첫 번째 차원은 제거합니다. 배열에 직접 할당할 것이기 때문입니다.
        return tensor.reshape((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1))
    
    def ocr(self, character):
        X = self.render_char_to_tensor(character)                               # shape: (N, 128,128,1)

        if X.ndim == 3:
            X = np.expand_dims(X, -1)
        X = X.astype(np.float32)
        if np.max(X) > 1.5:
            X /= 255.0

        # 모델 예측
        preds = self.model.predict(X, batch_size=256, verbose=1)  # (N,68)

        return preds

    def find_similar_unicode(self, korean_char, k=5):
        """한글 문자를 입력받아 FAISS 유니코드 인덱스에서 시각적으로 가장 유사한 상위 k개의 유니코드 문자를 반환합니다.
        """

        self.char_vector = self.ocr(korean_char)


        self.D, self.I = self.index.search(self.char_vector, k)

        results = []

        for rank, (idx, distance) in enumerate(zip(self.I[0], self.D[0])):
            if idx >= 0:
                try:
                    code_point = self.unicode_labels[idx]
                    unicode_char = chr(code_point)
                except Exception:
                    code_point = None
                    unicode_char = ""

                results.append({
                    "Rank": rank + 1,
                    "Character": unicode_char,
                    "Similarity_Distance": float(distance)
                })

        return results
    
    def find_similar_hangle(self, unicode_char, k=5):
        """유니코드 문자를 입력받아 FAISS 한글 인덱스에서 시각적으로 가장 유사한 상위 k개의 한글 문자를 반환합니다.
        """

        self.char_vector = self.ocr(unicode_char)

        self.D, self.I = self.index.search(self.char_vector, k)

        results = []
        for rank, (idx, distance) in enumerate(zip(self.I[0], self.D[0])):
            if idx >= 0:
                try:
                    code_point = self.hangle_labels[idx]
                    hangle_char = chr(code_point)
                except Exception:
                    code_point = None
                    hangle_char = ""

                results.append({
                    "Rank": rank + 1,
                    "Character": hangle_char,
                    "Similarity_Distance": float(distance)
                })

        return results