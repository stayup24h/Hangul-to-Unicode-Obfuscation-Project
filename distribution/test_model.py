import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from hangul_to_unicode_obfuscator import H2UObfuscator

# --- File paths ---
DATASET_DIR = "./dataSet/"

# --- Batch size for rendering ---
BATCH_SIZE_RENDER = 1000

# --- Instantiate H2UObfuscator ---
print("Initializing H2UObfuscator...")
obfuscator = H2UObfuscator()
print("H2UObfuscator initialized and model loaded successfully.")

# --- Load hangul labels for generating test data ---
print(f"Loading hangul labels from {DATASET_DIR}...")

# Generate Hangul unicode code points directly (0xAC00 ~ 0xD7A3)
all_hangul_labels = np.arange(0xAC00, 0xD7A4, dtype=np.int32)

# --- Batch processing and evaluation ---
num_samples = len(all_hangul_labels)
num_batches = (num_samples + BATCH_SIZE_RENDER - 1) // BATCH_SIZE_RENDER

total_loss = 0.0
total_acc_first = 0.0
total_acc_middle = 0.0
total_acc_last = 0.0
total_acc_joint = 0.0
total_count = 0

print("Starting batched evaluation...")
for i in range(num_batches):
    start_idx = i * BATCH_SIZE_RENDER
    end_idx = min((i + 1) * BATCH_SIZE_RENDER, num_samples)
    batch_labels = all_hangul_labels[start_idx:end_idx]

    X_batch_list = []
    y_batch_raw_list = []

    print(f"Processing batch {i+1}/{num_batches} (samples {start_idx}-{end_idx-1})...")
    for char_code in batch_labels:
        char = chr(char_code) # Convert unicode code point to character
        tensor = obfuscator.render_char_to_tensor(char)
        if tensor is not None:
            X_batch_list.append(tensor)
            y_batch_raw_list.append(char_code)

    if not X_batch_list:
        print(f"No valid samples in batch {i+1}. Skipping evaluation for this batch.")
        continue


    X_batch = np.array(X_batch_list)
    y_batch_raw = np.array(y_batch_raw_list)

    # 유니코드 한글을 초성, 중성, 종성 인덱스로 분해하여 one-hot(68)로 라벨 생성
    def split_hangul(code):
        code = code - 0xAC00
        cho = code // (21 * 28)
        jung = (code % (21 * 28)) // 28
        jong = code % 28
        return cho, jung, jong

    y_batch = []
    for code in y_batch_raw:
        cho, jung, jong = split_hangul(code)
        cho_onehot = np.eye(19)[cho]
        jung_onehot = np.eye(21)[jung]
        jong_onehot = np.eye(28)[jong]
        y = np.concatenate([cho_onehot, jung_onehot, jong_onehot])
        y_batch.append(y)
    y_batch = np.array(y_batch)

    # Model output is logits, so apply softmax for metrics
    y_pred_logits = obfuscator.model.predict(X_batch,batch_size=256, verbose=0)
    y_pred = tf.nn.softmax(y_pred_logits, axis=-1).numpy()
    # Compute loss (using logits)
    batch_loss = obfuscator.model.loss(y_batch, y_pred_logits).numpy().mean()
    # Compute metrics (using softmaxed predictions)
    batch_acc_first = obfuscator.acc_first(y_batch, y_pred).numpy()
    batch_acc_middle = obfuscator.acc_middle(y_batch, y_pred).numpy()
    batch_acc_last = obfuscator.acc_last(y_batch, y_pred).numpy()
    batch_acc_joint = obfuscator.acc_joint(y_batch, y_pred).numpy()
    

    print(f"  Batch {i+1} results:")
    print(f"    Loss: {batch_loss:.4f}")
    print(f"    Accuracy (First):  {batch_acc_first:.4f}")
    print(f"    Accuracy (Middle): {batch_acc_middle:.4f}")
    print(f"    Accuracy (Last):   {batch_acc_last:.4f}")
    print(f"    Accuracy (Joint):  {batch_acc_joint:.4f}")

    batch_size = len(X_batch_list)
    total_loss += batch_loss * batch_size
    total_acc_first += batch_acc_first * batch_size
    total_acc_middle += batch_acc_middle * batch_size
    total_acc_last += batch_acc_last * batch_size
    total_acc_joint += batch_acc_joint * batch_size
    total_count += batch_size

# Calculate overall averages
if total_count > 0:
    avg_loss = total_loss / total_count
    avg_acc_first = total_acc_first / total_count
    avg_acc_middle = total_acc_middle / total_count
    avg_acc_last = total_acc_last / total_count
    avg_acc_joint = total_acc_joint / total_count
else:
    avg_loss = 0.0
    avg_acc_first = 0.0
    avg_acc_middle = 0.0
    avg_acc_last = 0.0
    avg_acc_joint = 0.0

print("\n--- Model Evaluation Results ---")
print(f"Loss: {avg_loss:.4f}")

metrics = {
    "Accuracy (First)": avg_acc_first,
    "Accuracy (Middle)": avg_acc_middle,
    "Accuracy (Last)": avg_acc_last,
    "Accuracy (Joint)": avg_acc_joint,
}

metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(10, 6))
plt.bar(metric_names, metric_values, color=['blue', 'green', 'purple', 'orange'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Results')
plt.ylim(0, 1) # Accuracy typically ranges from 0 to 1

for i, value in enumerate(metric_values):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom')

plt.show()
