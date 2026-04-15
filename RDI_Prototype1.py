import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# -----------------------------
# 1. LOAD DATASET (memory-efficient)
# -----------------------------
def load_image_paths(data_dir, max_images=2000):
    paths = []
    labels = []

    classes = ['real', 'fake']

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        files = os.listdir(class_path)[:max_images]

        for f in files:
            paths.append(os.path.join(class_path, f))
            labels.append(label)

    return paths, labels


def parse_image(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label


def build_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# 2. DEGRADATION FUNCTIONS
# -----------------------------
def degrade_image(img, level):
    img = img.numpy()

    if level == 1:
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255)

    elif level == 2:
        img = cv2.GaussianBlur(img, (5,5), 0)

    elif level == 3:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//2, h//2))
        img = cv2.resize(img, (w, h))

    elif level == 4:
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        img = cv2.imdecode(enc, 1)

    return img.astype(np.float32) / 255.0


def apply_degradation_batch(dataset, level):
    degraded_images = []
    labels = []

    for batch_imgs, batch_labels in dataset:
        for img, label in zip(batch_imgs, batch_labels):
            degraded = degrade_image(img, level)
            degraded_images.append(degraded)
            labels.append(label.numpy())

    return np.array(degraded_images), np.array(labels)


# -----------------------------
# 3. PRETRAINED MODEL
# -----------------------------
def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(128,128,3),
        include_top=False,
        weights='imagenet'
    )

    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
def train_model(train_ds):
    model = build_model()

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        train_ds,
        epochs=10,
        callbacks=[early_stop],
        verbose=1
    )

    return model


# -----------------------------
# 5. EVALUATION
# -----------------------------
def evaluate_model(model, images, labels):
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int)

    return np.mean(preds.flatten() == labels)


# -----------------------------
# 6. RDI COMPUTATION
# -----------------------------
def compute_rdi(P0, performance_list):
    deltas = [(P0 - Pi) / P0 for Pi in performance_list]

    rdi = 0
    n = len(deltas)

    for i in range(n - 1):
        rdi += (deltas[i] + deltas[i+1]) / 2

    return rdi / (n - 1)


# -----------------------------
# 7. MAIN PIPELINE
# -----------------------------
def run_rdi_pipeline(data_dir):
    paths, labels = load_image_paths(data_dir)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42
    )

    train_ds = build_dataset(train_paths, train_labels)
    test_ds = build_dataset(test_paths, test_labels)

    model = train_model(train_ds)

    performance_list = []

    for level in range(5):
        X_test, y_test = apply_degradation_batch(test_ds, level)

        acc = evaluate_model(model, X_test, y_test)
        print(f"Level {level} Accuracy: {acc:.4f}")

        performance_list.append(acc)

    P0 = performance_list[0]
    rdi = compute_rdi(P0, performance_list)

    print("\nFinal RDI:", rdi)

    return rdi