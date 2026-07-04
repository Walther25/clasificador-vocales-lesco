"""Reconocimiento de vocales LESCO con redes neuronales.
Entrena y compara un MLP, una CNN y transfer learning (MobileNetV2) sobre un dataset propio.

Entrada: carpeta 'dataset/' con subcarpetas por clase (o un .zip).
Salida:  modelo .keras y figuras en 'reports/graficos/'.
"""
import os
import shutil
import zipfile
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_fscore_support, accuracy_score)
import pandas as pd
import seaborn as sns

# === Configuracion ===
DATA_DIR = "dataset"               # carpeta con subcarpetas por clase, o un .zip
IMG_SIZE = (128, 128)
BATCH = 32
SEED = 42
EPOCHS = 25
ACC_SVM_P1 = 0.977                 # accuracy del SVM del Proyecto 1 (referencia)
MODELO_NOMBRE = "C31037_Walther_Barrantes"
GRAF_DIR = "reports/graficos"

os.makedirs(GRAF_DIR, exist_ok=True)
keras.utils.set_random_seed(SEED)
print("TensorFlow:", tf.__version__)
print("GPU disponible:", "SI" if tf.config.list_physical_devices("GPU") else "NO")


def _slug(s):
    for a, b in [(" ", "_"), ("-", ""), ("(", ""), (")", "")]:
        s = s.replace(a, b)
    return s.lower()


# Funciones auxiliares
def plot_history(history, titulo):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title(f"{titulo} - Accuracy"); ax[0].set_xlabel("epoca"); ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title(f"{titulo} - Loss"); ax[1].set_xlabel("epoca"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{GRAF_DIR}/{_slug(titulo)}_curvas.png", dpi=110)
    plt.close()
    gap = h["accuracy"][-1] - h["val_accuracy"][-1]
    estado = "posible overfitting" if gap > 0.10 else "ok"
    print(f"{titulo}: accuracy final train={h['accuracy'][-1]:.3f} "
          f"val={h['val_accuracy'][-1]:.3f} (brecha {gap:+.3f} -> {estado})")


resultados = {}


def evaluar(model, ds, nombre):
    y_true, y_pred = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.append(y.numpy())
        y_pred.append(np.argmax(p, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    resultados[nombre] = {"Accuracy": acc, "Precision": prec,
                          "Recall": rec, "F1": f1,
                          "Parametros": model.count_params()}
    print(f"{nombre}:  accuracy={acc:.3f}  precision={prec:.3f}  "
          f"recall={rec:.3f}  F1={f1:.3f}")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.6, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matriz de confusion - {nombre}")
    plt.ylabel("Real"); plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(f"{GRAF_DIR}/{_slug(nombre)}_matriz_confusion.png", dpi=110)
    plt.close()


# === Cargar el dataset (acepta carpeta o .zip) ===
if DATA_DIR.lower().endswith(".zip"):
    _dest = "_dataset"
    shutil.rmtree(_dest, ignore_errors=True)
    with zipfile.ZipFile(DATA_DIR, "r") as z:
        z.extractall(_dest)
    _sub = [d for d in os.listdir(_dest) if os.path.isdir(os.path.join(_dest, d))]
    DATA_DIR = os.path.join(_dest, _sub[0]) if len(_sub) == 1 else _dest

AUTOTUNE = tf.data.AUTOTUNE

# Cargar todas las imagenes sin batch, en orden fijo
full = keras.utils.image_dataset_from_directory(
    DATA_DIR, labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=None, shuffle=False)
class_names = full.class_names
num_classes = len(class_names)
TOTAL = full.cardinality().numpy()

# Barajar una vez antes de dividir
full = full.shuffle(TOTAL, seed=SEED, reshuffle_each_iteration=False)

# Division 70 / 15 / 15
n_train = int(0.70 * TOTAL)
n_val = int(0.15 * TOTAL)
train_ds = (full.take(n_train).cache()
            .shuffle(n_train, seed=SEED, reshuffle_each_iteration=True)
            .batch(BATCH).prefetch(AUTOTUNE))
val_ds = full.skip(n_train).take(n_val).batch(BATCH).cache().prefetch(AUTOTUNE)
test_ds = full.skip(n_train + n_val).batch(BATCH).cache().prefetch(AUTOTUNE)

print("Clases:", class_names)
print(f"Total: {TOTAL}  |  train: {n_train}  val: {n_val}  test: {TOTAL - n_train - n_val}")


def distribucion(ds, nombre):
    cont = collections.Counter()
    for _, y in ds:
        cont.update(y.numpy().tolist())
    print(f"  {nombre:5s}:", {class_names[k]: cont[k] for k in sorted(cont)})


print("Distribucion por clase:")
distribucion(train_ds, "train")
distribucion(val_ds, "val")
distribucion(test_ds, "test")

# Explorar el dataset
plt.figure(figsize=(10, 6))
for images, labels in train_ds.take(1):
    for i in range(min(9, images.shape[0])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]]); plt.axis("off")
plt.tight_layout()
plt.savefig(f"{GRAF_DIR}/muestras_dataset.png", dpi=110)
plt.close()

# Data augmentation (vista previa)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="augmentation")

for images, _ in train_ds.take(1):
    plt.figure(figsize=(10, 6))
    for i in range(9):
        aug = data_augmentation(tf.expand_dims(images[0], 0))
        plt.subplot(3, 3, i + 1)
        plt.imshow(aug[0].numpy().astype("uint8")); plt.axis("off")
    plt.suptitle("Misma imagen con augmentation")
    plt.tight_layout()
    plt.savefig(f"{GRAF_DIR}/augmentation.png", dpi=110)
    plt.close()
    break

early = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# === Modelo A - MLP (baseline) ===
mlp = keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=IMG_SIZE + (3,)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
], name="MLP")

mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print(f"MLP - parametros entrenables: {mlp.count_params():,}")
hist_mlp = mlp.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                   callbacks=[early], verbose=2)
plot_history(hist_mlp, "MLP")
evaluar(mlp, test_ds, "MLP")

# === Modelo B - CNN ===
aug_cnn = keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
], name="aug_suave")

cnn = keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=IMG_SIZE + (3,)),
    aug_cnn,
    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu",
                 kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
], name="CNN")

cnn.compile(optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print(f"CNN - parametros entrenables: {cnn.count_params():,}")
early_cnn = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=15, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6)
hist_cnn = cnn.fit(train_ds, validation_data=val_ds, epochs=60,
                   callbacks=[early_cnn, reduce_lr], verbose=2)
plot_history(hist_cnn, "CNN")
evaluar(cnn, test_ds, "CNN")

# === Modelo C - Transfer learning (MobileNetV2) ===
base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

aug_tl = keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
], name="aug_tl")

tl = keras.Sequential([
    layers.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SIZE + (3,)),
    aug_tl,
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
], name="TransferLearning")

tl.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print(f"Transfer - parametros entrenables: {sum(w.shape.num_elements() for w in tl.trainable_weights):,}")
early_tl = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, restore_best_weights=True)
hist_tl = tl.fit(train_ds, validation_data=val_ds, epochs=30,
                 callbacks=[early_tl], verbose=2)
plot_history(hist_tl, "Transfer Learning")
evaluar(tl, test_ds, "Transfer")

# === Comparacion de modelos ===
tabla = pd.DataFrame(resultados).T[["Accuracy", "Precision", "Recall", "F1", "Parametros"]]
vista = tabla.copy()
for col in ["Accuracy", "Precision", "Recall", "F1"]:
    vista[col] = vista[col].map(lambda x: f"{x:.3f}")
vista["Parametros"] = tabla["Parametros"].map(lambda x: f"{int(x):,}")
print("Comparacion de modelos (set de prueba):\n")
print(vista.to_string())

ganador = tabla["Accuracy"].astype(float).idxmax()
print(f"\nMejor modelo: {ganador}  (accuracy {float(tabla.loc[ganador, 'Accuracy']):.3f})")
print(f"Referencia SVM Proyecto 1: accuracy {ACC_SVM_P1:.3f}")

nombres = list(tabla.index) + ["SVM (P1)"]
accs = [float(a) for a in tabla["Accuracy"]] + [ACC_SVM_P1]
plt.figure(figsize=(6.5, 4))
barras = plt.bar(nombres, accs, color=["#888", "#4c78a8", "#54a24b", "#cc4444"])
plt.ylabel("Accuracy (test)"); plt.ylim(0, 1); plt.title("Comparacion de modelos")
for b, v in zip(barras, accs):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(f"{GRAF_DIR}/comparacion_modelos.png", dpi=110)
plt.close()

modelos = {"MLP": mlp, "CNN": cnn, "Transfer": tl}
y_true, y_pred = [], []
for x, y in test_ds:
    y_true.append(y.numpy())
    y_pred.append(np.argmax(modelos[ganador].predict(x, verbose=0), axis=1))
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
print(f"\nDetalle por clase - {ganador}:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# === Guardar el modelo ===
mejor = {"MLP": mlp, "CNN": cnn, "Transfer": tl}[ganador]
ruta = f"{MODELO_NOMBRE}.keras"
mejor.save(ruta)
print(f"Modelo ganador ({ganador}) guardado en:", ruta)
print("Figuras guardadas en:", GRAF_DIR)
