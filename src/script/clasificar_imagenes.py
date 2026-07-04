"""Clasificacion de vocales LESCO.
Carga un modelo .keras entrenado y predice la vocal de nuevas imagenes.

Uso:
    python clasificar_imagenes.py imagen1.jpg imagen2.jpg ...
    python clasificar_imagenes.py            (modo interactivo: pide rutas hasta 'q')
"""
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array

# === Configuracion ===
MODELO = "C31037_Walther_Barrantes.keras"   # ruta al modelo entrenado
CLASES = ["A", "E", "I", "O", "U"]          # orden alfabetico, como las numero Keras

modelo = keras.models.load_model(MODELO)
IMG_SIZE = tuple(modelo.input_shape[1:3])
print("Modelo cargado:", MODELO, "| tamano:", IMG_SIZE, "| clases:", CLASES)


def clasificar(ruta):
    img = load_img(ruta, target_size=IMG_SIZE)
    x = np.expand_dims(img_to_array(img), axis=0)     # 0-255; el modelo reescala solo
    probs = modelo.predict(x, verbose=0)[0]
    pred = CLASES[int(np.argmax(probs))]
    conf = 100 * float(np.max(probs))
    print(f"{ruta}  ->  {pred}  ({conf:.1f}%)")
    for c, p in zip(CLASES, probs):
        print(f"     {c}: {100 * p:.1f}%")


def main():
    args = sys.argv[1:]
    if args:
        for ruta in args:
            try:
                clasificar(ruta)
            except Exception as e:
                print("Error con", ruta, ":", e)
    else:
        while True:
            ruta = input("Ruta de la imagen (q para salir): ").strip()
            if ruta.lower() == "q":
                break
            if ruta:
                try:
                    clasificar(ruta)
                except Exception as e:
                    print("Error:", e)


if __name__ == "__main__":
    main()
