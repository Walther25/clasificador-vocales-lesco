"""Preparacion del dataset - Vocales LESCO.
Convierte y redimensiona las fotos a 128x128 a color y genera el ZIP del dataset.

Entrada: una carpeta con subcarpetas por clase (A/ E/ I/ O/ U/) o un .zip con esa estructura.
Salida:  carpeta 'dataset/' con las imagenes 128x128 y el archivo 'dataset_128.zip'.
"""
import os
import shutil
import zipfile
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

register_heif_opener()

# === Configuracion ===
CLASES = ["A", "E", "I", "O", "U"]
IMG_SIZE = (128, 128)
CROP_CUADRADO = False           # True recorta al centro (sin deformar); False redimensiona directo
PREVIEW_POR_CLASE = 3
EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp")

ENTRADA = "fotos_crudas"        # carpeta con subcarpetas por clase, o un .zip
SALIDA = "dataset"              # carpeta de salida con las imagenes 128x128
ZIP_SALIDA = "dataset_128.zip"  # zip final para entrenar

_previews = []


def procesar(clase, rutas):
    destino = os.path.join(SALIDA, clase)
    os.makedirs(destino, exist_ok=True)
    ya = len(os.listdir(destino))
    n = 0
    for ruta in rutas:
        try:
            original = Image.open(ruta)
            original = ImageOps.exif_transpose(original)
            img = original.convert("RGB")
            if CROP_CUADRADO:
                img = ImageOps.fit(img, IMG_SIZE)
            else:
                img = img.resize(IMG_SIZE)
            n += 1
            img.save(os.path.join(destino, f"{clase}_{ya+n:03d}.png"))
            if sum(1 for p in _previews if p[0] == clase) < PREVIEW_POR_CLASE:
                orig_thumb = original.convert("RGB").copy()
                orig_thumb.thumbnail((256, 256))
                _previews.append((clase, orig_thumb, img.copy()))
        except Exception as e:
            print(f"   no pude procesar '{ruta}': {e}")
    print(f"   {clase}: {n} fotos nuevas (total {ya+n})")


def carpeta_de_clases(entrada):
    """Devuelve la carpeta que contiene las subcarpetas de clase (extrae el zip si hace falta)."""
    if entrada.lower().endswith(".zip"):
        crudo = "_crudo"
        shutil.rmtree(crudo, ignore_errors=True)
        with zipfile.ZipFile(entrada, "r") as z:
            z.extractall(crudo)
        shutil.rmtree(os.path.join(crudo, "__MACOSX"), ignore_errors=True)
        entrada = crudo
    subdirs = [d for d in os.listdir(entrada)
               if os.path.isdir(os.path.join(entrada, d)) and not d.startswith(".")]
    if len(subdirs) == 1 and subdirs[0] not in CLASES:
        return os.path.join(entrada, subdirs[0])
    return entrada


def main():
    os.makedirs(SALIDA, exist_ok=True)
    raiz = carpeta_de_clases(ENTRADA)
    disponibles = {d.lower(): d for d in os.listdir(raiz)
                   if os.path.isdir(os.path.join(raiz, d))}

    for clase in CLASES:
        if clase.lower() not in disponibles:
            print(f"   no encontre la carpeta de '{clase}', la salto")
            continue
        carpeta = os.path.join(raiz, disponibles[clase.lower()])
        rutas = [os.path.join(carpeta, f) for f in sorted(os.listdir(carpeta))
                 if f.lower().endswith(EXT)]
        procesar(clase, rutas)

    # Control de conversion: original vs 128x128
    if _previews:
        filas = len(_previews)
        plt.figure(figsize=(6, 2.6 * filas))
        for i, (clase, orig, trans) in enumerate(_previews):
            plt.subplot(filas, 2, 2 * i + 1)
            plt.imshow(orig); plt.axis("off")
            plt.title(f"{clase} - original ({orig.width}x{orig.height})", fontsize=9)
            plt.subplot(filas, 2, 2 * i + 2)
            plt.imshow(trans); plt.axis("off")
            plt.title(f"{clase} - 128x128", fontsize=9)
        plt.tight_layout()
        plt.savefig("control_conversion.png", dpi=110)
        plt.close()
        print("Control de conversion guardado en control_conversion.png")

    # Conteo por clase
    print("Fotos por clase:")
    total = 0
    for clase in CLASES:
        d = os.path.join(SALIDA, clase)
        c = len(os.listdir(d)) if os.path.isdir(d) else 0
        total += c
        print(f"  {clase}: {c}")
    print(f"  TOTAL: {total}")

    # Empaquetar en ZIP
    base = ZIP_SALIDA[:-4] if ZIP_SALIDA.endswith(".zip") else ZIP_SALIDA
    shutil.make_archive(base, "zip", ".", SALIDA)
    print("ZIP generado:", base + ".zip")


if __name__ == "__main__":
    main()
