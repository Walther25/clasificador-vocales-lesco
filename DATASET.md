# DATASET.md — Reconocimiento de vocales de LESCO

## Resumen

| Característica | Valor |
|---|---|
| Total de imágenes | 500 |
| Clases | A, E, I, O, U (vocales del alfabeto dactilológico de LESCO) |
| Balance | 100 por clase — 20 % cada una |
| Resolución | 128 × 128 px |
| Tipo de píxel | Color RGB (3 canales, 0–255) |
| Formato final | PNG, organizado en una carpeta por clase |
| Archivos fuente | `data/raw/` (originales) → `data/processed/` (PNG 128×128) → `data/dataset_128.zip` |

---

## Recolección

Las imágenes fueron tomadas por el autor con la cámara de un teléfono, capturando las **señas estáticas** de las cinco vocales del alfabeto manual de LESCO (una mano formando cada letra).

Para cada vocal se tomaron **100 fotos**, variando de forma deliberada:

- **Fondo** — paredes, escritorios, ropa y distintos entornos.
- **Iluminación** — luz natural y artificial, con y sin sombras.
- **Ángulo y distancia** — la mano de frente y ligeramente de lado, más cerca y más lejos.

Esa variedad es intencional: obliga al modelo a fijarse en la **forma de la mano** y no en el fondo, lo que ayuda a que generalice mejor.

---

## Preprocesamiento

A diferencia del Proyecto 1 —donde las imágenes se pasaban a blanco y negro, se binarizaban y se aplanaban a un vector—, aquí las fotos se mantienen **a color y con su estructura 2D**, porque la red convolucional aprende directamente de las formas espaciales de la imagen.

El paso de las fotos originales al dataset final se hace con el script `preparar_dataset_lesco`:

1. **Conversión** — de cualquier formato (incluido **HEIC** de iPhone) a PNG, usando `pillow-heif`.
2. **Corrección de orientación** — se aplica la rotación guardada en los metadatos EXIF, para que las fotos del celular no queden giradas.
3. **Color** — se asegura el modo RGB (3 canales).
4. **Redimensionado** — a 128×128 px. Por defecto se redimensiona directo; opcionalmente se recorta al centro para no deformar (opción `CROP_CUADRADO`).
5. **Etiquetado** — la clase se deduce automáticamente de la carpeta en la que está cada foto (`A/`, `E/`, ...).

### Convención de nombres

| Patrón de archivo | Clase |
|---|---|
| `A_001.png` … `A_100.png` | A |
| `E_001.png` … `E_100.png` | E |
| `I_001.png` … `I_100.png` | I |
| `O_001.png` … `O_100.png` | O |
| `U_001.png` … `U_100.png` | U |

No hubo etiquetado manual: la clase es inherente a la carpeta y al nombre asignado durante el procesamiento.

---

## Estructura de archivos

```
data/
├── raw/                 # Fotos originales, sin procesar
│   ├── A/  E/  I/  O/  U/
├── processed/           # Fotos convertidas a 128×128 a color
│   ├── A/  E/  I/  O/  U/
└── dataset_128.zip      # El contenido de processed/ empaquetado, listo para entrenar
```

---

## Limitaciones

- **Tamaño reducido (500 imágenes).** Es un dataset pequeño; limita lo que puede aprender un modelo entrenado desde cero y su capacidad de generalizar a casos nuevos.
- **Solo vocales estáticas.** No incluye consonantes ni señas con movimiento, que requerirían video en lugar de imágenes fijas.
- **Diversidad limitada.** Las fotos provienen de una sola fuente y un rango acotado de entornos, así que el modelo puede rendir peor con manos, tonos de piel o fondos muy distintos a los del dataset.
- **Condiciones no controladas.** La iluminación, el ángulo y la distancia varían entre fotos. Esto se hizo a propósito para ganar robustez, pero también introduce ruido.
- **Posible deformación por el redimensionado.** Pasar una foto rectangular a un cuadrado de 128×128 la estira un poco; se puede mitigar activando `CROP_CUADRADO`, que recorta al centro en vez de deformar.
