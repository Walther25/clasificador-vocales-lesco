# Reconocimiento de vocales de LESCO con redes neuronales

Un sistema que, a partir de una foto de una mano, reconoce qué **vocal** del alfabeto manual de **LESCO** (la Lengua de Señas Costarricense) se está señalando: **A, E, I, O o U**.

---

## ¿De qué trata este proyecto?

Las personas sordas en Costa Rica se comunican con **LESCO**. Una parte de esa lengua es el **alfabeto dactilológico** (o "deletreo con las manos"): una forma de la mano para cada letra, que se usa sobre todo para escribir nombres o palabras que no tienen una seña propia.

Este proyecto entrena a una computadora para que, al mostrarle la foto de una mano haciendo la seña de una vocal, adivine de qué vocal se trata. Es un pequeño **prototipo de tecnología de accesibilidad**: la misma idea, llevada más lejos, podría ayudar a traducir señas a texto y facilitar la comunicación con la comunidad sorda.

Es un proyecto **académico**, hecho para aprender cómo se construye un sistema de reconocimiento de imágenes de principio a fin: desde tomar las fotos hasta tener un modelo que funciona.

> **Nota:** el sistema reconoce solo señas **estáticas** (la mano quieta en una posición). Las vocales de LESCO se prestan para esto porque no requieren movimiento.

---

## ¿Cómo funciona? (en palabras simples)

La idea básica es sencilla: se le muestran a la computadora **muchas fotos de ejemplo** de cada vocal, y ella aprende sola a distinguir los patrones que diferencian una seña de otra. Ese "aprender de ejemplos" se hace con **redes neuronales**, que son programas inspirados en la forma en que el cerebro reconoce patrones.

En este proyecto se probaron **tres formas distintas** de resolver el problema, para comparar cuál funciona mejor:

- **MLP** — una red neuronal sencilla. Sirve como punto de partida ("línea base") para tener con qué comparar.
- **CNN** — una red *convolucional*, un tipo de red pensada especialmente para imágenes, que aprende a fijarse en formas y bordes.
- **Transfer learning** — en vez de aprender desde cero, se reutiliza un modelo (**MobileNetV2**) que ya fue entrenado con millones de imágenes, y solo se le enseña la parte final para nuestras vocales. Es como partir de alguien que ya sabe "ver" y solo enseñarle lo específico.

---

## Resultados

Cada modelo se evaluó con fotos que **nunca vio durante el entrenamiento**, para medir qué tan bien generaliza. La métrica principal es la **exactitud** (*accuracy*): el porcentaje de fotos que clasifica correctamente.

| Modelo | Exactitud |
|---|---|
| **Transfer Learning (MobileNetV2)** | **82.7 %** |
| MLP (línea base) | 81.3 % |
| CNN (desde cero) | 29.3 % |

**¿Qué significan estos números?**

- El **transfer learning** fue el mejor, como se esperaba: aprovechar un modelo ya entrenado da muy buenos resultados incluso con pocas fotos.
- El **MLP** funcionó sorprendentemente bien como línea base.
- La **CNN entrenada desde cero** fue la que más costó. Esto **no es un error**, sino un resultado conocido: entrenar una red convolucional desde cero necesita **muchísimos datos**, y aquí el conjunto es pequeño (500 fotos). La literatura sobre reconocimiento de gestos señala justamente este reto.

> Como contexto, el Proyecto 1 de este curso (clasificación de granos de arroz con un modelo clásico llamado SVM) alcanzó ~97.7 %, pero era un **problema distinto** y no es directamente comparable con este.

Las gráficas de entrenamiento y las matrices de confusión de cada modelo están en `reports/graficos/`.

---

## El dataset

Las fotos las tomó y organizó el autor del proyecto: **500 imágenes en total, 100 por cada vocal**, a color y redimensionadas a 128×128 píxeles. Se cuidó que hubiera variedad de fondos, iluminación y ángulos, para que el modelo aprenda la **forma de la mano** y no el fondo.

Los detalles completos (cómo se recolectaron, condiciones, limitaciones) están en **`DATASET.md`**.

---

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                 # Fotos originales, sin procesar (una carpeta por vocal)
│   ├── processed/           # Fotos ya convertidas a 128×128 a color
│   └── dataset_128.zip      # El dataset empaquetado, listo para entrenar
├── models/
│   └── C31037_Walther_Barrantes.keras   # El modelo entrenado
├── reports/
│   ├── graficos/            # Curvas de entrenamiento y matrices de confusión
│   └── resultados/          # Salida completa del entrenamiento (métricas)
├── src/
│   ├── preparar_dataset_lesco.ipynb     # 1) Prepara el dataset
│   ├── proyecto2_cnn_multiclase.ipynb   # 2) Entrena y compara los modelos
│   └── clasificar_imagenes.ipynb        # 3) Clasifica fotos nuevas
├── DATASET.md
├── MODEL_CARD.md
├── README.md
└── LICENSE
```

---

## ¿Cómo usarlo?

El flujo tiene **tres pasos**, cada uno con su propio archivo. La forma más fácil es abrir los notebooks (`.ipynb`) en **Google Colab**, que es gratis y no requiere instalar nada.

### 1. Preparar el dataset — `preparar_dataset_lesco`
Toma las fotos originales, las convierte a 128×128 a color (incluso si son HEIC de iPhone) y genera el archivo `dataset_128.zip`.

### 2. Entrenar los modelos — `proyecto2_cnn_multiclase`
Carga ese `dataset_128.zip`, entrena los tres modelos, muestra las gráficas y las métricas, y guarda el mejor modelo como un archivo `.keras`.

> Para el entrenamiento conviene activar la GPU en Colab: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*.

### 3. Clasificar fotos nuevas — `clasificar_imagenes`
Carga el modelo `.keras` ya entrenado y, al darle una foto de una mano, responde con la vocal predicha y su nivel de confianza. Por ejemplo:

```
mano_A.jpg  →  A  (94.6 %)
```

---

## Tecnologías

- **Python**
- **TensorFlow / Keras** — para construir y entrenar las redes neuronales
- **scikit-learn** — para las métricas de evaluación
- **Google Colab** — entorno de ejecución con GPU gratuita

---

## Limitaciones

- Reconoce solo **vocales** y solo señas **estáticas** (no señas con movimiento).
- El dataset es **pequeño** (500 fotos), así que el modelo puede fallar con fondos, manos o condiciones de luz muy distintas a las de las fotos de entrenamiento.
- Es un **prototipo educativo**, no una herramienta lista para uso real.

---

## Contexto

Proyecto 2 del curso de **Inteligencia Artificial aplicada (IE0435)**, carrera de **Ingeniería Eléctrica**, **Universidad de Costa Rica**.

Autor: **Walther Barrantes** — carné C31037.

---

## Licencia

Ver el archivo [`LICENSE`](LICENSE).
