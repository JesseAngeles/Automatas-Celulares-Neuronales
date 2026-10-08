# Limpieza de imágenes térmicas → evolución del simulador

> Las secuencias de embriones se documentan en [`README_embryo.md`](README_embryo.md).

Convierte las secuencias térmicas en vista nadir del dataset **FLAME 3** en evoluciones
`(T, n, n) int32` que el simulador acepta como entrada (`.npy` / `.mat` con la variable
`evolution`, el mismo formato que produce `png_to_evolution.py`).

Hay dos procesos:

| Script | Entrada | Estado |
|---|---|---|
| [`celsius_to_evolution.py`](celsius_to_evolution.py) | GeoTIFF radiométricos en **°C** | **Recomendado.** Aplicado a plot 1 (§2–§3) |
| [`thermal_to_evolution.py`](thermal_to_evolution.py) | JPG en pseudocolor | Primera versión; afectado por el AGC de la cámara (§4) |

## 1. Datos de origen

| | |
|---|---|
| Dataset | [FLAME 3 – Nadir Thermal Plot Subset](https://www.kaggle.com/datasets/brycehopkins/flame-3-nadir-thermal-plot-subset) (`extract_data.py`) |
| Sitio | Hanna Hammock, quema prescrita del 2023-02-28 |
| Cámara | Autel Robotics XT709 (dron), 640×512 |
| Copias locales | `Dataset/plot_N/raw_thermal_jpg`, `Dataset/plot_1/geo_thermal_tiff_celsius` |

| Escenario | JPG | GeoTIFF °C | Inicio → fin | Δt mediana | Huecos > 10 s |
|---|---|---|---|---|---|
| plot 1 | 297 (`IRX_0528`–`0824`) | 296 (`IRX_0529`–`0824`) | 11:46:19 → 12:11:32 (25 min) | 5 s | 15 s y 26 s al inicio |
| plot 2 | 522 (`IRX_0825`–`1346`) | 498 (en dos partes) | 12:41:27 → 13:32:29 (51 min) | 5 s | **460 s** en t 344→345 |
| plot 3 | — | — | — | — | sin imágenes térmicas (solo `raw_jpg_rgb`) |

El orden por nombre de archivo coincide con el orden temporal EXIF y no hay huecos en la
numeración. Los GeoTIFF no traen fecha; se toma del JPG con el mismo número `IRX_XXXX`.

## 2. Proceso con GeoTIFF en °C (`celsius_to_evolution.py`)

### 2.1 Lectura y alineación

Cada `IRX_XXXX_ref_geo.TIFF` es un raster `float32` de 654×531 en **°C absolutos**,
georreferenciado en EPSG:26916 (UTM 16N) con resolución de 0.153 m × 0.162 m por píxel.
**Todos los frames comparten exactamente la misma malla** (mismo `transform`), así que el
movimiento del dron ya está corregido: el píxel (i, j) es el mismo punto del terreno en
todos los frames. El script verifica esto y se detiene si algún frame no coincide.

### 2.2 Recorte de la región válida

Al reproyectar, las zonas que la cámara no vio quedan en **0 °C** (sin dato). La huella de
cada frame es distinta, por lo que se calcula la máscara de píxeles con dato en **todos**
los frames y se toma el **rectángulo alineado a los ejes de mayor área** dentro de ella
(algoritmo del histograma por filas, O(H·W)).

Plot 1: filas 74:490, columnas 53:605 → **416×552 px ≈ 67.5 m × 84.3 m**.

### 2.3 Reducción a n×n con kernels de convolución

Cada frame `X` (416×552) se convoluciona con un kernel 2D separable y se muestrea en una
malla de n×n centros equiespaciados. Al ser separable se escribe como dos matrices:

```
Y_t = K_r · X_t · K_cᵀ        K_r ∈ ℝ^{n×416},  K_c ∈ ℝ^{n×552}
```

La fila i de `K` es el kernel centrado en `c_i = (i + ½)·s − ½`, con paso `s = L / n`
(4.16 en filas, 5.52 en columnas), normalizado para sumar 1. Así una celda de 100×100
representa ≈ 0.84 m × 0.68 m del terreno. Kernels disponibles (`--kernel`):

| Kernel | Peso `w(d)` (d = distancia al centro en px) | Soporte |
|---|---|---|
| `box` | traslape del píxel con la ventana `[c − s/2, c + s/2]` (promedio por área) | s |
| `tent` | `max(0, 1 − |d|/s)` (triangular) | 2s |
| `gaussian` (default) | `exp(−d² / 2σ²)`, σ = s/2 | 6σ = 3s |

Se usa el **gaussiano** por ser el filtro paso-bajas sin *ringing* que mejor evita el
*aliasing* al reducir resolución. Los tres conservan la temperatura media; difieren en
cuánto suavizan los picos (toda la simulación de plot 1):

| | Original 416×552 | box | tent | gaussian |
|---|---|---|---|---|
| Máximo (°C) | 604.6 | 573.5 | 558.6 | 548.2 |
| Percentil 99.9 (°C) | 392.6 | 384.1 | 380.1 | 377.3 |
| Mediana (°C) | 35.5 | 35.6 | 35.7 | 35.8 |

![kernels](docs/thermal/flame3_plot1_celsius_100_kernels.png)

### 2.4 Número de clusters de temperatura

Se agrupan las temperaturas de **todas** las celdas de **todos** los frames (296 × 100 × 100
= 2.96 M valores), de modo que el estado *i* corresponde al mismo rango de °C en todo
momento:

1. Se muestrean 200 000 celdas (semilla 0).
2. Para k = 2 … 10 se ajusta K-Means 1D sobre °C (`n_init=4`) y se registran la
   **inercia** y el **coeficiente de silueta** (10 000 celdas).
3. k se elige con el **método del codo** (criterio *Kneedle*: mayor distancia de la curva
   normalizada a la recta entre sus extremos).
4. La silueta decrece monótonamente (máximo en k = 2, que solo separa "fuego / no fuego"),
   así que se usa como verificación y no como criterio.
5. Se reajusta con el k elegido (`n_init=10`).

Se agrupa en °C lineales: en escala logarítmica K-Means gastaba dos clusters en partir el
suelo a temperatura ambiente (31 °C vs 42 °C) y perdía resolución en el fuego.

### 2.5 Conversión a estados

En 1D, asignar el centroide más cercano equivale a cortar en los **puntos medios** entre
centroides ordenados. Esos umbrales definen los estados (0 = más frío) y se aplican con
`np.digitize` a las temperaturas `n×n`.

## 3. Resultados — plot 1 (°C, gaussian, n = 100)

```bash
cd Simulator/training
../GUI/venv/bin/python preprocessing/celsius_to_evolution.py Dataset/plot_1/geo_thermal_tiff_celsius CA/FLAME3/flame3_plot1_celsius_100
```

**`CA/FLAME3/flame3_plot1_celsius_100.npy` — shape `(296, 100, 100)`, k = 4**

| k | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Inercia (×10⁷) | 9.63 | 4.63 | **2.76** | 1.83 | 1.28 | 0.93 | 0.73 | 0.58 | 0.47 |
| Silueta | 0.874 | 0.815 | **0.746** | 0.676 | 0.634 | 0.633 | 0.628 | 0.623 | 0.612 |

![k](docs/thermal/flame3_plot1_celsius_100_k.png)

| Estado | Centroide | Rango | Fracción | Interpretación |
|---|---|---|---|---|
| 0 | 35.8 °C | < 56 °C | 83.7 % | Suelo / vegetación sin quemar o ya fría |
| 1 | 76.1 °C | 56 – 116 °C | 11.4 % | Zona caliente: precalentamiento o rescoldo |
| 2 | 156.7 °C | 116 – 225 °C | 3.6 % | Combustión latente |
| 3 | 293.5 °C | ≥ 225 °C | 1.3 % | Frente de llama |

![histograma](docs/thermal/flame3_plot1_celsius_100_hist.png)
![frames](docs/thermal/flame3_plot1_celsius_100_frames.png)
![estados](docs/thermal/flame3_plot1_celsius_100_states.png)

La proporción de estados evoluciona de forma suave: el frente cruza el encuadre, la quema
alcanza su máximo alrededor de t ≈ 200–230 (≈ 50 % de celdas en estados 1–3) y después
la zona se enfría. No aparecen los saltos de un frame que produce el AGC en los JPG.

### Opciones

| Opción | Default | Descripción |
|---|---|---|
| `--size` | 100 | Lado n de la malla |
| `--kernel` | `gaussian` | `box`, `tent` o `gaussian` |
| `--nodata` | 0 | Valores ≤ a este se consideran sin dato |
| `--k` | — | Número de clusters; si se omite se elige con el codo |
| `--k-min`, `--k-max` | 2, 10 | Rango de k evaluado |
| `--sample` | 200000 | Celdas para ajustar K-Means |
| `--sil-size` | 10000 | Celdas para la silueta |
| `--seed` | 0 | Semilla |
| `--jpg-dir` | `../raw_thermal_jpg` | JPG de donde leer los timestamps |
| `--fig-dir` | `preprocessing/docs/thermal` | Carpeta de figuras |

Dependencias (en `GUI/venv`): `numpy`, `scipy`, `pillow`, `scikit-learn`, `matplotlib`, `rasterio`.

### Salidas

| Archivo | Contenido |
|---|---|
| `CA/FLAME3/<nombre>.npy` | Estados `(T, n, n) int32` — entrada del simulador |
| `CA/FLAME3/<nombre>.mat` | Lo mismo, variable `evolution` |
| `CA/FLAME3/<nombre>_celsius.npy` | Temperaturas `(T, n, n) float32` después del kernel, antes de agrupar |
| `CA/FLAME3/<nombre>.json` | Archivos, timestamps, recorte, kernel, centroides, umbrales, fracciones y curvas de k |
| `docs/thermal/<nombre>_k.png` | Codo y silueta |
| `docs/thermal/<nombre>_kernels.png` | Comparación de kernels en el frame más caliente |
| `docs/thermal/<nombre>_hist.png` | Histograma de °C con los umbrales |
| `docs/thermal/<nombre>_frames.png` | °C vs. estados en 5 momentos |
| `docs/thermal/<nombre>_states.png` | Fracción de cada estado en el tiempo |

`Dataset/` y `CA/FLAME3/` están en `.gitignore`; las figuras de `preprocessing/docs/thermal/` sí se versionan.

---

## 4. Proceso con JPG en pseudocolor (`thermal_to_evolution.py`, primera versión)

```bash
../GUI/venv/bin/python preprocessing/thermal_to_evolution.py Dataset/plot_1/raw_thermal_jpg CA/FLAME3/flame3_plot1_100
../GUI/venv/bin/python preprocessing/thermal_to_evolution.py Dataset/plot_2/raw_thermal_jpg CA/FLAME3/flame3_plot2_100
```

1. **Escalado:** cada JPG de 640×512 se lleva a 100×100 estirando, con `PIL` y filtro BOX
   (promedio por área).
2. **Clusters:** K-Means sobre los colores RGB de todos los píxeles de la simulación
   (200 000 muestreados), k = 2…10, codo + silueta, igual que en §2.4.
3. **Estados:** clusters ordenados por luminancia del centroide
   (0.299 R + 0.587 G + 0.114 B); la paleta va de morado (frío) a amarillo/blanco (caliente).

Opciones: `--size`, `--k`, `--k-min`, `--k-max`, `--sample`, `--sil-size`, `--seed`,
`--fig-dir`. Salidas: `.npy`, `.mat`, `.json` y figuras `_k`, `_palette`, `_frames`, `_states`.

### Resultados

| Escenario | Salida | Shape | k (codo) | Estados (centroide, fracción) |
|---|---|---|---|---|
| plot 1 | `CA/FLAME3/flame3_plot1_100` | (297, 100, 100) | 5 | `#2a1170` 4.3 %, `#551094` 62.8 %, `#7c139b` 21.9 %, `#a93778` 8.2 %, `#d38e72` 2.8 % |
| plot 2 | `CA/FLAME3/flame3_plot2_100` | (522, 100, 100) | 5 | `#581095` 63.4 %, `#7a139b` 15.1 %, `#aa2b83` 9.7 %, `#cd4a5d` 9.6 %, `#da9e6f` 2.2 % |

![estados plot 1 jpg](docs/thermal/flame3_plot1_100_states.png)
![estados plot 2 jpg](docs/thermal/flame3_plot2_100_states.png)

### Limitación: la paleta de los JPG no es absoluta

La cámara aplica control automático de ganancia (AGC): estira la paleta entre la
temperatura mínima y máxima *de cada frame*, así que un mismo color no corresponde a la
misma temperatura en frames distintos. Efectos observados:

- **plot 2, t = 0…60:** sin fuego en el encuadre, el suelo aparece naranja (estados 2–3);
  en t = 61 entra el frente y el mismo suelo pasa a morado (estado 0).
- **plot 2, t = 338…344:** frames casi uniformes en naranja seguidos del hueco de 460 s.
- **plot 2, t ≈ 463** y **plot 1, t = 0…26:** saltos de un frame por reajustes de paleta.
- Los JPG no están georreferenciados: el encuadre se desplaza con el dron.

El proceso con GeoTIFF en °C (§2) elimina los tres problemas: temperaturas absolutas,
malla común y recorte a la región observada en todos los frames.
