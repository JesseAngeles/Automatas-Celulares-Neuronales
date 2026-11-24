# Últimos entrenamientos de Physarum: resultados (2026-11-16)

Ejecución de `../PLAN_ULTIMOS_ENTRENAMIENTOS.md`. Las medidas son las de `tabla.py`:
- ***C***: coincidencia del plasmodio, IoU × área menor / área mayor, promedio desde el primer paso.
- **Área final**: fracción de la caja con plasmodio al final.
- **Frontera**: fracción del plasmodio que toca algo que no es plasmodio, en la segunda mitad.
- **Brillo**: fracción del plasmodio que cambia de nivel entre pasos, en la segunda mitad.

Todo se mide en las 4 cajas de prueba (dish5) con simulación libre y argmax desde la primera imagen real, salvo que
se indique otra cosa. Los valores reales a 6 h son: área final 0.337 (0.344 a 30 min), área 001 0.307, frontera 0.427
y brillo 0.236.

## Resumen

- **Experimento 0.** La hipótesis del reporte se confirma: a 5–6 h, el plasmodio que llegó en el último paso se
  retira 4.5–4.7 veces más que el que lleva ≥ 4 pasos (32–35% contra 7%).
- **Experimentos A y B.** Con argmax, ni la memoria de edad ni el peso a la retirada mejoran la red con sesgo. La
  mejor sigue siendo `e4_5h_9x9_da` (0.240).
- **Experimento C.** La edad **sí mejora la red con ruido**. La red final es `c_e6edad_s43`:
  - *C* 0.155 ± 0.001 con 3 semillas, contra 0.105 de la red con ruido del reporte.
  - El área final es casi la real (0.349 contra 0.344, antes 0.157).
  - Frontera 0.380 (real 0.429) y brillo 0.340 (real 0.236).
  - Forma una red conectada entre las avenas en lugar de grupos sueltos.
  - Sin edad, la misma receta da 0.137 ± 0.007, así que la edad aporta +0.018 (más de dos desviaciones estándar).
- **Verificación y generalización.** La red final corre igual en el núcleo de la GUI que en PyTorch (0 células
  distintas por paso). Con un experimento completo excluido, supera a congelarse y a llenar la caja en los 4 casos.

**Integración (2026-11-16):**
- **Reporte:** subsección 11.3.6 «Memoria de la edad del plasmodio» (tabla 11.17), además de las conclusiones, el
  trabajo a futuro, el resumen y el abstract. Sigue en 275 páginas.
- **Artículo:** fila «Ruido y edad» en la tabla 10, el párrafo de resultados, la síntesis, la conclusión P4, el trabajo
  a futuro y el resumen. Sigue en 14 páginas.
- **Presentación:** la diapositiva 25 (video de 008-dish5 de `video_suave.py`, tabla y guion), la hipótesis, el
  trabajo a futuro y los anexos. Sigue en 57 diapositivas.

## Experimento 0: retirada según la edad (`edad.py` → `edad.json`)

Probabilidad de que una célula con plasmodio no lo tenga en el siguiente paso, en las 20 cajas de entrenamiento,
con todos los desfases. Las edades están en pasos.

| Paso | Edad 1 | Edad 2–3 | Edad ≥ 4 | Nuevo / establecido |
|---|---|---|---|---|
| 30 min | 0.0% | 9.5% | 2.3% | — |
| 2 h | 21.7% | 16.2% | 5.6% | 3.9× |
| 5 h | 32.3% | 23.2% | 7.2% | 4.5× |
| 6 h | 34.9% | 23.7% | 7.4% | 4.7× |

A 30 min la edad 1 da 0% por el filtro de persistencia: un cambio debe durar 2 imágenes. Con edades 2–3 contra ≥ 4
la razón es 4.1×.

## Experimento A: memoria de edad (fase `edad` de `barrido.py`)

Opción `edad=True` de `barrido.py`:
- El plasmodio que no lo era un paso antes pasa a un estado "nuevo", que va después de todos los demás.
- Con 4 estados quedan 5; con 6 estados quedan 9.
- El plasmodio de la primera imagen cuenta como establecido.
- `≥ pl` sigue siendo "hay plasmodio", así que el sesgo y las medidas no cambian.

Resultados con 3 semillas (`fase_edad.json`, con `medir.py`):

| Red | Calibración | *C* | Área final | Área 001 | Frontera |
|---|---|---|---|---|---|
| `e4_5h_9x9_da` (base) | 0.233 | **0.240 ± 0.000** | 0.420 | 0.585 | 0.169 |
| `e4_5h_9x9_da_edad` | 0.225 | 0.233 ± 0.007 | 0.466 | 0.663 | 0.137 |
| `e4_5h_9x9_da_edad_vector` | 0.242 | 0.225 ± 0.009 | 0.413 | 0.614 | 0.228 |
| `e4_6h_9x9_da` (base) | 0.241 | 0.228 ± 0.004 | 0.444 | 0.622 | 0.163 |
| `e4_6h_9x9_da_edad` | 0.233 | 0.228 ± 0.004 | 0.460 | 0.644 | 0.142 |
| `e6_6h_circ6_da` (base) | 0.225 | 0.231 ± 0.005 | 0.404 | 0.624 | 0.168 |
| `e6_6h_circ6_da_edad` | 0.210 | 0.228 ± 0.009 | 0.504 | 0.672 | 0.144 |

**No cumple el criterio.** La edad sola hace que la red crezca más y en manchas más sólidas. Con un sesgo por estado,
la calibración baja el del plasmodio establecido (1.25 → −0.25) y deja el del nuevo en 1.25. Eso mejora la frontera
(0.228), pero *C* baja 0.015. La calibración volvió a no predecir la prueba (0.242 contra 0.225).

**Explicación.** Con argmax, una retirada de 32% nunca es el estado más probable.

## Experimento B: más peso a la retirada (fase `retirada`)

Opción nueva `--retreat-weight W --retreat-from 3` de `scripts/main.py` (`retirada=W` en `barrido.py`). Una semilla:

| Red | *C* | Área final | Área 001 | Frontera |
|---|---|---|---|---|
| `e4_5h_9x9_da` | 0.239 | 0.422 | 0.579 | 0.162 |
| `_ret2` / `_ret4` / `_ret8` | 0.236 / 0.230 / 0.206 | 0.435 / 0.492 / 0.314 | 0.594 / 0.648 / 0.387 | 0.179 / 0.151 / 0.559 |
| `_edad_ret2` / `_edad_ret4` / `_edad_ret8` | 0.242 / 0.217 / 0.192 | 0.437 / 0.359 / 0.344 | 0.614 / 0.575 / 0.386 | 0.139 / 0.192 / 0.599 |

**No cumple el criterio.** Con W = 8 el área de 001 se acerca a la real, pero la frontera alta viene de rayas y de un
patrón de tablero que oscila, no de venas, y *C* baja 0.03–0.05. Con W = 2 y W = 4 casi no hay efecto.

## Experimento C: red con ruido con maestra de paso largo y edad

### Prueba previa (`ideal.py`)

Proceso ideal: se muestrea de la maestra con el azar correlacionado de `dado/comun.py`, sin entrenar alumna.
- **Sin empujón:** con cualquier maestra crece de menos (área 0.13–0.18) y tiene una frontera realista (~0.46). La
  edad sube *C* entre 0.016 y 0.023 (`ideal.json`).
- **Con empujón** (`ideal_crecer*.json`), la mejor maestra es `e6_6h_circ6_da_edad` con 0.35: *C* 0.144, área
  0.311 y frontera 0.382.

### Alumnas (`dado_largo.py`)

Configuración común:
- 18 pares reales y 18 simulados por caja.
- Vecindad circ6 (113 vecinos).
- 36 estados: 9 visibles × 4 de ruido.
- Red 16:128:ReLU, *one-hot*, 30 épocas.

Medidas cada 6 h con 3 semillas (42, 43, 44):

| Alumna | Calibración | *C* prueba | Área final | Área 001 | Frontera | Brillo |
|---|---|---|---|---|---|---|
| `c_e6edad` (6 estados + edad) | 0.150 ± 0.010 | **0.155 ± 0.001** | 0.349 | 0.424 | 0.380 | 0.340 |
| `c_e4edad` (4 estados + edad) | 0.148 ± 0.004 | 0.151 ± 0.002 | 0.367 | 0.427 | 0.353 | — |
| `c_e6` (sin edad, control) | 0.128 ± 0.002 | 0.137 ± 0.007 | 0.313 | 0.377 | 0.395 | 0.325 |
| Red con ruido del reporte (`b16x128`) | — | 0.105 | 0.157 | — | 0.498 | 0.413 |
| Red con sesgo del reporte (`e6_6h_circ6_da`) | — | 0.225 | 0.353 | — | 0.154 | 0.007 |

**Red final:** `c_e6edad_s43`, elegida por calibración (0.163).

Corrida en el núcleo de la GUI (`cierre.py` → `redes_dado/c_e6edad_s43/gui.json`):
- 0 células distintas contra PyTorch en cada paso.
- *C* 0.159, área final 0.330, frontera 0.387 y brillo 0.334.

| Caja | *C* | Área final |
|---|---|---|
| 001 | 0.171 | 0.398 |
| 002 | 0.142 | 0.399 |
| 008 | 0.169 | 0.171 |
| 009 | 0.154 | 0.349 |

`videos/finales.png` compara la última imagen de las 4 cajas: la real, la red con sesgo, la red con ruido anterior y
la nueva. La nueva forma una red conectada entre las avenas; la anterior, grupos sueltos. Las venas todavía son más
gruesas que las reales, y en 009 crece de más.

### Generalización a un experimento no visto (`fuera_dado.py` → `fuera_dado/resumen.json`)

Para cada experimento excluido se reentrenan la maestra y la alumna con las 18 cajas de los otros 3. La prueba es en
las 6 cajas del excluido.

| Excluido | *C* | Congelarse | Llenar la caja | Área final red / real | Red con sesgo (`fuera.py`): *C*, área red / real |
|---|---|---|---|---|---|
| 001 | 0.149 | 0.025 | 0.125 | 0.40 / 0.38 | 0.246, 0.69 / 0.37 |
| 002 | 0.173 | 0.033 | 0.159 | 0.39 / 0.51 | 0.289, 0.75 / 0.50 |
| 008 | 0.108 | 0.073 | 0.015 | 0.15 / 0.17 | 0.173, 0.18 / 0.17 |
| 009 | 0.151 | 0.030 | 0.029 | 0.49 / 0.18 | 0.176, 0.33 / 0.18 |

Supera a las dos referencias en los 4 casos. Comparada con la red con sesgo:
- **Coincidencia:** tiene menos.
- **Área en 001 y 002:** acierta mucho mejor.
- **Área en 009:** crece de más.

## Archivos

- `edad.py`, `edad.json`: experimento 0.
- `../barrido/barrido.py`:
  - opciones `edad` y `retirada`
  - fases `edad` y `retirada`
  - las filas quedan en `resultados_v3.csv` y las redes en `redes/`
- `../../../scripts/main.py`: opción `--retreat-weight` / `--retreat-from` (documentada en `main.py --help`).
- `medir.py`: las medidas de `tabla.py` sobre las simulaciones libres de `barrido/redes/`.
  - Salidas: `bases.json`, `fase_edad.json`, `fase_retirada.json`.
- `ideal.py`: proceso ideal con maestras de paso largo.
  - Salidas: `ideal*.json`.
- `dado_largo.py`: datos (`datos_dado/`), entrenamiento y evaluación de alumnas (`redes_dado/`).
- `cierre.py`: paquete para la GUI (`gui/c_e6edad_s43/`), verificación en el núcleo y videos (`videos/`).
- `fuera_dado.py`: generalización (`fuera_dado/`).
- `video_suave.py`: video real (cada 30 min) | NCA a 20 fps. Entre dos pasos de la red (6 h), los cuadros
  intermedios son una transición: las células que se vuelven plasmodio aparecen en orden de distancia al plasmodio.
  Se usa en la presentación (008-dish5).

### Cargarla en la GUI (`gui/c_e6edad_s43/`)

- **Load NCA weights:** `c_e6edad_s43.npz`. Tiene 113 vecinos, *one-hot* 36 y no lleva sesgo.
- **Load colors:** `physarum36.csv`. Los estados nuevos se ven en naranja.
- **Import:** `iniciales/<caja>_inicio.npy` (160 × 160), la primera imagen con ruido al azar.
- Frontera abierta.
- Un paso equivale a 6 h.
- **Para entrenarla desde cero** no se puede usar el panel Training (la vecindad llega a radio 6). Se usa
  `python NCA/Physarum/edad/dado_largo.py c_e6edad_s43 e6_6h_circ6_da_edad 0.35 18 18 43` desde
  `Simulator/training/`, que arma el comando de `main.py` en `entrenar()`. La maestra sale de
  `python NCA/Physarum/barrido/barrido.py edad`.
