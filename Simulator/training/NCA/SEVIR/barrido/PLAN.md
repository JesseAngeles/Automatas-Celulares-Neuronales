# Barrido SEVIR (2026-10-29)

Script: `barrido.py` (corre todo en secuencia, se puede reanudar). Entrenamiento con `scripts/main.py`;
vecindades y evaluación en `comun.py`.

## Datos

| Conjunto | Tormentas | Origen |
|---|---|---|
| Disponibles para entrenar | 1,996 (1,286 de 2018 + 710 de 2017) | `CA/SEVIR/todo_2018`, `CA/SEVIR/todo_2017` |
| Prueba | 200 de 2019 | `CA/SEVIR/prueba_200` |

Mismo preprocesamiento que el primer experimento (sección 2 de `notebooks/SEVIR.ipynb`): 96×96 (4 km), 7 estados con los
umbrales de SEVIR, tormentas con al menos 2 % de células en estado ≥ 2 en la primera imagen. Las primeras 40 de
`todo_2018` son las 40 del primer experimento y las primeras 20 de `prueba_200` son sus 20 de prueba.

No se usan todas: `main.py` arma todas las vecindades en RAM y en la GPU (float32), así que el límite es ~9 GB de
muestras. 40 tormentas completas con 11×11 ya son 8.7 GB (18 GB de RAM como máximo). Para usar más tormentas se toman
solo algunas transiciones (imagen t → t + 1) de cada una; el archivo `(N, 2, 96, 96)` lo lee `main.py` como N
evoluciones de 2 imágenes. Las transiciones de cada tormenta salen de una permutación fija (semilla 0).

## Fijo en todas las corridas

- `main.py --open-boundary` (fuera de la imagen todo es estado 0), también en la simulación.
- Salida lineal de 7 estados (4 en la prueba de segmentación), CrossEntropy, Adam, lote 8192, 30 épocas.
- `--lr 1e-3 --lr-schedule exp --lr-min 1e-5`, semilla 42.
- Evaluación en las 200 tormentas de prueba contra la predicción sin cambio: un paso (exactitud, exactitud donde llueve,
  coincidencia del área con tormenta) y simulación libre desde la imagen 0 (coincidencia del área a 30, 60, 120 y
  240 min y proporción entre el área con tormenta simulada y la real).

## Fases

**Fase 1. Pruebas pequeñas (80 corridas).** 40 tormentas completas (1,920 transiciones).

- Vecindades (todas ≤ 128 células, caben en la GUI):
  - cuadradas: 5×5 (25), 9×9 (81), 11×11 (121)
  - circulares: radio 3 (29), radio 5 (81), radio 6 (113)
  - dispersas: la de FLAME 3 (41, alcance 5) y la propuesta `estrella57` (57, alcance 11): núcleo 5×5 y 8 rayos
    cada 45° con puntos a 4, 6, 8 y 11 células (16–44 km). El núcleo ve el movimiento de un paso (~1 célula cada
    5 min) y los rayos la tormenta que llega desde lejos, sin pagar las 529 células de un cuadrado de 23×23.
- Redes (capas:neuronas, ReLU): 2:32, 3:64, 8:64, 16:64, 16:128.
- Entrada: cruda (el estado como número) y one-hot.

Orden: promedio de los lugares en (1) área con tormenta a un paso, (2) y (3) área con tormenta en simulación libre a
60 y 120 min y (4) cercanía de la cantidad de tormenta simulada a la real a 120 min. Pasan las 4 primeras.

**Fase 2. Más tormentas (hasta 12 corridas).** Cada finalista con:

| Nivel | Tormentas | Transiciones por tormenta | Para ver |
|---|---|---|---|
| A | 40 | 48 (completas) | la fase 1 |
| B | 400 | 5 (≈ las mismas 1,920–2,000 transiciones que A) | más variedad con la misma cantidad de datos |
| C | 400 | las que quepan en ~9 GB (24 con 5×5, 5 con 11×11) | más datos |
| D | 1,000 | las que quepan (9 con 5×5, 2 con 11×11) | aún más variedad |

**Fase 3. La mejor de la fase 2 (4 corridas).** Semillas 1 y 2 (estabilidad), 4 estados (sin lluvia, lluvia,
tormenta 74–160, tormenta intensa ≥ 160; misma definición de tormenta, así que el área es comparable, la exactitud no)
y 100 épocas.

## Nota

`main.py` entrena con todas las células, incluidas las del borde. En una prueba previa con un entrenador propio
(descartado para usar `main.py`), quitar del entrenamiento las células cuya vecindad sale de la imagen dio peor
simulación libre (proporción del área a 120 min 0.49 contra 0.73), así que no se pierde nada.

## Salidas

`resultados.csv` (todas las corridas), `ranking_fase1.csv`, `mejor.json`, `area_libre.csv` y `area_real.csv` (área con
tormenta en cada paso de la simulación libre), `redes/fase*/` (pesos `.npz` y curvas de pérdida), `libre/` (simulación
libre de las fases 2 y 3), `registro.log`. Transiciones elegidas en `CA/SEVIR/barrido/`.
