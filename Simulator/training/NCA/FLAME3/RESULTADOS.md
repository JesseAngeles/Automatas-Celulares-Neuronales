# Resultados de FLAME 3 (al 2026-10-15)

Resumen de todos los experimentos de autómatas celulares neuronales con FLAME 3, para consulta al redactar el reporte.
Los datos completos están en el CSV de cada carpeta.

## Carpetas

| Carpeta | Contenido |
|---|---|
| `burned_states/` | Configuración final (estados de quemado): scripts, resultados y curvas de pérdida |
| `historial/` | Experimentos previos 0–11 (tabla de abajo), cada uno con su script, resultados y curvas de pérdida |
| `Personal/` | Red entrenada en la GUI (`test_01.npz`), de donde sale la vecindad dispersa de 41 células |

Las evoluciones de entrada están en `CA/FLAME3/` y los datos crudos en `Dataset/FLAME3/`, separados por selección de
frames: `seleccion1` (experimentos 0–10) y `seleccion2` (experimento 11 y `burned_states`). Los pesos (`.npz`) y los datos
no se versionan.

## Cómo se evalúa

- **Validación cruzada**: se entrena con un conjunto de imágenes y se prueba en otro que la red no vio.
- **Copia** (referencia): predecir que el siguiente paso es igual al actual. El fuego cambia pocas células por paso, así
  que la copia ya tiene exactitud alta; una red solo sirve si mejora a la copia en el conjunto de prueba.
- **Exactitud**: proporción de células con el estado correcto.
- **MAE (°C)**: a cada estado se le asigna el centro geométrico de su intervalo, √(a·b), y se compara con la temperatura
  real de la TIFF.
- **Un paso**: se predice el siguiente paso desde cada imagen real.
- **Simulación libre**: desde la primera imagen, la red evoluciona con sus propias predicciones hasta el final; se compara
  el último paso.
- **IoU quemado**: intersección sobre unión de las células quemadas al final de la simulación libre (la copia da 0).
- **IoU fuego**: de las células que alguna vez estuvieron a ≥ 130 °C durante la simulación libre.
- **Relativo**: MAE de la red ÷ MAE de la copia (< 1 es mejor que la copia).

Todos los entrenamientos (salvo que se indique): capa de salida lineal, 200 épocas, tasa de aprendizaje 0.001 con
reducción exponencial, frontera abierta, `scripts/main.py`.

## Experimentos previos (0–11, en `historial/`)

Los experimentos 0–10 usan una primera selección (`seleccion1`): p1 = plot 1 IRX 533–824, p2 = plot 2 IRX 826–1147,
p3 = plot 2 IRX 1171–1346, 15 estados logarítmicos 22.3–604.6 °C, con ignición (sin ella desde el 4).
El 11 usa la selección actual (`seleccion2`: flame_plot_1 IRX 529–654, 126 imágenes; flame_plot_2 IRX 662–824, 163;
flame_plot_3 IRX 1189–1346, 158) con 12 estados.

| # | Carpeta | Configuración | Resultado |
|---|---|---|---|
| 0 | `flame_plot_1/` (`scripts/barrido_flame.py`) | Entrena solo con p1 y evalúa en p1, p2 y p3; radio 1–4, cruda y one-hot, tasa de aprendizaje, arquitectura, frontera, lote, 400 épocas (21 redes) | Las 21 mejoran a la copia en p1 (con el que se entrenan) y ninguna en p2 ni p3 → validación cruzada desde el 1 |
| 1 | `cruzado/` | Paso de 5 s, radio 1–3, cruda y one-hot, 2:32, 3:128, 3:512, 300 épocas; entrena con 2 plots, prueba en el 3.º | Ninguna supera la exactitud de la copia; las grandes sobreajustan (3:512 one-hot: pérdida 0.04 entrenando, 2.8 en prueba) |
| 2 | `cruzado_pasos/` | Paso de 3 y 6 imágenes (15 y 30 s) | La mejor: 9×9 cruda 2:32 paso 6, exactitud 0.588 vs 0.552 de la copia en p3 |
| 3 | `k6/` | La anterior con 5 semillas y 5–30 estados (log y k-means) | Semillas estables; MAE p3 19.53 vs 21.30 °C; p1 8.40 vs 8.84; p2 empata. El número de estados no cambia el MAE relativo |
| 4 | `sin_ignicion/` | Sin imágenes de ignición (790 → 467) | Casi nada supera a la copia ni enciende; las redes de k6 (con ignición) son mejores evaluadas sin ignición |
| 5 | `sin_p2/` | Sin p2, p1 ↔ p3, 15/20/30/50 estados | Mejor log15 9×9 paso 6: MAE relativo 0.940 / 0.965; más estados bajan la cuantización (5.2 → 1.5 °C) pero no el error de dinámica |
| 6 | `rotaciones/` | Entrenamiento rotado 0/90/180/270° | MAE de un paso 0.913 / 0.923 de la copia; la simulación libre no mejora de forma consistente |
| 7 | `cuatro_estados/` | 4 estados, vecindad 9×9 a 25×25 | Agrandar la vecindad empeora; MAE 29.6 vs 20.8 °C con 15 estados en p1 → p3, misma configuración y semilla (domina la cuantización: 19.0 vs 5.2 °C) |
| 8 | `bandera/` | 13 estados + bandera "alguna vez ≥ 135.9 °C" (26 estados) | No ayuda: 5× más encendidos con precisión ≈ 0.2 en p1; simulación libre diverge en p1 |
| 9 | `opciones/` (README.md) | Ensamble, salida estocástica, terreno, pesos por clase, varios pasos, 200×200, pasos 9/12 | Ensamble y 200×200 mejoran el MAE (0.876–0.922 de la copia) pero encienden menos; estocástico/pesos/varios pasos propagan más (IoU quemado hasta 0.40) con peor MAE |
| 10 | `recomendaciones/` | Regla sin vecinos (solo enfriar), sesgo de encendido, combustible de la nube de puntos | La regla sin vecinos gana en MAE a todas las redes (0.898 / 0.817): el MAE premia enfriar. El sesgo sube el IoU de nuevas quemas 0.17 → 0.39 pero el MAE libre 22 → 37 °C |
| 11 | `flame3_12estados/` | 12 estados, pasos 6/3/1 × 2:32, 3:64, 4:256 × 9×9, 13×13, dispersa 41; 5 semillas (270 redes) | Mejor paso 6, 9×9, 2:32: MAE 0.928, simulación libre 0.776 de la copia, pero casi no propaga el fuego |

**Conclusión de 1–11**: la red aprende a enfriar las células calientes pero no a propagar el fuego. Una célula fría que
aún no se quema y una que ya se quemó tienen la misma temperatura, así que la red no las distingue → estados de quemado.

## Estados de quemado (`CA/FLAME3/seleccion2/BurnedStates/`, `notebooks/FLAME3.ipynb`, `estados_quemado`)

Desde `estados12` (estados 0–3 de log15 unidos: 12 estados, 25.3–604.6 °C), imagen por imagen y para cada célula:

- si su estado de temperatura baja respecto a la imagen anterior → queda quemada;
- si su temperatura sube más de 30 °C sobre la mínima desde que se quemó → deja de estar quemada (el umbral baja los
  regresos de quemado a sin quemar de 28,537 a 9,102);
- si no, conserva lo que tenía.

Sin quemar: estado t (0–11). Quemada: 12 + ⌊t/2⌋ → 18 estados. Paleta `GUI/colors/flame_log12_quemado6.csv`
(0–11 como flame_log12; 12–17 de negro a blanco).

| Estado | Rango | Estado | Rango |
|---|---|---|---|
| 12 | 25.3–72.9 °C (quemado, t 0–1) | 15 | 170.0–259.5 °C (quemado, t 6–7) |
| 13 | 72.9–111.4 °C (quemado, t 2–3) | 16 | 259.5–396.1 °C (quemado, t 8–9) |
| 14 | 111.4–170.0 °C (quemado, t 4–5) | 17 | 396.1–604.6 °C (quemado, t 10–11) |

Conjuntos (flame_plot_1 descartado: casi sin fuego):

- **p2** = `BurnedStates/flame_plot_1`: IRX 662–777, 116 imágenes (9 min 37 s) → 19 pasos de 30 s en simulación libre.
- **p3** = `BurnedStates/flame_plot_2`: IRX 1189–1254, 66 imágenes (5 min 25 s) → 10 pasos.

Imágenes cada 5 s; paso de 6 imágenes (30 s); las 6 secuencias con desfase 0–5 forman el entrenamiento.

## Configuración elegida (`burned_states/entrenar.py`)

Vecindad 9×9, 2:32:ReLU & 18:Linear, one-hot de 18 estados (81 × 18 = 1,458 entradas), frontera abierta, paso de
6 imágenes, 200 épocas, lote 8,192, tasa 0.001 exponencial hasta 0.00001, semillas 42, 1, 2, 3, 4.
Redes en `burned_states/redes/<ronda>/k6_9x9_2x32_onehot_s<semilla>.npz`; GIF en `burned_states/gif/`.

Promedio ± desviación de las 5 semillas (`resumen.csv`):

| Prueba | | Exactitud | MAE un paso (°C) | MAE simulación libre (°C) | IoU quemado | IoU fuego |
|---|---|---|---|---|---|---|
| p3 → p2 | Red | 0.707 ± 0.004 | 22.90 ± 0.36 | 36.49 ± 6.20 | 0.852 ± 0.005 | 0.623 ± 0.045 |
| | Copia | 0.769 | 20.95 | 20.42 | 0.000 | 0.024 |
| p2 → p3 | Red | 0.547 ± 0.003 | 46.33 ± 0.40 | 74.94 ± 1.09 | 0.255 ± 0.034 | 0.203 ± 0.038 |
| | Copia | 0.476 | 51.20 | 76.47 | 0.000 | 0.036 |

Por semilla (`resultados.csv`), MAE simulación libre / IoU quemado / IoU fuego:

| Semilla | p3 → p2 | p2 → p3 |
|---|---|---|
| 42 | 30.04 / 0.854 / 0.554 | 74.53 / 0.299 / 0.267 |
| 1 | 30.50 / 0.855 / 0.662 | 73.60 / 0.279 / 0.222 |
| 2 | 34.38 / 0.855 / 0.653 | 74.20 / 0.261 / 0.189 |
| 3 | 42.39 / 0.854 / 0.661 | 76.60 / 0.228 / 0.183 |
| 4 | 45.17 / 0.842 / 0.583 | 75.77 / 0.206 / 0.156 |

Observaciones:

- A diferencia de 1–11, el autómata celular neuronal **sí propaga el fuego** en la simulación libre.
- p3 → p2: el área quemada final coincide con IoU 0.852, pero el frente avanza más rápido y con otra forma; el MAE
  libre es peor que la copia y varía entre semillas (30–45 °C). En la esquina inferior derecha aparecen franjas
  horizontales por la frontera abierta.
- p2 → p3: propaga más lento que el real (IoU quemado 0.255), pero mejora a la copia en exactitud, MAE de un paso y MAE
  libre.
- Nuevas quemas en un paso: recall 0.72 / 0.76, precisión 0.50 / 0.61 (p3 → p2 / p2 → p3).
- Para la presentación se usaron las semillas 42 y 4 de p3 → p2.

## Barrido de 27 combinaciones (`burned_states/barrido.py`, `barrido.csv`)

Mismo entrenamiento, semilla 42. Dispersa = cuadrado 5×5 + 16 células a distancia 5 (esquinas y 3 centros de cada lado
del 11×11), 41 en total.

| Paso | Vecindad | Capa oculta | p3→p2 MAE 1 paso | p3→p2 MAE libre | p3→p2 IoU q | p2→p3 MAE 1 paso | p2→p3 MAE libre | p2→p3 IoU q |
|---|---|---|---|---|---|---|---|---|
| 30 s | Copia | | 20.95 | 20.42 | 0.000 | 51.20 | 76.47 | 0.000 |
| 30 s | 9×9 | 2:32 | 23.05 | 30.04 | 0.854 | 45.73 | 74.53 | 0.299 |
| 30 s | 9×9 | 3:64 | 26.09 | 34.72 | 0.839 | 47.50 | 78.05 | 0.363 |
| 30 s | 9×9 | 4:256 | 31.09 | 96.15 | 0.732 | 47.46 | 73.53 | 0.487 |
| 30 s | 13×13 | 2:32 | 29.27 | 30.83 | 0.852 | 47.71 | 76.60 | 0.321 |
| 30 s | 13×13 | 3:64 | 27.16 | 21.07 | 0.842 | 48.50 | 79.27 | 0.504 |
| 30 s | 13×13 | 4:256 | 30.23 | 73.95 | 0.724 | 47.02 | 65.95 | 0.557 |
| 30 s | Dispersa | 2:32 | 23.84 | 47.43 | 0.840 | 45.25 | 74.44 | 0.247 |
| 30 s | Dispersa | 3:64 | 25.03 | 28.54 | 0.855 | 46.22 | 77.04 | 0.325 |
| 30 s | Dispersa | 4:256 | 30.48 | 77.97 | 0.739 | 46.99 | 71.67 | 0.385 |
| 15 s | Copia | | 14.01 | 20.42 | 0.000 | 32.02 | 66.83 | 0.000 |
| 15 s | 9×9 | 2:32 | 15.15 | 91.84 | 0.849 | 32.24 | 65.77 | 0.229 |
| 15 s | 9×9 | 3:64 | 16.51 | 79.83 | 0.845 | 33.15 | 66.27 | 0.430 |
| 15 s | 9×9 | 4:256 | 20.29 | 66.52 | 0.786 | 33.47 | 59.64 | 0.671 |
| 15 s | 13×13 | 2:32 | 16.84 | 61.35 | 0.852 | 33.81 | 80.35 | 0.382 |
| 15 s | 13×13 | 3:64 | 18.67 | 40.78 | 0.854 | 35.07 | 70.56 | 0.607 |
| 15 s | 13×13 | 4:256 | 20.29 | 69.15 | 0.827 | 34.46 | 52.04 | 0.687 |
| 15 s | Dispersa | 2:32 | 14.83 | 74.24 | 0.840 | 32.36 | 71.18 | 0.225 |
| 15 s | Dispersa | 3:64 | 15.78 | 82.44 | 0.856 | 32.12 | 74.63 | 0.405 |
| 15 s | Dispersa | 4:256 | 19.59 | 58.65 | 0.782 | 33.42 | 62.63 | 0.496 |
| 5 s | Copia | | 9.98 | 19.66 | 0.000 | 20.63 | 62.90 | 0.000 |
| 5 s | 9×9 | 2:32 | 10.15 | 236.45 | 0.742 | 22.24 | 94.00 | 0.161 |
| 5 s | 9×9 | 3:64 | 10.65 | 178.71 | 0.730 | 22.63 | 79.33 | 0.277 |
| 5 s | 9×9 | 4:256 | 12.56 | 84.31 | 0.790 | 24.39 | 54.92 | 0.566 |
| 5 s | 13×13 | 2:32 | 10.57 | 196.55 | 0.759 | 24.02 | 82.66 | 0.304 |
| 5 s | 13×13 | 3:64 | 11.14 | 156.15 | 0.752 | 24.10 | 78.54 | 0.396 |
| 5 s | 13×13 | 4:256 | 12.87 | 58.31 | 0.803 | 25.24 | 56.78 | 0.526 |
| 5 s | Dispersa | 2:32 | 10.05 | 235.00 | 0.723 | 21.45 | 109.01 | 0.100 |
| 5 s | Dispersa | 3:64 | 10.20 | 231.72 | 0.737 | 21.86 | 81.06 | 0.211 |
| 5 s | Dispersa | 4:256 | 11.85 | 139.38 | 0.827 | 24.22 | 48.71 | 0.606 |

Observaciones:

- Más equilibrada: 13×13, 3:64, paso de 30 s (MAE libre 21.07 °C ≈ copia 20.42 con IoU quemado 0.842; p2 → p3 IoU
  0.504), pero sus 169 vecinos superan el máximo de 128 de la GUI.
- 4:256 propaga más en p2 → p3 (IoU hasta 0.687), pero en p3 → p2 el fuego no se apaga (MAE libre hasta 96 °C).
- Paso de 5 s: todas fallan en la simulación libre; las células se quedan quemadas en gris sin enfriarse (hasta 236 °C).
- Equilibradas que caben en la GUI: 9×9 2:32 y dispersa 3:64, ambas con paso de 30 s.

## Pendientes ofrecidos (no pedidos)

- 5 semillas para las 3 finalistas (13×13 3:64, 9×9 4:256 paso 15 s, dispersa 3:64).
- La misma configuración sin estados de quemado en los mismos frames, para aislar su efecto.
