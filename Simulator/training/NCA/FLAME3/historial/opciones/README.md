# Opciones para mejorar la NCA de FLAME 3 (2026-10-14)

Cada opción cambia **una sola cosa** respecto a la referencia y se evalúa igual (`comun.py`).

**Referencia (0_base):** `../sin_p2`: log15 (24.7–604.6 °C), p1 → p3 y p3 → p1 sin ignición,
9×9, paso de 6 frames (30 s), entrada cruda, 2 capas de 32, 5 semillas.

**Métricas** (escenario no visto, promedio de semillas). Entre paréntesis, la proporción contra la
copia ("el siguiente paso es igual"); menos de 1 es mejor que copiar.
- `mae`: error medio de un paso en °C.
- `rollout`: error en °C tras 10 min de simulación libre (120 frames).
- `IoU q`: coincidencia del área quemada al final de la simulación.

| Opción | Variante | p1 → p3 mae | p1 → p3 rollout | p1 → p3 IoU q | p3 → p1 mae | p3 → p1 rollout | p3 → p1 IoU q |
|---|---|---|---|---|---|---|---|
| copia | | 22.32 | 29.50 | 0.048 | 14.37 | 18.10 | 0.029 |
| 0 referencia | sin_p2 | 20.97 (0.940) | 21.91 ± 0.85 (0.743) | 0.210 | 13.87 (0.965) | 15.78 ± 3.58 (0.872) | 0.037 |
| 0 referencia | rotaciones | 20.37 (0.913) | 20.61 ± 1.33 (0.699) | 0.083 | 13.27 (0.923) | 19.26 ± 2.01 (1.064) | 0.038 |
| 1 ensamble | sin_p2 | 20.58 (0.922) | 24.16 (0.819) | 0.201 | 13.11 (0.912) | 11.96 (0.661) | 0.029 |
| 1 ensamble | rotaciones | 20.27 (0.908) | 20.12 (0.682) | 0.072 | 12.59 (0.876) | 17.02 (0.941) | 0.029 |
| 2 estocástico | muestreo | 26.00 (1.165) | 33.68 ± 6.07 (1.142) | 0.404 | 19.79 (1.377) | 28.78 ± 5.73 (1.590) | 0.205 |
| 2 estocástico | esperado | 21.55 (0.966) | 30.47 ± 5.86 (1.033) | 0.408 | 16.35 (1.137) | 47.64 ± 5.88 (2.632) | 0.262 |
| 3 terreno | mediana2 | 23.22 (1.041) | 18.90 ± 0.30 (0.641) | 0.050 | 12.39 (0.862) | 17.41 ± 4.12 (0.962) | 0.041 |
| 3 terreno | terciles3 | 25.00 (1.120) | 18.49 ± 0.18 (0.627) | 0.048 | 12.57 (0.874) | 13.98 ± 0.28 (0.772) | 0.031 |
| 5 pesos | α 0 (control) | 20.97 (0.940) | 21.91 ± 0.85 (0.743) | 0.210 | 13.87 (0.965) | 15.78 ± 3.58 (0.872) | 0.037 |
| 5 pesos | α 3 | 22.39 (1.004) | 20.10 ± 5.06 (0.681) | 0.222 | 15.68 (1.091) | 32.37 ± 13.10 (1.788) | 0.249 |
| 5 pesos | α 10 | 23.72 (1.063) | 20.11 ± 3.85 (0.682) | 0.239 | 17.76 (1.235) | 60.53 ± 15.17 (3.345) | 0.367 |
| 6 multipaso | K 1 (control) | 20.94 (0.938) | 22.23 ± 0.80 (0.754) | 0.219 | 13.85 (0.964) | 15.53 ± 3.02 (0.858) | 0.037 |
| 6 multipaso | K 2 | 21.54 (0.965) | 20.59 ± 1.18 (0.698) | 0.255 | 14.72 (1.024) | 16.57 ± 1.75 (0.915) | 0.051 |
| 6 multipaso | K 4 | 22.42 (1.005) | 20.31 ± 1.23 (0.689) | 0.349 | 13.49 (0.939) | 18.62 ± 4.93 (1.029) | 0.127 |
| 7 200×200 | 9×9 | 20.54 (0.901) | 18.78 ± 0.21 (0.632) | 0.082 | 13.15 (0.893) | 13.87 ± 2.03 (0.760) | 0.037 |
| 7 200×200 | 17×17 (3 semillas) | 21.28 (0.934) | 20.26 ± 0.96 (0.682) | 0.161 | 15.29 (1.039) | 31.23 ± 0.11 (1.711) | 0.094 |
| 8 pasos | 9 frames, 9×9 | 27.15 (0.889) | 20.41 ± 0.78 (0.667) | 0.144 | 18.90 (0.955) | 19.30 ± 2.78 (0.982) | 0.063 |
| 8 pasos | 9 frames, 13×13 | 27.78 (0.910) | 22.12 ± 2.67 (0.723) | 0.215 | 23.49 (1.187) | 33.28 ± 4.98 (1.694) | 0.181 |
| 8 pasos | 12 frames, 9×9 | 31.41 (0.839) | 19.08 ± 1.03 (0.647) | 0.135 | 23.15 (0.943) | 20.97 ± 1.41 (1.159) | 0.279 |
| 8 pasos | 12 frames, 13×13 | 32.47 (0.867) | 18.16 ± 0.92 (0.616) | 0.178 | 29.18 (1.188) | 30.51 ± 5.27 (1.686) | 0.329 |

**Notas sobre la comparación:**
- En la opción 8 el `mae` es de un paso de 45 o 60 s, así que solo es comparable su proporción contra la copia.
- En la opción 7 los °C son sobre la malla de 200×200; la copia cambia muy poco (22.80 y 14.73 °C).

La opción 4 no es un experimento: es la revisión de qué datos trae FLAME 3; está en `4_datos/hallazgos.md`.

## Archivos

- `comun.py`: evaluación común (un paso, simulación libre, IoU) y utilidades.
- `<N>_<opción>/<script>.py`: cada opción.
- `<N>_<opción>/resultados.csv`: una fila por red (o ensamble), ronda y semilla.
- `<N>_<opción>/registro.log`: qué se hizo y cuándo.
- `<N>_<opción>/redes/`: los pesos de las redes entrenadas.
- `resumen.py` → `resumen.csv`: promedio y desviación por opción, variante y ronda.
- `5_pesos/pesos.py` con α = 0 reproduce exactamente a `scripts/main.py`: mismos pesos, verificado con las 10 redes de la referencia.
- `6_multipaso` parte de las redes de `../sin_p2/semillas` y las ajusta 60 épocas.
- `7_resolucion200` usa `Dataset/FLAME3/seleccion1/flame_plot_{1,3}/geo_thermal_tiff_celsius_aligned_200`, generado con `reducir(..., size=200)` de `notebooks/FLAME3.ipynb`. El 17×17 se entrenó con el entrenador de `5_pesos` porque `main.py` se queda sin memoria en la GPU: decide subir los datos comparando con la VRAM total, y la copia barajada de cada época ya no cabe.
