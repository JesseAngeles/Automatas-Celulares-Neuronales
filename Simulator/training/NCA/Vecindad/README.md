# B3/S23 con vecindad ampliada

Vecindades cuadradas `3x3/`, `5x5/`, `7x7/` y `8x8/` (64 celdas, desplazamientos -4..3: el máximo que admite
el kernel, `c_neighbourhood[128]` en `core/nca.cu`). La pérdida mínima alcanzable es 0 en todas.
Mismo contenido por carpeta que `NCA/B3S23_NR/` (pesos, curvas, `_evo`, `test_*`, `resultados.csv`).
Tabla: `tabla.tex`. Se regenera con `scripts/barrido_vecindad.py`.

## 7 × 7 sin un vecino (solo red 16:16)

`7x7_sin_NO/` (sin la esquina noroeste `(-1,-1)`) y `7x7_sin_centro/` (sin `(0,0)`): 48 celdas, las 4 configuraciones
de activación con la red 16:16, mismos hiperparámetros que el resto del barrido. Mismo contenido por carpeta
(pesos `.npz`, curvas `.csv`, `_evo`, `test_*`, `resultados.csv`). Tabla:
`Documentation/Experimentacion/datos/vecindad_7x7_reducida.tex` (`scripts/tabla_vecindad_reducida.py`).
La pérdida mínima que calcula `barrido_vecindad.py` para estas vecindades (≈ 0.002) no es confiable: con 48 celdas
casi cada patrón aparece una sola vez en los datos.
