# B3/S23 con vecindad ampliada

Vecindades cuadradas `3x3/`, `5x5/`, `7x7/` y `8x8/` (64 celdas, desplazamientos -4..3: el máximo que admite
el kernel, `c_neighbourhood[128]` en `core/nca.cu`). La pérdida mínima alcanzable es 0 en todas.
Mismo contenido por carpeta que `NCA/B3S23_NR/` (pesos, curvas, `_evo`, `test_*`, `resultados.csv`).
Tabla: `tabla.tex`. Se regenera con `scripts/barrido_vecindad.py`.
