# B3/S23 con vecindad reducida

Carpeta por máscara: primer dígito = centro, luego los 8 vecinos del borde en orden de lectura
(NO, N, NE, O, E, SO, S, SE); 0 = vecino eliminado.

| Carpeta | Vecino eliminado | Pérdida mínima alcanzable |
|---|---|---|
| `1_01111111/` | Noroeste (-1,-1) | 0.121294 |
| `1_10111111/` | Norte (0,-1) | 0.118412 |
| `0_11111111_barrido/` | Centro (0,0) | 0.078650 |
| `0_11111111/` | Pruebas manuales previas (GUI); su vecindad usa (2,-3) en lugar del centro | — |

En cada carpeta del barrido: `<capas>-<neuronas>-<act>_<salida>-<act>.npz` (pesos), `.csv` (pérdida por época),
`_evo.npy|mat` (configuración de 512 patrones, 1 paso) y `<modelo>/test_{0..9}.npy|mat` (50 pasos desde las
condiciones aleatorias de `CA/B3S23/aleatorias/`), más `resultados.csv`.
Tablas: `tabla_borde.tex`, `tabla_centro.tex`. Hiperparámetros: `ajuste_hiperparametros/`.
Se regenera con `scripts/barrido_vecindad.py` (ver su docstring).
