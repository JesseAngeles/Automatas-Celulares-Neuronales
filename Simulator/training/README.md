# Entrenamiento y experimentos

Entrenamiento del autómata celular neuronal (NCA) y datos de los tres experimentos del
capítulo *Experimentación y resultados* (`Documentation/Experimentacion/`).

```
training/
├── scripts/        main.py (entrenamiento), genetic_search.py, test.py
├── preprocessing/  datos externos → evoluciones (T, n, n) int32 en .npy/.mat
├── CA/             evoluciones de referencia (entrada del entrenamiento)
├── NCA/            pesos entrenados (.npz), evoluciones que generan y notebooks de análisis
├── notebooks/      notebooks generales de Mathematica
└── Dataset/        datos crudos descargados (no versionado)
```

Todos los comandos se corren desde `Simulator/training` con el venv de la GUI:

```bash
cd Simulator/training
PY=../GUI/venv/bin/python
```

## Estado

**Fecha límite del reporte de avances: viernes 2026-10-09.**

| # | Experimento | Datos (`CA/`) | Modelos (`NCA/`) | Documento | Estado |
|---|---|---|---|---|---|
| 1.1 | Juego de la vida (B3/S23 y variantes) | `B3S23/`, `B3S236/`, `B3S238/`, `B38S23/`, `B38S238/` | mismas carpetas | `duplicacion.tex` | Hecho |
| 1.2 | Tamaño de vecindad | `B3S23/` | `Vecindad/` | `exp1_vecindad.tex` | TODO |
| 1.3 | Sandpile | `Sandpile/` | `Sandpile/` | `exp1_sandpile_langton.tex` | TODO |
| 1.3 | Hormiga de Langton | `LangtonAnt/` | `LangtonAnt/` | `exp1_sandpile_langton.tex` | TODO |
| 2.1 | Incendio (FLAME 3) | `FLAME3/` | `FLAME3/` | `exp2_flame3.tex` | Limpieza hecha; falta entrenar |
| 2.2 | Desarrollo embrionario | `embryo/` → `embryo_states/` | `embryo/` | `exp2_embriones.tex` | Limpieza hecha (3 embriones); falta lote y entrenar |
| 2.3 | Cell Tracking Challenge | `CTC/` (crudos en `Dataset/CTC/`) | `CTC/` | `exp2_ctc.tex` | TODO: descargar y limpiar |
| 3 | Morfogénesis | — | `Morfogenesis/` | `exp3_morfogenesis.tex` | TODO |

Los pendientes también aparecen como recuadros amarillos `\TODO{…}` en el PDF
(`Documentation/build/main.pdf`, sección 10).

## Pendientes

### 1.2 Tamaño de vecindad

`scripts/main.py` ya acepta la vecindad por argumento:

- `--radius r`: cuadrado de (2r+1)×(2r+1) celdas (default 1 = Moore, 9 celdas).
- `--neighbourhood "dx,dy dx,dy …"`: lista explícita de desplazamientos.

```bash
# Moore r=1 (referencia de la sección 10.1.1)
$PY scripts/main.py --data CA/B3S23/evo.npy --radius 1 --output NCA/Vecindad/moore_r1.npz --log-csv NCA/Vecindad/moore_r1.csv
# von Neumann (5 celdas)
$PY scripts/main.py --data CA/B3S23/evo.npy --neighbourhood "0,-1 -1,0 0,0 1,0 0,1" --output NCA/Vecindad/vonneumann.npz --log-csv NCA/Vecindad/vonneumann.csv
# Radio 2 (25 celdas)
$PY scripts/main.py --data CA/B3S23/evo.npy --radius 2 --output NCA/Vecindad/moore_r2.npz --log-csv NCA/Vecindad/moore_r2.csv
```

Usar la misma arquitectura, `--loss-target` y `--seed` que en la sección 10.1.1 para que
la comparación sea justa.

### 1.3 Sandpile y hormiga de Langton

- [ ] Definir los estados (sandpile: reloj de estados para la caída y propagación de granos;
      Langton: estados de orientación de la hormiga sobre cada color de celda).
- [ ] Generar las evoluciones en `CA/Sandpile/` y `CA/LangtonAnt/` con el formato
      `(T, H, W) int32` (`.npy` y `.mat` con la variable `evolution`).
- [ ] Entrenar (salida multiclase: `--out-size` = número de estados) y comparar contra la regla.

### 2.1 FLAME 3

- [ ] Entrenar con `CA/FLAME3/flame3_plot1_celsius_100.npy` (4 estados):
      `$PY scripts/main.py --data CA/FLAME3/flame3_plot1_celsius_100.npy --out-size 4 --out-act linear --output NCA/FLAME3/plot1.npz --log-csv NCA/FLAME3/plot1.csv`
- [ ] (Opcional) descargar los GeoTIFF en °C de plot 2.

Detalle del proceso: `preprocessing/README.md`.

### 2.2 Embriones

- [ ] Procesar el lote completo (sin figuras):
      `$PY preprocessing/embryo_to_evolution.py CA/embryo/* --no-figures`
- [ ] Decidir qué embriones excluir (pegados a la pared del pozo, pocos frames).
- [ ] Decidir si se borran las copias duplicadas de `CA/embryo/*/F0/` (~3 GB).
- [ ] Entrenar (5 estados).

Detalle del proceso: `preprocessing/README_embryo.md`.

### 2.3 Cell Tracking Challenge

- [ ] Descargar los 8 datasets 2D ligeros (~635 MB). El servidor
      `data.celltrackingchallenge.net` rechazaba conexiones el 2026-10-07; reintentar:
      `$PY preprocessing/download_ctc.py`
- [ ] Revisar qué secuencias tienen segmentación completa (`<seq>_ST/SEG` en todos los
      frames) y elegir con cuáles trabajar.
- [ ] Definir los estados a partir de las máscaras (p. ej. fondo / interior / borde de célula)
      y escribir `preprocessing/ctc_to_evolution.py`.
- [ ] Entrenar.

### 3 Morfogénesis

- [ ] Definir condición inicial y configuración final.
- [ ] Agregar a `scripts/main.py` un modo cuya pérdida se calcule sobre el estado después de
      N pasos (hoy solo entrena con pares de pasos consecutivos).
- [ ] Entrenar y analizar la evolución intermedia.
