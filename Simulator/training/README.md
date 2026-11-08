# Entrenamiento y experimentos

Entrenamiento del autómata celular neuronal (NCA) y datos de los tres experimentos del
capítulo *Experimentación y resultados* (`Documentation/Experimentacion/`).

```
training/
├── scripts/        main.py (entrenamiento), genetic_search.py, test.py, png_to_evolution.py, barridos
├── notebooks/      un notebook por experimento: Python (.ipynb, datos externos → evoluciones (T, n, n))
│                   y Wolfram (.nb, análisis y figuras del reporte)
├── CA/             evoluciones de referencia (entrada del entrenamiento)
├── NCA/            pesos entrenados (.npz) y evoluciones que generan
└── Dataset/        datos crudos descargados (no versionado)
```

Cada experimento usa el mismo nombre en `CA/` y en `NCA/` (p. ej. `CA/Sandpile/` → `NCA/Sandpile/`).
FLAME 3 separa sus datos por selección de frames: crudos en `Dataset/FLAME3/seleccion{1,2}/` y evoluciones en
`CA/FLAME3/seleccion{1,2}/`. En `NCA/FLAME3/`, `burned_states/` es la configuración final e `historial/` los
experimentos previos (ver `NCA/FLAME3/RESULTADOS.md`).

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
| 2.1 | Incendio (FLAME 3) | `FLAME3/` | `FLAME3/` | `naturales/flame.tex` | Hecho (`NCA/FLAME3/RESULTADOS.md`) |
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

Hecho. Resultados en `NCA/FLAME3/RESULTADOS.md`; proceso de datos en `notebooks/FLAME3.ipynb`.

### 3 Morfogénesis

- [ ] Definir condición inicial y configuración final.
- [ ] Agregar a `scripts/main.py` un modo cuya pérdida se calcule sobre el estado después de
      N pasos (hoy solo entrena con pares de pasos consecutivos).
- [ ] Entrenar y analizar la evolución intermedia.
