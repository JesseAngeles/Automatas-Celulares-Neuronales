# NCA de Physarum con dado (2026-11-04)

La GUI elige siempre el estado más probable, así que una NCA normal no puede reproducir la parte aleatoria del
Physarum (el parpadeo de brillo dentro del plasmodio y las ramas). Solución sin tocar el simulador: cada célula lleva un
**dado** (0–3) dentro de su estado, `estado = 4 · visible + dado` → 24 estados.

- El dado cambia con una regla fija tipo regla 30 (`regla_dado.py`): `a' = a_izq ^ (a | a_der) ^ b`,
  `b' = b_arr ^ (b | b_ab) ^ a` (dado = 2a + b), frontera abierta. Uniforme, sin correlación espacial ni temporal, sin
  ciclo en 400 pasos. Con todos los dados en 0 se queda en 0: las condiciones iniciales llevan dados al azar.
- El visible siguiente lo decide la **maestra** (`prueba/160_6estados/red.npz`, paso 30 min, 53 vecinos, 16:64) por CDF
  inversa con u = Φ(suma gaussiana σ = 2 de los dados de 9 × 9), así que vecinos cambian juntos (sin grano).
- **Regla de localidad**: el plasmodio solo aparece si hay plasmodio en la vecindad (97.4 % del plasmodio nuevo real
  aparece a ≤ 6 células). Sin ella brotaba en la orilla de la caja desde el paso 1.
- La **alumna** aprende todo el procedimiento con `scripts/main.py`: 97 vecinos (9 × 9 + 16 de la maestra), 16:128,
  24:Linear, one-hot 24, 30 épocas, 20 pares reales (dados al azar) + 20 simulados por caja de entrenamiento.

## Red final: `b16x128` (paquete `gui/b16x128/`)

En la GUI: `--size 160`, 24 estados, **Load colors** `physarum24.csv` (un color por estado, parecido entre los 4 dados),
frontera **Open**, **Import** `iniciales/<caja>_inicio.npy` (ya traen dados al azar), **Load NCA weights**
`b16x128.npz`. Un paso = 30 min. Verificada en el núcleo real (`gui.py`): 0 células distintas contra PyTorch.

Prueba (4 dish5), medido cada 6 h (`../tabla.py`, `../tabla.json`):

| | coincidencia | plasmodio final | borde del plasmodio | cambio de brillo 6 h | 30 min |
|---|---:|---:|---:|---:|---:|
| real | – | 0.344 | 0.429 | 23.6 % | 4.7 % |
| NCA con sesgo (`barrido/e6_6h_circ6_da`) | 0.225 | 0.353 | 0.154 | 0.8 % | – |
| NCA con dado (`b16x128`) | 0.105 | 0.157 | 0.498 | 41.3 % | 6.5 % |

Ramas y movimiento interno sí; crece menos que el real y no forma venas largas.

## Pruebas de mejora (`barrido.py`, `barrido.csv`, `barrido.log`)

21 redes (la inicial + 20). Selección planeada: coincidencia en las 8 cajas de calibración (dish0/dish1), promedio
de semillas. Ganó "empujón 0.25" (`c025`: cal 0.173 ± 0.010, prueba 0.136 ± 0.005), pero al revisar las simulaciones
las redes que crecen más cubren toda la caja en 008 y 009 (donde el real queda localizado) y ninguna forma venas: la
mejora de la coincidencia viene de crecer más, no de parecerse más. Se conservó `b16x128`. Hallazgos:

- Más épocas (60/100) o coseno: área 0.15 → 0.27–0.31, coincidencia 0.08 → 0.14 en calibración, mismo borde y brillo.
- Empujón de crecimiento (constante a los logits de plasmodio de la maestra al generar datos): 0.15 → área 0.37–0.39,
  0.25 → 0.42–0.48 y borde 0.38–0.40 (ramas más gruesas).
- 24:128 no aprende (pérdida 0.056); 8:128 ≈ 16:128; 16:256 no mejora. Semillas: ±0.02 de coincidencia.
- Exploración previa del proceso ideal: temperatura < 1 apaga el crecimiento y el movimiento (`maestra.csv`); σ = 2
  de correlación quita el grano (borde 0.85 → 0.50); recortar probabilidades bajas mata el plasmodio (`pmin.csv`);
  empujón vs área en `crecer.csv`; localidad en `local.csv`/`local.png`.

## Archivos

`comun.py` (proceso ideal, métricas, paletas), `datos.py`, `entrenar.py`, `evaluar.py`, `barrido.py`, `paquete.py`,
`gui.py`, `video.py` (`videos/<red>_008-dish5.mp4` con 24 colores, `_6.mp4` con 6, `_prueba.mp4` las 4 cajas),
`redes/<nombre>/` (red.npz, evaluacion.json, libre_*.npy, gui_*.npy), `datos_*` (regenerables, ~40 MB cada uno).
Figuras del reporte: `../figuras.py` → `Documentation/img/experimentacion/naturales/Physarum/`.
