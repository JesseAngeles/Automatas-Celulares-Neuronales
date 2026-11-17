# Últimos entrenamientos de *Physarum polycephalum* (plan, 2026-11-16)

> **Estado: ejecutado el 2026-11-16.** Resultados, archivos y cómo cargar la red final en la GUI: `edad/RESULTADOS.md`.
> - Experimento 0: la hipótesis se confirma (4.5–4.7× a 5–6 h).
> - Experimentos A y B: con argmax no mejoran la red con sesgo.
> - Experimento C: se hizo aunque A no cumplió su criterio, porque A falló por el argmax y C muestrea. La red con
>   ruido y edad sube de *C* 0.105 a 0.155 ± 0.001 con el área casi real.
> - Propuesta de texto para el reporte, sin aplicar: `edad/propuesta_physarum.tex`.

Objetivo: probar en una última ronda, acotada a ~2 noches de GPU, si se puede replicar mejor el crecimiento real sin
perder lo que ya funciona en la GUI (argmax, ≤ 128 vecinos, sesgo dentro del `.npz`). Si nada mejora, se conservan las
redes actuales y el resultado negativo se reporta con sus números.

## Punto de partida

Medidas en las 4 cajas de prueba (dish5), simulación libre con argmax desde la primera imagen real
(`barrido/resultados_v3.csv`, `tabla.json`, `dado/barrido.csv`):

| Red | Qué es | *C* | Área final | Frontera | Cambio de brillo (6 h) |
|---|---|---|---|---|---|
| Real | — | — | 0.344 | 0.429 | 23.6% |
| `e4_5h_9x9_da` | sesgo, 4 estados, 5 h, 9 × 9, 16:64 | **0.240 ± 0.000** (3 semillas) | 0.42 | — | — |
| `e6_6h_circ6_da` | sesgo, 6 estados, 6 h, circ6, 16:64 (la del reporte) | 0.225 (semilla 42); 0.231 ± 0.005 | 0.353 | 0.154 | 0.8% |
| `b16x128` | ruido, 24 estados, 30 min, 97 vecinos, 16:128 | 0.105 | 0.157 | 0.498 | 41.3% |
| Congelada / llenar la caja | referencias | 0.036–0.058 / 0.065–0.067 | | | |

Lo que falla:

1. **Forma.** La red con sesgo crece en manchas sólidas: frontera de 0.154 contra 0.429 y casi sin cambios de brillo.
2. **No se detiene.** En 001-dish5 el área real se estanca en ~0.31 y la red llega a 0.58; el plasmodio casi nunca
   se retira.
3. **La red con ruido** sí ramifica, pero crece de menos (0.157 contra 0.344) y no forma venas largas entre avenas.
4. Todo lo que subió *C* en la red con ruido (más épocas, empuje de crecimiento) lo hizo creciendo más, e inunda
   008 y 009, donde el plasmodio real se queda junto a la avena.

La hipótesis central del reporte es que el plasmodio recién llegado se retira 4–6 veces más que el que lleva horas, y
la red solo ve el estado actual. No hay ningún script que calcule esa cifra.

## Reglas de la ronda

- Mismo protocolo de siempre: `barrido.py` para redes con sesgo y `dado/evaluar.py` para redes con ruido.
- Se elige con las 8 cajas de calibración (dish0 y dish1). Las cajas de prueba solo se usan para reportar la red final.
- Cada candidata se entrena con 3 semillas (42, 43, 44). Solo cuenta como mejora si la media supera a la base por más
  de dos desviaciones estándar. El ruido entre semillas es de ±0.003–0.012 con sesgo y ±0.02 con ruido.
- Además de *C*, se reportan área final, frontera, cambio de brillo y brotes. La meta es acercar la forma, no solo *C*.
- Toda red final se verifica en el núcleo de la GUI (`gui.py`: 0 células distintas contra PyTorch).
- Tiempos de referencia en la RTX 5080:
  - Red con sesgo: ~3 min (40–180 s de entrenamiento más calibración).
  - Red con ruido: ~5 min a 30 épocas.

## Experimento 0. Medir la hipótesis (≈ 1 h, sin entrenar)

Script nuevo `edad.py`, que sobre las 20 cajas de entrenamiento de `CA/Physarum/estados_cada30min_160` calcula

    P(plasmodio → no plasmodio en el siguiente paso | edad del plasmodio)

- Pasos de 30 min, 2 h, 5 h y 6 h.
- Edad en pasos: 1, 2–3 y ≥ 4.
- Salida: una tabla y una gráfica, que además sirven como evidencia para el reporte.

**Decisión:**
- Si el plasmodio de un paso se retira al menos 3 veces más que el de ≥ 4 pasos, se hace el experimento A.
- Si no, se descarta A y se pasa a B.

## Experimento A. Memoria de edad (principal, ≈ 1 noche)

**Idea.** Es la misma solución que funcionó en FLAME 3, donde los estados de quemado le dieron memoria al sistema.
Se agrega al estado si el plasmodio es **nuevo** (llegó en el último paso) o **establecido**.

**Por qué es viable:**
- Es determinista: la edad del siguiente estado depende solo del estado actual de la célula y de si sigue siendo
  plasmodio, así que la red la puede aprender.
- Corre en la GUI sin cambiar el núcleo; solo agrega estados.

**Datos.** En `barrido.py:datos`, opción `edad=True`, aplicada **después** de tomar una imagen cada `paso`:
- Plasmodio en *t* sin plasmodio en *t − 1* → nuevo.
- El plasmodio de la primera imagen (el inóculo) cuenta como establecido.
- Con 4 estados quedan 5 (borde, fondo, avena, plasmodio establecido, plasmodio nuevo).
- Con 6 estados quedan 9 (3 niveles × 2 edades).
- La calibración del sesgo debe sumar *b* a **todas** las salidas de plasmodio, incluidas las nuevas.

**Fase `edad` en `FASES`** (3 semillas cada una, 12 redes, ~40 min):

| Nombre | Base | Cambio |
|---|---|---|
| `e4_5h_9x9_da_edad` | `e4_5h_9x9_da` (0.240) | 5 estados |
| `e4_6h_9x9_da_edad` | `e4_6h_da` (0.235) | 5 estados, vecindad 9 × 9 |
| `e6_6h_circ6_da_edad` | `e6_6h_circ6_da` (0.231) | 9 estados |
| `e4_5h_9x9_da_edad_vector` | la primera | `vector=True`: un sesgo por estado, para que el nuevo tenga el suyo |

**Éxito**, cualquiera de las dos condiciones sin bajar *C* de su base:
- *C* ≥ 0.25.
- Frontera > 0.20, o área final de 001-dish5 por debajo de 0.45.

## Experimento B. Más peso a la retirada (≈ medio día de código y una noche)

Ataca que la red no se detiene. Requiere un cambio pequeño en `scripts/main.py`:
- **Opción nueva `--retreat-weight W`.** Multiplica por *W* la pérdida de las muestras cuya célula central es
  plasmodio y cuyo siguiente estado no lo es (`CrossEntropyLoss(reduction="none")`).
- **Default 1**, así que no cambia nada de lo anterior.
- **Documentación.** Documentarla en `CLAUDE.md` junto a las demás opciones.

Corridas: *W* ∈ {2, 4, 8} sobre la mejor configuración de A, o sobre `e4_5h_9x9_da` si A no mejoró. Con 2 semillas
son 6 redes.

**Éxito:** área final de 001-dish5 más cerca de la real (~0.31), sin que *C* baje más de 0.01 ni que 008/009 dejen de
crecer.

## Experimento C. Red con ruido con memoria (opcional, solo si A funciona y queda tiempo)

La red con ruido es la única que forma ramas. Ninguno de estos dos cambios se ha probado:

- **C1, maestra a paso largo.** La maestra actual (30 min) da *p* ≈ 9% a las células que sí crecen. A 2 h el
  crecimiento es más probable y el muestreo menos disperso.
- **C2, maestra con edad.** Se usa como maestra la mejor red de A. Al muestrear, lo nuevo puede retirarse y lo
  establecido quedarse, que es como se forman las venas. Con 9 estados visibles y 4 de ruido son 36 estados con
  *one-hot*; hay que adaptar `dado/comun.py` y `dado/datos.py`.

Corridas: 2 configuraciones × 2 semillas, ~5 min cada una.

**Éxito:** *C* ≥ 0.15 con frontera ≈ 0.43 y área ≈ 0.34.

## Fuera de alcance (queda como trabajo a futuro)

- Pérdida a varios pasos (BPTT), que requiere una simulación diferenciable en `main.py`.
- Muestreo dentro de `core/nca.cu` (NCA estocástica en la GUI).
- Vecindades de más de 128 células.

## Cierre (se hace aunque nada mejore)

1. Elegir la red final por *C* de calibración con 3 semillas.
2. Reportar *C* de prueba y la prueba dejando fuera un experimento (`fuera.py`).
3. Verificar con `gui.py`, empaquetar con `paquete.py` y generar los videos con `video.py`.
4. Actualizar:
   - `physarum.tex`
   - las diapositivas 23–25 y los anexos de *Physarum* de la presentación
   - `pruebas/instrucciones.csv` (con el comando de la red final)

## Calendario sugerido

| Día | Trabajo | Noche (GPU) |
|---|---|---|
| 1 | Experimento 0; opción `edad` en `barrido.py` | Fase `edad` (12 redes) |
| 2 | Evaluar A; `--retreat-weight` en `main.py` | Experimento B (6 redes) |
| 3 | (Opcional) adaptar `dado/` para C | Experimento C (4 redes) |
| 4 | Cierre: semillas, `fuera.py`, GUI y videos | — |
| 5 | Reporte y presentación | — |

## Discrepancias encontradas al revisar (corregir en el reporte)

- **Duración de los experimentos.** `physarum.tex` dice 63 h para 001 y 002 y 141 h para 008 y 009 ("respectivamente").
  Los datos limpios a 30 min dan otra cosa:
  - 001: 141 h (283 imágenes)
  - 002: 127 h
  - 008: 63 h
  - 009: 113 h
- **Pares y simetrías.** El reporte dice que la red de 6 estados usó todos los pares a 6 h y las 8 simetrías de cada
  uno. El código toma 37 pares por caja y aplica **una** simetría al azar a cada par.
- **Estados del barrido.** El reporte habla de "122 redes con 4 y 6 estados", pero el barrido también usó 3, 5, 5b
  y 8 estados.
- **Mejor red.** La mejor red del barrido, `e4_5h_9x9_da` (0.240), no aparece en el reporte, y la tabla de pasos de
  tiempo omite el de 5 h.
- **Valor de *C* de la red con ruido.** Aparece como 0.105, 0.110 o 0.115 según cómo se midió:
  - 0.105: cada 6 h, en la GUI
  - 0.110: cada 30 min, en PyTorch
  - 0.115: media de 3 semillas

  Conviene usar uno solo y decir cuál.
