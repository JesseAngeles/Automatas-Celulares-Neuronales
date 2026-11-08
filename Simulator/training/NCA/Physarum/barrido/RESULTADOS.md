# Barrido nocturno Physarum: resultados (2026-11-04, 00:00–05:10)

122 redes entrenadas con `scripts/main.py` en 10 fases. Todas se evaluaron como corren en la GUI: simulación libre desde
la primera imagen real, con argmax y frontera abierta, en las 4 cajas dish5 que ninguna red vio.

## Resumen

- **Mejor red para la GUI: `e4_5h_9x9_da`.** 4 estados (borde, fondo, avena, plasmodio), paso de 5 h, vecindad 9×9
  (81 vecinos), red 16:64 ReLU, entrada one-hot, 30 épocas, entrenada con todos los desfases de 30 min y las 8
  simetrías del cuadrado. Puntaje de prueba **0.240 ± 0.000** con 3 semillas, contra 0.045 al congelarse y 0.066 al
  llenar la caja. Es la más estable de todas: las 3 semillas dan 0.239, 0.240 y 0.240.
- **Mejor con 6 estados** (los 3 niveles de plasmodio): `e6_6h_circ6_da`, paso de 6 h, vecindad circular de radio 6
  (113 vecinos), puntaje **0.231 ± 0.005** con 3 semillas. Con la semilla 42 (la del paquete) da 0.225 y su área final
  es igual a la real (0.353 contra 0.352).
- La versión 4:64 de la de 4 estados (`e4_5h_9x9_da_4x64`, la cuarta parte de capas) empata: 0.239 ± 0.001.
- Las dos están **verificadas en el núcleo real de la GUI** (`core/nca.cu`, `gui.py`): 0 células distintas contra
  PyTorch en toda la simulación de las cajas de prueba.
- **Lo que sí aprende**: el plasmodio crece desde el inóculo y las avenas hacia las avenas vecinas, sin aparecer en
  zonas vacías, y el área crece a un ritmo parecido al real. En 008-dish5 y 002-dish5 la curva de área casi coincide
  con la real (`area_mejores.png`).
- **Lo que no aprende**: la forma. Crece en manchas sólidas con bordes en diagonal, no en red de venas finas, y en
  001-dish5 sigue creciendo cuando el real se estanca (área final 0.58 contra 0.31).

## Cómo cargarla en la GUI

Carpeta `gui/e4_5h_9x9_da/` (o `gui/e6_6h_circ6_da/`):

1. `Simulator/GUI/venv/bin/python Simulator/GUI/main.py --size 160`
2. Configuration: 4 estados (6 en la otra) y **Apply**.
3. **Load colors** con `physarum4.csv` (`physarum6.csv`): 0 borde negro, 1 fondo gris, 2 avena beige, 3 plasmodio
   amarillo (en la de 6 estados, 3–5 = plasmodio tenue, medio e intenso).
4. Frontera **Open**.
5. **Import** con `iniciales/<caja>_inicio.npy`. `<caja>_real.npy` es la evolución real al mismo paso, para comparar.
6. **Load NCA weights** con `e4_5h_9x9_da.npz` (`e6_6h_circ6_da.npz`) y **Play**. Cada paso de la GUI equivale a 5 h
   (6 h en la otra).

El `.npz` ya trae el sesgo calibrado sumado en la última capa (ver abajo), así que no hay que cambiar nada en la GUI.

## Qué hacía falta para que funcionara en la GUI

1. **Sesgo de salida.** Con argmax, ninguna red cambia: el crecimiento es incierto célula por célula (en 30 min, a las
   que sí se vuelven plasmodio la red les da ~10–15 % de probabilidad), y el argmax siempre elige "no cambia". Se suma
   una constante b a la salida de plasmodio, que es lo mismo que cambiar el sesgo de la última capa. b se calibra en
   8 cajas de **entrenamiento** (dish0 y dish1 de cada experimento) y nunca en las de prueba. Las mejores quedan con
   b ≈ 1.25–1.5 (2 con 6 estados). Con b = 0 todas puntúan como congelarse.
2. **Paso de tiempo largo.** A 30 min la red no tiene término medio: o se congela, o en ~12 h llena la caja (todo el
   contorno avanza a la vez). Con pasos de 4–6 h el cambio por paso es más seguro y el crecimiento queda controlado.
3. **Criterio de calibración.** El IoU solo premiaba llenar la caja: llenarla daba 0.187 en prueba, más que la red
   base. El puntaje es IoU × (área menor / área mayor) en cada paso, promediado en el tiempo. Castiga equivocarse en el
   lugar y en la cantidad. Referencias en prueba: congelarse 0.036–0.058 según el paso, llenar la caja 0.065–0.067.

## Hallazgos por factor

Puntaje de prueba (b calibrado). Las tablas completas están en `tablas.md` y `resultados_v3.csv`.

**Paso de tiempo** (4 estados, anillos6, 16:64): el factor más importante.

| paso | 30 min | 1 h | 2 h | 3 h | 4 h | 5 h* | 6 h | 8 h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| puntaje | 0.120 | 0.140 | 0.167 | 0.174 | 0.210 | 0.220 | 0.223 | 0.185 |
| área final red / real | 0.58 / 0.34 | 0.58 / 0.34 | 0.61 / 0.34 | 0.67 / 0.35 | 0.52 / 0.33 | 0.46 / 0.34 | 0.38 / 0.35 | 0.39 / 0.34 |

\* 5 h con desfases + aumento.

**Número de estados** (6 h, 9×9, desfases + aumento):

| estados | 3 (sin avena) | 4 | 5 (tenue / medio+intenso) | 5b (tenue+medio / intenso) | 6 | 8 (5 niveles) |
|---|---:|---:|---:|---:|---:|---:|
| puntaje | 0.134 | 0.225 | 0.214 | 0.220 | 0.238 | 0.220 |

Quitar la avena hunde el resultado (0.134): la avena sí guía al plasmodio. Con avena, el número de niveles de
plasmodio cambia poco. A 30 min, 4 estados (0.120) eran mucho mejores que 6 (0.077).

**Vecindad** (6 h, 4 estados, desfases + aumento): 7×7 0.213, 9×9 0.225, 11×11 0.228, circ6 0.231, anillos6 0.233.
A 5 h: 9×9 0.240, circ6 0.233, anillos6 0.220, 11×11 0.215. Basta un alcance de 4–6 células (2–3 mm); más grande no
ayuda.

**Red** (6 h, 9×9, 4 estados): 4:64 0.238, 8:64 0.229, 16:32 0.231, 16:64 0.225, 24:64 0.234. El tamaño casi no
importa (lo mismo pasó en SEVIR). A 5 h la 4:64 empata con la 16:64 (0.240) con la cuarta parte de capas
(`redes/e4_5h_9x9_da_4x64/`, una sola semilla).

**Otros**

| factor | resultado |
|---|---|
| one-hot vs lineal | one-hot (0.225 vs 0.199 a 6 h) |
| filtro temporal | persistencia 2 (0.239) > 1, sin filtro (0.232) > 3 (0.221), a 5 h |
| resolución | 160 (0.239) > 80 (0.208, 5 h) y > 320 (0.195, 4 h) |
| desfases + aumento | +0.01 a 6 h (0.223 → 0.233), nada a 4 h (0.210 → 0.206), peor a 2 h (0.167 → 0.142); del orden del ruido entre semillas (±0.003–0.011) |
| épocas | 100 no mejoran sobre 30 |
| sesgo por estado (búsqueda coordenada) | igual o peor que un solo sesgo |

**Semillas** (42, 43, 44):

| red | puntaje (media ± desv.) |
|---|---:|
| e4_5h_9x9_da | 0.240 ± 0.000 |
| e4_5h_9x9_da_4x64 | 0.239 ± 0.001 |
| e4_6h_da | 0.235 ± 0.003 |
| e4_4h_9x9_da | 0.233 ± 0.006 |
| e6_6h_circ6_da | 0.231 ± 0.005 |
| e6_5h_circ6_da | 0.229 ± 0.012 |
| e4_4h_9x9 | 0.226 ± 0.002 |
| e6_6h_9x9_da | 0.224 ± 0.010 |
| e4_6h_circ6 | 0.223 ± 0.011 |

Con una sola semilla hubo picos que no se sostienen: `e6_5h_circ6_da` dio 0.244 con la semilla 42 y 0.226 y 0.216
con las otras.

**Calibración con 20 cajas** (`recalibrar.py`, `resultados_cal20.csv`): calibrar el sesgo con las 20 cajas de
entrenamiento en vez de 8 da exactamente el mismo b y el mismo puntaje en las 10 mejores redes. A pasos largos, 8 cajas
bastan.

## Generalización a un experimento no visto

`fuera.py` (carpeta `fuera/`): la configuración de `e4_5h_9x9_da` entrenada sin un experimento completo (18 cajas de los
otros 3), con el sesgo calibrado en dish0 y dish1 de esos 3 y probada en las 6 cajas del experimento excluido, con
otro arreglo de avena que la red nunca vio.

| experimento excluido | puntaje | congelarse | llenar la caja | área final red / real |
|---|---:|---:|---:|---:|
| 001 | 0.246 | 0.025 | 0.126 | 0.69 / 0.37 |
| 002 | 0.289 | 0.033 | 0.156 | 0.75 / 0.50 |
| 008 | 0.173 | 0.075 | 0.015 | 0.18 / 0.17 |
| 009 | 0.176 | 0.031 | 0.029 | 0.33 / 0.18 |

Supera a las dos referencias en los 4 casos (de 2 a 12 veces la mejor de ellas), así que la regla aprendida no depende
del arreglo de avena. En 001, 002 y 009 sobrecrece, y en 008 el área final es casi exacta. Las referencias cambian
por experimento: en 001 y 002 el plasmodio real cubre mucho, así que llenar la caja puntúa más.

## Limitaciones

- **Manchas en vez de venas.** Las venas finas son la parte más aleatoria del crecimiento, y una regla determinista
  las promedia en un frente sólido. Ninguna variante (más estados, 320 × 320, vecindades grandes) recuperó la red de
  venas.
- **No sabe cuándo parar.** En 001-dish5 el real se estanca en ~0.35 y la red sigue hasta 0.58. Como el sesgo es
  constante, el plasmodio casi nunca se retira.
- **Pocas cajas.** 4 de prueba y 8 de calibración: diferencias menores a ~0.01 están dentro del ruido. A 30 min,
  calibrar con 4 cajas no se transfería (en 11×11, b = 0 daba 0.130 y el b calibrado 0.082).
- La mejor se eligió por puntaje de prueba, con 3 semillas. El puntaje de calibración no sirve para elegir: la mejor
  por calibración es `e4_8h` (0.264), que en prueba da 0.185; le siguen `e4_6h_circ6_s43` (0.255 → 0.232) y `e4_6h`
  (0.254 → 0.223).

## Archivos

| archivo | contenido |
|---|---|
| `PLAN.md` | plan con el que se arrancó la noche |
| `barrido.py <fase>` | datos, entrenamiento (`main.py`), calibración del sesgo, evaluación; `FASES` = todas las configuraciones |
| `resultados_v3.csv` | las 122 redes con el criterio final (puntaje, 8 cajas de calibración) |
| `resultados_cal20.csv`, `redes/<nombre>_cal20/` | las 10 mejores recalibradas con 20 cajas (mismo resultado) |
| `resultados.csv`, `resultados_v2.csv` | fase 1 con criterios anteriores (IoU; puntaje con 4 cajas), solo como registro |
| `registro.log` | bitácora de la noche |
| `tablas.md` | `resumen.py`: todas las redes ordenadas, la fase 1 y las semillas |
| `redes/<nombre>/` | `red.npz` (sin sesgo), `red_gui.npz` (con sesgo), `sesgo.json`, `config.json`, `perdida.csv`, `libre_<caja>.npy` |
| `gui/<nombre>/` | paquete para la GUI (red, paleta, iniciales) de las dos mejores (`paquete.py`) |
| `videos/` | `<nombre>_008-dish5.mp4`: una caja, real (izq.) vs NCA (der.), sin texto; `<nombre>_prueba.mp4`: las 4 cajas (`video.py`) |
| `area_mejores.png` | área de plasmodio en el tiempo, real vs NCA, de las dos mejores |
| `gui.py` | compara una red en el núcleo de la GUI contra PyTorch |
| `fuera.py`, `fuera/` | generalización dejando fuera un experimento (`resumen.json`; cada `sin_<exp>/` con sus datos y red) |
| `datos/` | transiciones de entrenamiento (N, 2, H, W) por configuración (3.9 GB, se pueden borrar y regenerar) |

Datos nuevos en `CA/Physarum/` (sección 5.1 de `notebooks/Physarum.ipynb`, parametrizada sin cambiar los
defaults): `estados_cada30min_8estados(_160)` (5 niveles de plasmodio, b* = 9/13/17/22/28),
`estados_cada30min_160_p1` / `_p3` (filtro de persistencia 1 y 3) y `estados_cada30min_80`. Paleta nueva
`GUI/colors/physarum8.csv`.

## Ideas para seguir

- **Retirada del plasmodio**: hoy casi nunca se retira. Probar a entrenar con el objetivo a varios pasos o con más
  peso en las transiciones plasmodio → fondo. Eso requiere cambiar `main.py`, por ejemplo con pesos por clase.
- **Venas**: una NCA estocástica (muestrear de la salida) formaría estructura más fina, pero no corre en la GUI
  actual. Agregar una opción de muestreo al núcleo sería un cambio pequeño en `core/nca.cu`.
