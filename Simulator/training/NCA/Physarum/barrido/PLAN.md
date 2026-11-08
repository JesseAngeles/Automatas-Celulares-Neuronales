# Barrido nocturno Physarum (2026-11-04, 00:00–07:00)

Pedido: "prioriza que funcione en la GUI. Haz muchas configuraciones de número de estados, tamaños de vecindad y esas
cosas. Tienes un total de 7 horas."

## Qué significa "funciona en la GUI"

El núcleo NCA de la GUI (`Simulator/core/nca.cu`) es determinista: toma el argmax de la salida, acepta hasta 128
vecinos y con frontera abierta pone 0 fuera (0 = borde, igual que en los datos). Las pruebas anteriores
(`NCA/Physarum/prueba/`) mostraron que con argmax la red no cambia nada: a las células que sí se vuelven plasmodio
les da ~10–15 % de probabilidad (contra ~0.7 % a las demás), nunca más de 50 %.

Arreglo compatible con la GUI: sumar una constante b a las salidas de plasmodio, que es lo mismo que cambiar el sesgo
de la última capa (`red_gui.npz`). b se elige en simulación libre sobre 4 cajas de entrenamiento (dish0 de cada
experimento) maximizando el IoU medio en el tiempo de la máscara de plasmodio; la prueba es en las 4 dish5, que
ninguna red ve.

## Métrica

- **IoU** (principal): IoU de la máscara de plasmodio real vs NCA libre desde t = 0 con argmax y b*, promedio en todos
  los pasos y en las 4 cajas de prueba. Castiga tanto congelarse como llenar la caja de ruido.
- También: IoU con b = 0, IoU a 6/12/24/48 h y al final, área final red/real, error medio de área, un paso
  (exactitud, plasmodio nuevo detectado, falso plasmodio).

## Fases

1. **Fase 1 (~2 h, 26 corridas)**: un factor a la vez sobre la base (160 × 160, 6 estados, anillos6 = 53 vecinos
   alcance 6, 16:64 one-hot, 30 min, 30 épocas, semilla 42):
   - estados: 3 (sin avena), 4, 5, 5b, 6, 8 (5 niveles de plasmodio, datos nuevos `estados_cada30min_8estados_160`)
   - vecindad: anillos3/4/6/8/10, 5×5, 7×7, 9×9, 11×11, circ4, circ6
   - red: 4:64, 8:64, 16:64, 8:128, 16:128; entrada lineal vs one-hot
   - paso: 30 min, 1 h, 2 h; filtro de persistencia 1 (sin filtro), 2, 3; resolución 80 × 80
2. **Fase 2 (~2.5 h)**: combinaciones de los mejores factores de la fase 1, más épocas (100) y otros sesgos/criterios
   si hace falta.
3. **Fase 3 (~1 h)**: 3 semillas de las mejores, prueba de la mejor en el núcleo real de la GUI, videos de una caja
   real vs NCA (sin texto) y de las 4 cajas, `RESULTADOS.md`, paleta.

Margen ~45 min. Todo se entrena con `scripts/main.py`; cada corrida usa ~8.6 GB de muestras (`por_caja` se ajusta
al tamaño de la vecindad y de la cuadrícula).

## Archivos

- `barrido.py <fase>`: corre una fase (se salta lo que ya está en `resultados.csv`).
- `resultados.csv`, `registro.log`
- `redes/<nombre>/`: `red.npz` (sin sesgo), `red_gui.npz` (con b*, para la GUI), `config.json`, `perdida.csv`,
  `libre_<caja>.npy` (simulación libre de prueba con b*).
- `datos/<clave>/`: transiciones de entrenamiento (N, 2, H, W).
