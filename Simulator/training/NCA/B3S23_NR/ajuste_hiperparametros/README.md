# Ajuste de hiperparámetros (prueba corta)

Vecindad 3 × 3 sin la esquina noroeste `(-1,-1)`, datos `CA/B3S23/evo.mat`, 100 épocas, semilla 42,
las 4 configuraciones (ReLU/2:Linear, TanH/1:Sigmoid, TanH/2:Linear, ReLU/1:Sigmoid).
`brecha` = pérdida media de las últimas 10 épocas − pérdida mínima alcanzable (0.121294).

- `ajuste_16x16.csv`: red de 16 capas de 16 neuronas (la notación 16:16 del documento).
  lr ∈ {0.0003, 0.001, 0.003} × lote ∈ {1024, 4096, 8192}. Con lr 0.003 varias redes se estancan
  (brecha hasta 0.29).
- `ajuste_2x16.csv`: primera prueba, por error con 2 capas de 16 neuronas.
  lr ∈ {0.0003, 0.001, 0.003, 0.01} × lote ∈ {1024, 4096, 8192, 32768}.

Elegido: **lr 0.001, lote 4096**. En 16:16 queda a 0.00042 del mínimo en promedio (la mejor,
lr 0.0003 / lote 1024, a 0.00038 pero 3.7 veces más lenta) y en 2 capas supera a esa combinación.
Se usa en todos los barridos de `scripts/barrido_vecindad.py`.
