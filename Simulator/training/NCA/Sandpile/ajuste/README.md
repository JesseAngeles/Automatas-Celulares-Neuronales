# Ajuste de hiperparámetros (sandpile)

Datos `CA/Sandpile/evo.mat` (25 × 25, 2 000 pasos, un generador al centro, 17 estados), vecindad de Von Neumann,
frontera abierta (`--open-boundary`), red 16:16 con salida de 17 neuronas, lote 4096, semilla 42.
`exactitud` = fracción de celdas cuyo estado siguiente acierta la red (argmax);
`exactitud_generador` = lo mismo solo en celdas generadoras (estados 8–16). La pérdida mínima alcanzable es 0.

- `ajuste_lr.csv`: las 9 configuraciones × lr ∈ {0.003, 0.001, 0.0003, 0.0001}, 100 épocas sin objetivo.
  Solo aprenden ReLU/Linear y TanH/Linear; con capas ocultas Sigmoid o salida Sigmoid/TanH la red se queda
  prediciendo 0 (exactitud 0.553 = proporción de ceros). Mejor lr: 0.0003 (ReLU/Linear 0.0187, TanH/Linear 0.0391).
- `ajuste_objetivo.csv`: ReLU/Linear con lr 0.0003, 500 épocas y objetivo ∈ {0.001, 0.0003, 0.0001, 0}.
  Ninguna llega a 0.001 (final 0.0099, exactitud 0.9942, generadores 0.916), así que el objetivo no cambia el resultado.

Elegido: **lr 0.0003, objetivo 0.001**. Se usa en `scripts/barrido_sandpile.py entrenar`.
