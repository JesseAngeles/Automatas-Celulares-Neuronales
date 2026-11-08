"""NCA para FLAME 3 con solo 4 estados y vecindades grandes (prueba por curiosidad).

Mismo esquema que ../sin_p2 (sin ignición, sin p2; p1 → p3 y p3 → p1) pero con log4: 4 estados con cortes
iguales en log(°C) sobre la escala de p1 y p3 sin ignición (24.7–604.6 °C), representante = centro geométrico.
Sin rotaciones.

Configuraciones (2 capas de 32 ReLU, 200 épocas, exp 1e-3 → 1e-5, lote 8192, frontera abierta, semilla 42):
    pasos de 3 y 6 frames × vecindades 9×9, 13×13, 17×17, 21×21, 25×25 × entrada cruda / one-hot.

Escribe pasos.csv, registro.log y pasos/<ronda>/<nombre>.npz en esta carpeta. Métricas: las de
../k6/barrido_k6.py (mae_c en °C, comparable con log15; exact solo dentro de log4).

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/cuatro_estados/barrido_4estados.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[3]
sys.path.insert(0, str(CARPETA.parent / "sin_p2"))
import barrido_sin_p2 as sp                                                  # noqa: E402

sp.CARPETA = sp.bc.CARPETA = sp.k6.CARPETA = CARPETA      # CSV, registro y datos en esta carpeta
sp.PASOS = {3: (4, 6, 8, 10, 12), 6: (4, 6, 8, 10, 12)}
log = sp.log


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = {k: np.load(TRAINING / f"CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{k[1]}/evo_celsius.npy") for k in sp.ESC}
    disc = sp.preparar("log4", temps)
    log(f"Inicio barrido 4 estados ({dev}): cortes °C " + ", ".join(f"{x:.1f}" for x in disc["cortes"])
        + ", representantes " + ", ".join(f"{x:.1f}" for x in disc["rep"]))
    for paso, radios in sp.PASOS.items():
        for radio in radios:
            for entrada in ("cruda", "onehot"):
                for entrena, prueba in sp.RONDAS:
                    c = dict(sp.BASE, paso=paso, radio=radio, entrada=entrada)
                    sp.correr("pasos", "log4", disc, c, entrena, prueba, temps, dev)
    log("Fin")


if __name__ == "__main__":
    main()
