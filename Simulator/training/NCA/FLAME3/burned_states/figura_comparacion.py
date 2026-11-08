"""Imágenes para las figuras del reporte: evolución real contra la simulación libre de 9 configuraciones del barrido
(semilla 42), en 1/4, 2/4 y 3/4 de la secuencia.

    --plot 1   prueba en flame_plot_1 (entrenadas con flame_plot_2)
    --plot 2   prueba en flame_plot_2 (entrenadas con flame_plot_1)
En ambas, las 9 configuraciones de la selección visual de las dos pruebas (comparacion/prueba_flame_plot_1/best y
comparacion/prueba_flame_plot_2/best).

La simulación libre parte de la primera imagen, como en barrido.py. Los cuartos se redondean a múltiplos de 6 imágenes
(30 s) para que todas las redes, también las de paso de 30 s, tengan un estado en ese tiempo:
    flame_plot_1: 116 imágenes → 30, 60 y 84 (150, 300 y 420 s)
    flame_plot_2:  66 imágenes → 18, 30 y 48 (90, 150 y 240 s)

Escribe Documentation/img/experimentacion/naturales/Flame3/comparacion/flame_plot_<N>/<nombre>_<segundos>s.png
(400 × 400 px).

Uso (desde Simulator/training):
    python NCA/FLAME3/burned_states/figura_comparacion.py --plot 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenar as e                                               # noqa: E402

SELECCION = [(3, "9x9", "2x32"), (3, "9x9", "3x64"), (3, "dispersa41", "3x64"), (3, "dispersa41", "4x256"),
             (6, "9x9", "2x32"), (6, "9x9", "3x64"), (6, "dispersa41", "3x64"),
             (1, "9x9", "4x256"), (3, "9x9", "4x256")]                # selección de la prueba en flame_plot_2
PLOTS = {                                                          # p2 = flame_plot_1, p3 = flame_plot_2
    1: {"prueba": "p2", "entrena": "p3", "frames": [30, 60, 84]},
    2: {"prueba": "p3", "entrena": "p2", "frames": [18, 30, 48]},
}
ZOOM = 4
IMG = e.TRAINING.parents[1] / "Documentation" / "img" / "experimentacion" / "naturales" / "Flame3" / "comparacion"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", type=int, choices=PLOTS, required=True)
    p = PLOTS[ap.parse_args().plot]
    destino = IMG / f"flame_plot_{ap.parse_args().plot}"
    destino.mkdir(parents=True, exist_ok=True)

    def guardar(estado: np.ndarray, nombre: str):
        Image.fromarray(e.PALETA[estado].repeat(ZOOM, 0).repeat(ZOOM, 1)).save(destino / f"{nombre}.png")

    dev = torch.device("cuda")
    S = np.load(e.DATOS / e.ESC[p["prueba"]] / "evo.npy").astype(np.int32)
    for f in p["frames"]:
        guardar(S[f], f"real_{f * 5}s")
    for paso, vec, red in SELECCION:
        nombre = f"k{paso}_{vec}_{red}"
        ronda = f"{p['entrena']}_a_{p['prueba']}"
        pred, nb = e.predictor(e.CARPETA / "redes" / "barrido" / ronda / f"{nombre}_onehot_s42.npz", dev)
        estado, t = S[0], 0
        for f in p["frames"]:
            while t < f:
                estado = e.paso_red(pred, nb, estado)
                t += paso
            guardar(estado, f"paso{paso * 5}s_{vec}_{red}_{f * 5}s")
        print(nombre, "listo")


if __name__ == "__main__":
    main()
