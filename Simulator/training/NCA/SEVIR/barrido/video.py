"""Video de la simulación libre de redes del barrido (por default la mejor, mejor.json) contra la evolución real.

Columnas: las 4 tormentas de prueba (2019) con más área con tormenta (estado ≥ 2) al inicio. Filas: real y cada red
(libre/<clave>.npy), desde la imagen 0. Debajo de cada panel, el área con tormenta. 49 imágenes (4 h, paso de 5 min),
4 fps.

Escribe videos/<nombre>.mp4 en esta carpeta (mejor.mp4 por default).

Uso (desde la raíz del proyecto):
    Simulator/GUI/venv/bin/python Simulator/training/NCA/SEVIR/barrido/video.py
    Simulator/GUI/venv/bin/python Simulator/training/NCA/SEVIR/barrido/video.py anillos <clave> <clave> ...
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.animation import FFMpegWriter                      # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[2]
REPO = TRAINING.parents[1]
PALETA = ListedColormap((REPO / "Simulator" / "GUI" / "colors" / "sevir_vil7.csv").read_text().split())
COLUMNAS = 4
TORMENTA = 2

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def main(salida="mejor", claves=None):
    claves = claves or [json.loads((CARPETA / "mejor.json").read_text())["clave"]]
    corridas = {r["clave"]: r for r in csv.DictReader(open(CARPETA / "resultados.csv"))}
    real = np.load(TRAINING / "CA" / "SEVIR" / "prueba_200" / "evo.npy")
    meta = json.loads((TRAINING / "CA" / "SEVIR" / "prueba_200" / "evo.json").read_text())
    elegidas = np.argsort(-(real[:, 0] >= TORMENTA).mean((1, 2)))[:COLUMNAS]
    filas = [("Real", real)]
    for k in claves:
        r = corridas[k]
        filas.append((f"{r['vecindad'].replace('x', '×')} ({r['n_vecinos']}), {r['red'].replace('x', ':')}",
                      np.load(CARPETA / "libre" / f"{k}.npy")))

    fig, ax = plt.subplots(len(filas), COLUMNAS, figsize=(2.6 * COLUMNAS, 2.9 * len(filas) + 0.5))
    paneles = []
    for i, (titulo, datos) in enumerate(filas):
        for j, e in enumerate(elegidas):
            a = ax[i, j]
            im = a.imshow(datos[e, 0], cmap=PALETA, vmin=-0.5, vmax=6.5, interpolation="nearest")
            paneles.append((a, im, datos, e))
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(meta["events"][e]["id"], fontsize=10)
            if j == 0:
                a.set_ylabel(titulo, fontsize=10)
    reloj = fig.suptitle("0 min")
    fig.tight_layout()

    (CARPETA / "videos").mkdir(exist_ok=True)
    escritor = FFMpegWriter(fps=4, codec="libopenh264", bitrate=4000)
    with escritor.saving(fig, CARPETA / "videos" / f"{salida}.mp4", dpi=120):
        for t in range(real.shape[1]):
            for a, im, datos, e in paneles:
                im.set_data(datos[e, t])
                a.set_xlabel(f"tormenta {100 * (datos[e, t] >= TORMENTA).mean():.1f} %", fontsize=9)
            reloj.set_text(f"{5 * t} min")
            escritor.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:2], sys.argv[2:])
