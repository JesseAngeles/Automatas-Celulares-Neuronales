"""Video de la simulación libre de SEVIR: evolución real y de cada red desde el frame 0 (libre/*.npy de entrenar.py).

Filas: las 4 tormentas de prueba (2019) con más área con tormenta (estado ≥ 2) al inicio. Columnas: real y las 4 redes.
49 frames (4 h, paso de 5 min), 4 fps.

Escribe videos/evolucion.mp4 en esta carpeta.

Uso (desde la raíz del proyecto):
    Simulator/GUI/venv/bin/python Simulator/training/NCA/SEVIR/video.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.animation import FFMpegWriter                      # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[1]
REPO = TRAINING.parents[1]
PALETA = ListedColormap((REPO / "Simulator" / "GUI" / "colors" / "sevir_vil7.csv").read_text().split())
NOMBRES = {"5x5_2x32": "5×5, 2:32", "5x5_3x64": "5×5, 3:64", "9x9_2x32": "9×9, 2:32", "9x9_3x64": "9×9, 3:64"}
FILAS = 4
TORMENTA = 2

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def main():
    real = np.load(CARPETA / "libre" / "real.npy")
    redes = {k: np.load(CARPETA / "libre" / f"{k}.npy") for k in NOMBRES}
    meta = json.loads((TRAINING / "CA" / "SEVIR" / "prueba" / "evo.json").read_text())
    elegidas = np.argsort(-(real[:, 0] >= TORMENTA).mean((1, 2)))[:FILAS]
    columnas = [("Real", real)] + [(NOMBRES[k], v) for k, v in redes.items()]

    fig, ax = plt.subplots(FILAS, len(columnas), figsize=(2.2 * len(columnas), 2.2 * FILAS + 0.5))
    imagenes = []
    for i, e in enumerate(elegidas):
        for j, (nombre, datos) in enumerate(columnas):
            a = ax[i, j]
            imagenes.append((a.imshow(datos[e, 0], cmap=PALETA, vmin=-0.5, vmax=6.5, interpolation="nearest"),
                             datos, e))
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(nombre, fontsize=10)
            if j == 0:
                a.set_ylabel(meta["events"][e]["id"], fontsize=8)
    reloj = fig.suptitle("0 min")
    fig.tight_layout()

    (CARPETA / "videos").mkdir(exist_ok=True)
    escritor = FFMpegWriter(fps=4, codec="libopenh264", bitrate=4000)
    with escritor.saving(fig, CARPETA / "videos" / "evolucion.mp4", dpi=120):
        for t in range(real.shape[1]):
            for im, datos, e in imagenes:
                im.set_data(datos[e, t])
            reloj.set_text(f"{5 * t} min")
            escritor.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    main()
