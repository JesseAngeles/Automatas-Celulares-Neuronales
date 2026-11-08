"""Figuras y datos de SEVIR para el reporte (Documentation/img/experimentacion/naturales/SEVIR/).

    preprocesamiento.png  VIL original (384×384, 1 km) y sus 7 estados (96×96, 4 km) de una tormenta de prueba
    comparacion.png       real, predicción sin cambio y simulación libre de cada red a los 0, 30, 60 y 120 min
    area.png              proporción del área con tormenta (estado ≥ 2) en la simulación libre, promedio de las 20
    estados.csv           proporción de cada estado en entrenamiento y prueba

Uso (desde Simulator/training, después de entrenar.py):
    python NCA/SEVIR/figuras.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[1]
REPO = TRAINING.parents[1]
DESTINO = REPO / "Documentation" / "img" / "experimentacion" / "naturales" / "SEVIR"
PALETA = ListedColormap((REPO / "Simulator" / "GUI" / "colors" / "sevir_vil7.csv").read_text().split())
NOMBRES = {"5x5_2x32": "5×5, 2:32", "5x5_3x64": "5×5, 3:64", "9x9_2x32": "9×9, 2:32", "9x9_3x64": "9×9, 3:64"}
MINUTOS = (0, 30, 60, 120)

plt.rcParams.update({"font.family": "serif", "font.size": 10, "mathtext.fontset": "cm"})


def mapa(ax, S, titulo=None):
    ax.imshow(S, cmap=PALETA, vmin=-0.5, vmax=6.5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=9)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    real = np.load(CARPETA / "libre" / "real.npy")
    meta = json.loads((TRAINING / "CA" / "SEVIR" / "prueba" / "evo.json").read_text())
    e = int(np.argmax((real[:, 0] >= 4).mean((1, 2))))      # la de más área intensa al inicio
    evento = meta["events"][e]["id"]

    # Preprocesamiento
    vil = np.load(TRAINING / "Dataset" / "SEVIR" / "events" / f"{evento}.npz")["vil"][:, :, 0]
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(vil, cmap="turbo", vmin=0, vmax=255)
    ax[0].set_title("VIL original (384 × 384, 1 km)", fontsize=9)
    mapa(ax[1], real[e, 0], "7 estados (96 × 96, 4 km)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(DESTINO / "preprocesamiento.png", dpi=200)
    plt.close(fig)

    # Comparación visual
    filas = {"Real": real[e], "Sin cambio": np.repeat(real[e, :1], real.shape[1], 0)}
    filas.update({NOMBRES[k]: np.load(CARPETA / "libre" / f"{k}.npy")[e] for k in NOMBRES})
    fig, ax = plt.subplots(len(filas), len(MINUTOS), figsize=(1.7 * len(MINUTOS) + 0.9, 1.7 * len(filas)))
    for i, (nombre, L) in enumerate(filas.items()):
        for j, m in enumerate(MINUTOS):
            mapa(ax[i, j], L[m // 5], f"{m} min" if i == 0 else None)
        ax[i, 0].set_ylabel(nombre, fontsize=9)
    fig.tight_layout()
    fig.savefig(DESTINO / "comparacion.png", dpi=200)
    plt.close(fig)

    # Área con tormenta
    area = list(csv.DictReader(open(CARPETA / "area_libre.csv")))
    t = [float(r["minuto"]) for r in area]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t, [100 * float(r["real"]) for r in area], color="black", lw=2, label="Real")
    ax.plot(t, [100 * float(r["copia"]) for r in area], color="gray", ls="--", label="Sin cambio")
    for k, c in zip(NOMBRES, ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")):
        ax.plot(t, [100 * float(r[k]) for r in area], color=c, label=NOMBRES[k])
    ax.set_xlabel("Tiempo de simulación libre (min)")
    ax.set_ylabel("Área con tormenta (%)")
    ax.set_xlim(0, 240)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(DESTINO / "area.png", dpi=200)
    plt.close(fig)

    # Proporción de estados
    ent = np.load(TRAINING / "CA" / "SEVIR" / "entrenamiento" / "evo.npy")
    with open(CARPETA / "estados.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["estado", "entrenamiento_pct", "prueba_pct"])
        for s, (a, b) in enumerate(zip(np.bincount(ent.ravel(), minlength=7) / ent.size,
                                       np.bincount(real.ravel(), minlength=7) / real.size)):
            w.writerow([s, round(100 * a, 2), round(100 * b, 2)])
    print(f"Evento de ejemplo: {evento} ({meta['events'][e]['time_utc']}, {meta['events'][e]['event_type']})")


if __name__ == "__main__":
    main()
