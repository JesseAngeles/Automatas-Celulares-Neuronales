"""Figuras y datos del barrido de SEVIR para el reporte (Documentation/img/experimentacion/naturales/SEVIR/).

    barrido_vecindades.png   forma de las 10 vecindades (8 del barrido + 2 de anillos)
    barrido_comparacion.png  real, predicción sin cambio y simulación libre de la mejor red y de las de anillos
                             a los 0, 60, 120 y 240 min (la misma tormenta que comparacion.png)
    barrido_area.png         área con tormenta (estado ≥ 2) en la simulación libre, promedio de las 200 de prueba
    desplazamiento.csv       desplazamiento de la tormenta a 60 min, real y de la mejor red, por tormenta

Uso (desde la raíz del proyecto, después de barrido.py y anillos.py):
    Simulator/GUI/venv/bin/python Simulator/training/NCA/SEVIR/barrido/figuras.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
import comun as c                                                  # noqa: E402

TRAINING = CARPETA.parents[2]
REPO = TRAINING.parents[1]
DESTINO = REPO / "Documentation" / "img" / "experimentacion" / "naturales" / "SEVIR"
PALETA = ListedColormap((REPO / "Simulator" / "GUI" / "colors" / "sevir_vil7.csv").read_text().split())
MINUTOS = (0, 60, 120, 240)
NOMBRES_VEC = {"5x5": "5 × 5", "9x9": "9 × 9", "11x11": "11 × 11", "circ3": "Circular r = 3",
               "circ5": "Circular r = 5", "circ6": "Circular r = 6", "flame41": "FLAME 3",
               "estrella57": "Estrella", "anillos6": "Anillos r = 6", "anillos11": "Anillos r = 11"}
REDES = {"f1_11x11_16x64_cruda_t40_k48_e7_s42_ep30": "11 × 11, 40 tormentas",
         "f2_11x11_16x64_cruda_t400_k5_e7_s42_ep30": "11 × 11, 400 tormentas",
         "f4_anillos6_16x64_cruda_t400_k5_e7_s42_ep30": "Anillos r = 6",
         "f4_anillos11_16x64_cruda_t400_k5_e7_s42_ep30": "Anillos r = 11"}
DESPLAZAMIENTO = 12      # búsqueda de ±12 células

plt.rcParams.update({"font.family": "serif", "font.size": 10, "mathtext.fontset": "cm"})


def mapa(ax, S, titulo=None):
    ax.imshow(S, cmap=PALETA, vmin=-0.5, vmax=6.5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=9)


def corrimiento(a, b, M=DESPLAZAMIENTO):
    """(dx, dy) que mejor lleva las células con tormenta de a sobre las de b (dx > 0 = este, dy > 0 = sur)."""
    a, b = (a >= c.TORMENTA).astype(np.float32), (b >= c.TORMENTA).astype(np.float32)
    mejor = (-1.0, 0, 0)
    for dy in range(-M, M + 1):
        for dx in range(-M, M + 1):
            s = float((np.roll(a, (dy, dx), (0, 1))[M:-M, M:-M] * b[M:-M, M:-M]).sum())
            if s > mejor[0]:
                mejor = (s, dx, dy)
    return mejor[1:]


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    vecindades = {**c.VECINDADES, **c.ANILLOS}

    # Vecindades
    fig, ax = plt.subplots(2, 5, figsize=(9, 4.1))
    for a, (k, nb) in zip(ax.ravel(), vecindades.items()):
        g = np.zeros((23, 23))
        for dx, dy in nb:
            g[dy + 11, dx + 11] = 1
        g[11, 11] = 2
        a.imshow(g, cmap=ListedColormap(["#f2f2f2", "#2e9e3a", "#d7191c"]), vmin=0, vmax=2)
        a.set_xticks(np.arange(-0.5, 23, 1), minor=True)
        a.set_yticks(np.arange(-0.5, 23, 1), minor=True)
        a.grid(which="minor", color="white", lw=0.3)
        a.tick_params(which="both", length=0)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"{NOMBRES_VEC[k]} ({len(nb)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(DESTINO / "barrido_vecindades.png", dpi=200)
    plt.close(fig)

    # Comparación visual: la tormenta de comparacion.png (las 20 primeras de prueba_200 son CA/SEVIR/prueba)
    real = c.cargar(["prueba_200"])
    e = int(np.argmax((real[:20, 0] >= 4).mean((1, 2))))
    filas = {"Real": real[e], "Sin cambio": np.repeat(real[e, :1], real.shape[1], 0)}
    filas.update({v: np.load(CARPETA / "libre" / f"{k}.npy", mmap_mode="r")[e] for k, v in REDES.items()
                  if "40 tormentas" not in v})
    fig, ax = plt.subplots(len(filas), len(MINUTOS), figsize=(1.7 * len(MINUTOS) + 1.1, 1.7 * len(filas)))
    for i, (nombre, L) in enumerate(filas.items()):
        for j, m in enumerate(MINUTOS):
            mapa(ax[i, j], L[m // 5], f"{m} min" if i == 0 else None)
        ax[i, 0].set_ylabel(nombre.replace(", ", ",\n"), fontsize=9)
    fig.tight_layout()
    fig.savefig(DESTINO / "barrido_comparacion.png", dpi=200)
    plt.close(fig)

    # Área con tormenta
    area_real = list(csv.DictReader(open(CARPETA / "area_real.csv")))[0]
    area = {r["clave"]: r for r in csv.DictReader(open(CARPETA / "area_libre.csv"))}
    t = [int(k[1:]) for k in area_real]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t, [100 * float(v) for v in area_real.values()], color="black", lw=2, label="Real")
    ax.plot(t, [100 * float(area_real["m0"])] * len(t), color="gray", ls="--", label="Sin cambio")
    for (k, nombre), col in zip(REDES.items(), ("#9467bd", "#1f77b4", "#2ca02c", "#d62728")):
        ax.plot(t, [100 * float(area[k][f"m{m}"]) for m in t], color=col, label=nombre)
    ax.set_xlabel("Tiempo de simulación libre (min)")
    ax.set_ylabel("Área con tormenta (%)")
    ax.set_xlim(0, 240)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(DESTINO / "barrido_area.png", dpi=200)
    plt.close(fig)

    # Desplazamiento a 60 min
    mejor = json.loads((CARPETA / "mejor.json").read_text())["clave"]
    red = np.load(CARPETA / "libre" / f"{mejor}.npy", mmap_mode="r")
    k = 60 // 5
    filas_d = []
    for i in range(len(real)):
        rx, ry = corrimiento(real[i, 0], real[i, k])
        nx, ny = corrimiento(real[i, 0], red[i, k])
        filas_d.append((i, rx, ry, nx, ny))
    with open(CARPETA / "desplazamiento.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tormenta", "real_dx", "real_dy", "red_dx", "red_dy"])
        w.writerows(filas_d)
    D = np.array(filas_d)[:, 1:].astype(float)
    print("Desplazamiento a 60 min (células; dx > 0 = este, dy > 0 = sur)")
    print(f"  real: promedio {D[:, 0].mean():.2f}, {D[:, 1].mean():.2f}; desviación {D[:, 0].std():.2f}, "
          f"{D[:, 1].std():.2f}")
    print(f"  red:  promedio {D[:, 2].mean():.2f}, {D[:, 3].mean():.2f}; desviación {D[:, 2].std():.2f}, "
          f"{D[:, 3].std():.2f}")
    print(f"  correlación dx {np.corrcoef(D[:, 0], D[:, 2])[0, 1]:.2f}, dy {np.corrcoef(D[:, 1], D[:, 3])[0, 1]:.2f}")
    print(f"  error medio red {np.abs(D[:, 2:] - D[:, :2]).mean(0)}, desplazamiento promedio "
          f"{np.abs(D[:, :2] - D[:, :2].mean(0)).mean(0)}")
    print(f"Tormenta de ejemplo: índice {e}")


if __name__ == "__main__":
    main()
