"""Figuras de Physarum para el reporte (Documentation/img/experimentacion/naturales/Physarum/).

    preprocesamiento_1-4.png  una caja de prueba a mitad del experimento, una imagen por paso: original, limpia,
                              6 estados (320 × 320) y 24 estados (160 × 160, visible × 4 + ruido al azar)
    sim_<k>_original.png  misma caja, real (6 estados), y simulación libre de cada resultado (estado visible):
    sim_<k>_nca.png       1 estado más probable y 2 muestreo (prueba/160_6estados, paso 30 min), 3 sesgo (paso 6 h,
                          a las 48 h) y 4 ruido (b16x128, carpeta dado/); todas a las 50 h salvo la 3
    comparacion.png       real, NCA con sesgo (paso 6 h) y NCA con ruido (paso 30 min) a los 0, 12, 24, 48 y 60 h
    finales.png           última imagen de las 4 cajas de prueba: real y NCA con ruido
    ruido_colores.png     una misma imagen de la NCA con ruido con la paleta de 6 colores y con la de 24
    ruido_regla.png       ejemplo de un paso: regla del ruido en una célula y elección del estado visible
    ruido_manchas.png     valores de ruido, número entre 0 y 1 y plasmodio con 20 % (manchas vs puntos sueltos)
    area.png              fracción de la caja con plasmodio en el tiempo, real y las dos NCA, en las 4 cajas de prueba

La NCA con ruido (antes "dado"; carpeta dado/) es la corrida del núcleo de la GUI (dado/redes/<nombre>/gui_<caja>.npy, de dado/gui.py); la de
sesgo, barrido/redes/e6_6h_circ6_da/gui_<caja>.npy.

    python NCA/Physarum/figuras.py [nombre=b16x128]      (desde Simulator/training)
    python NCA/Physarum/figuras.py preprocesamiento    solo preprocesamiento_1-4.png y sim_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[1]
REPO = TRAINING.parents[1]
sys.path.insert(0, str(CARPETA / "dado"))
import comun as C                                                  # noqa: E402

DESTINO = REPO / "Documentation" / "img" / "experimentacion" / "naturales" / "Physarum"
P6 = ListedColormap(C.PALETA6)
P24 = ListedColormap(C.PALETA24)
SESGO = CARPETA / "barrido" / "redes" / "e6_6h_circ6_da"
CAJA = C.PRUEBA[2]                                                 # 008-dish5
HORAS = (0, 12, 24, 48, 60)
COLORES = {"real": "#2a78d6", "sesgo": "#eb6834", "dado": "#1baf7a"}

plt.rcParams.update({"font.family": "serif", "font.size": 10, "mathtext.fontset": "cm"})


def mapa(ax, S, cmap=P6, n=6, titulo=None):
    ax.imshow(S, cmap=cmap, vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=9)


def series(nombre, caja):
    real = C.cargar(caja)
    sesgo = np.load(SESGO / f"gui_{caja}.npy")                     # un paso = 6 h
    dado = np.load(CARPETA / "dado" / "redes" / nombre / f"gui_{caja}.npy")
    return real, sesgo, dado


def preprocesamiento(lado=2.0, dpi=200):
    t = 100                                                        # 50 h
    nombre = f"{CAJA}.npz"
    orig = np.load(TRAINING / "Dataset" / "Physarum" / "cada30min" / nombre)["frames"][t]
    limpia = np.load(TRAINING / "Dataset" / "Physarum" / "limpio_cada30min" / nombre)["frames"][t]
    e320 = np.load(TRAINING / "CA" / "Physarum" / "estados_cada30min" / f"{CAJA}.npy")[t]
    e160 = C.cargar(CAJA)[t].astype(np.int64)
    e24 = e160 * C.DADOS + np.random.default_rng(0).integers(0, C.DADOS, e160.shape)   # ruido al azar, como paquete.py
    for k, (img, kw) in enumerate(((orig, {}), (limpia, {}),
                                   (e320, dict(cmap=P6, vmin=-0.5, vmax=5.5)),
                                   (e24, dict(cmap=P24, vmin=-0.5, vmax=C.NE - 0.5))), 1):
        fig = plt.figure(figsize=(lado, lado))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, interpolation="nearest", **kw)
        ax.set_axis_off()
        fig.savefig(DESTINO / f"preprocesamiento_{k}.png", dpi=dpi)
        plt.close(fig)


def mapa_solo(S, archivo, cmap=P6, n=6, lado=2.0, dpi=200):
    fig = plt.figure(figsize=(lado, lado))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(S, cmap=cmap, vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(DESTINO / archivo, dpi=dpi)
    plt.close(fig)


def simulaciones(nombre="b16x128"):
    """Real vs simulación libre de los 4 resultados, a las 50 h (48 h la de sesgo, que da pasos de 6 h)."""
    real = C.cargar(CAJA)
    prueba = CARPETA / "prueba" / "160_6estados"
    casos = {1: (100, np.load(prueba / f"libre_argmax_{CAJA}.npy")[100]),
             2: (100, np.load(prueba / f"libre_muestreo_{CAJA}.npy")[100]),
             3: (96, np.load(SESGO / f"gui_{CAJA}.npy")[8]),
             4: (100, np.load(CARPETA / "dado" / "redes" / nombre / f"gui_{CAJA}.npy")[100] // C.DADOS)}
    for k, (t, sim) in casos.items():
        mapa_solo(real[t], f"sim_{k}_original.png")
        mapa_solo(sim, f"sim_{k}_nca.png")


def comparacion(nombre):
    real, sesgo, dado = series(nombre, CAJA)
    fig, ax = plt.subplots(3, len(HORAS), figsize=(9, 5.6))
    for j, h in enumerate(HORAS):
        mapa(ax[0, j], real[2 * h], titulo=f"{h} h")
        mapa(ax[1, j], sesgo[h // 6])
        mapa(ax[2, j], dado[2 * h] // C.DADOS)
    for i, fila in enumerate(("Real", "NCA con sesgo", "NCA con ruido")):
        ax[i, 0].set_ylabel(fila, fontsize=9)
    fig.tight_layout()
    fig.savefig(DESTINO / "comparacion.png", dpi=200)
    plt.close(fig)


def finales(nombre):
    fig, ax = plt.subplots(2, 4, figsize=(9, 4.7))
    for j, caja in enumerate(C.PRUEBA):
        real, _, dado = series(nombre, caja)
        mapa(ax[0, j], real[-1], titulo=f"Experimento {caja[9:12]}, {(len(real) - 1) / 2:g} h")
        mapa(ax[1, j], dado[-1] // C.DADOS)
    ax[0, 0].set_ylabel("Real", fontsize=9)
    ax[1, 0].set_ylabel("NCA con ruido", fontsize=9)
    fig.tight_layout()
    fig.savefig(DESTINO / "finales.png", dpi=200)
    plt.close(fig)


def ruido_colores(nombre):
    _, _, dado = series(nombre, CAJA)
    E = dado[2 * 48]
    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
    mapa(ax[0], E // C.DADOS, titulo="6 colores (estado visible)")
    mapa(ax[1], E, P24, 24, titulo="24 colores (estado con ruido)")
    fig.tight_layout()
    fig.savefig(DESTINO / "ruido_colores.png", dpi=200)
    plt.close(fig)


def ruido_regla():
    """(a) un paso de la regla del ruido en una célula; (b) cómo el número entre 0 y 1 elige el estado visible."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    cruz = {(0, 0): 2, (-1, 0): 1, (1, 0): 0, (0, -1): 0, (0, 1): 0}          # (dx, dy) → ruido en t
    nombres = {(0, 0): "", (-1, 0): "izq", (1, 0): "der", (0, -1): "arr", (0, 1): "ab"}

    def celda(x, y, r, color="#ffffff", grueso=1.0, nota=""):
        a.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=color, ec="0.2", lw=grueso))
        a.text(x, y + 0.08, str(r), ha="center", va="center", fontsize=13)
        a.text(x, y - 0.3, f"$a={r >> 1},\\ b={r & 1}$", ha="center", va="center", fontsize=7)
        if nota:
            a.text(x, y + 0.38, nota, ha="center", va="center", fontsize=7, color="0.35")

    for (dx, dy), r in cruz.items():
        celda(dx, -dy, r, "#fff3b0" if (dx, dy) == (0, 0) else "#ffffff", 1.8 if (dx, dy) == (0, 0) else 1.0,
              nombres[(dx, dy)])
    a.add_patch(FancyArrowPatch((1.8, 0), (3.2, 0), arrowstyle="-|>", mutation_scale=14, lw=1.2, color="0.2"))
    a.text(2.5, 0.25, "un paso", ha="center", fontsize=8)
    a.add_patch(Rectangle((3.5, -0.5), 1, 1, fc="#fff3b0", ec="0.2", lw=1.8))
    a.text(4, 0.08, "3", ha="center", va="center", fontsize=13)
    a.text(4, -0.3, "$a'=1,\\ b'=1$", ha="center", va="center", fontsize=7)
    a.set_xlim(-1.6, 4.7)
    a.set_ylim(-1.6, 1.6)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("(a) Regla del ruido", fontsize=9)

    b = ax[1]
    probs = [("Fondo", 0.70, C.PALETA6[1], "white"), ("Tenue", 0.10, C.PALETA6[3], "white"),
             ("Medio", 0.15, C.PALETA6[4], "black"), ("Intenso", 0.05, C.PALETA6[5], "black")]
    x0 = 0.0
    for nom, p, col, txt in probs:
        b.add_patch(Rectangle((x0, 0), p, 1, fc=col, ec="white", lw=1))
        if p >= 0.1:
            b.text(x0 + p / 2, 0.5, f"{nom}\n{p:.0%}", ha="center", va="center", fontsize=8,
                   color=txt)
        else:
            b.text(x0 + p / 2, 1.35, f"{nom}\n{p:.0%}", ha="center", va="center", fontsize=8)
        x0 += p
    u = 0.83
    b.add_patch(FancyArrowPatch((u, -0.65), (u, -0.04), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#c0392b"))
    b.text(u, -0.85, f"número de la célula = {u}", ha="center", va="center", fontsize=8, color="#c0392b")
    for t in (0, 0.7, 0.8, 0.95, 1):
        b.text(t, 1.08, f"{t:g}", ha="center", va="bottom", fontsize=7, color="0.35")
    b.set_xlim(-0.03, 1.03)
    b.set_ylim(-1.1, 1.75)
    b.axis("off")
    b.set_title("(b) Elección del estado visible", fontsize=9)
    fig.tight_layout()
    fig.savefig(DESTINO / "ruido_regla.png", dpi=200)
    plt.close(fig)


def ruido_manchas(n=40, p=0.2):
    """Mismo ruido: (a) valores, (b) número entre 0 y 1 con los 9 × 9 vecinos, (c) plasmodio con probabilidad p usando
    (b) y (d) usando un número independiente por célula."""
    import torch
    rng = np.random.default_rng(3)
    r = C.RADIO_RUIDO
    D = torch.from_numpy(rng.integers(0, C.DADOS, (1, n + 2 * r, n + 2 * r))).to(torch.int8)
    u = C.azar(D, C.pesos_ruido(torch.device("cpu")))[0, r:-r, r:-r].numpy()
    D = D[0, r:-r, r:-r].numpy()
    ind = rng.random((n, n))
    gris = ListedColormap(["#ffffff", "#bdbdbd", "#737373", "#252525"])
    pl = ListedColormap([C.PALETA6[1], C.PALETA6[5]])
    fig, ax = plt.subplots(1, 4, figsize=(9, 2.6))
    ax[0].imshow(D, cmap=gris, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax[1].imshow(u, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[2].imshow(u >= 1 - p, cmap=pl, vmin=0, vmax=1, interpolation="nearest")
    ax[3].imshow(ind >= 1 - p, cmap=pl, vmin=0, vmax=1, interpolation="nearest")
    for a, t in zip(ax, ("(a) Valores de ruido (0 a 3)", "(b) Número entre 0 y 1",
                         "(c) Plasmodio con 20% de\nprobabilidad usando (b)",
                         "(d) Plasmodio con 20% de\nprobabilidad sin vecinos")):
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(DESTINO / "ruido_manchas.png", dpi=200)
    plt.close(fig)


def area(nombre):
    fig, ax = plt.subplots(2, 2, figsize=(8, 5.6), sharey=True)
    for a, caja in zip(ax.flat, C.PRUEBA):
        real, sesgo, dado = series(nombre, caja)
        d = real[0] != 0
        h = np.arange(len(real)) / 2
        a.plot(h, [(E[d] >= C.PL).mean() for E in real], color=COLORES["real"], label="Real")
        a.plot(np.arange(len(sesgo)) * 6, [(E[d] >= C.PL).mean() for E in sesgo], color=COLORES["sesgo"],
               marker="o", ms=3, label="NCA con sesgo (paso de 6 h)")
        a.plot(h, [(E[d] // C.DADOS >= C.PL).mean() for E in dado], color=COLORES["dado"],
               label="NCA con ruido (paso de 30 min)")
        a.set_title(f"Experimento {caja[9:12]}", fontsize=9)
        a.grid(color="0.85")
        for s in a.spines.values():
            s.set_color("0.4")
    for a in ax[1]:
        a.set_xlabel("Horas")
    for a in ax[:, 0]:
        a.set_ylabel("Fracción con plasmodio")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DESTINO / "area.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    nombre = sys.argv[1] if len(sys.argv) > 1 else "b16x128"
    DESTINO.mkdir(parents=True, exist_ok=True)
    preprocesamiento()
    simulaciones()
    if nombre == "preprocesamiento":                               # solo preprocesamiento_1-4 y sim_*
        sys.exit()
    comparacion(nombre)
    finales(nombre)
    ruido_colores(nombre)
    ruido_regla()
    ruido_manchas()
    area(nombre)
    print(DESTINO, sorted(p.name for p in DESTINO.iterdir()))
