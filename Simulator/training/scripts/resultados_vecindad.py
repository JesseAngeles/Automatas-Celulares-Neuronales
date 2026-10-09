"""Tablas y figuras del barrido de vecindades (sección 1.2 del experimento 1) para la documentación.

Lee lo que deja `barrido_vecindad.py` (resultados.csv, curvas de pérdida, evoluciones) y compara cada NCA
contra B3/S23 real:
  - patrones distintos: centros de los 512 patrones de 3 × 3 (CA/B3S23/base) que difieren tras 1 paso
  - error paso k: fracción de células distintas a B3/S23 en el paso k, media de las 10 condiciones aleatorias
  - densidad paso 50: fracción de células vivas en el paso 50, media de las 10 condiciones aleatorias

Escribe Documentation/Experimentacion/datos/*.tex y Documentation/img/vecindad_*.png.

Uso (desde Simulator/training):
    python scripts/resultados_vecindad.py           tablas y figuras
    python scripts/resultados_vecindad.py tablas    solo tablas
    python scripts/resultados_vecindad.py ajuste    imprime las tablas de ajuste (van dentro de gol.tex)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from barrido_vecindad import (ACT_TXT, CONFIGS, EPOCAS, EXPERIMENTOS, LOSS_TARGET, LOTE, LR, SEED,  # noqa: E402
                              TAMANOS, b3s23, nombre)

DOC = Path("../../Documentation")
TEX = DOC / "Experimentacion/datos"
IMG = DOC / "img"
AJUSTE = Path("NCA/B3S23_NR/ajuste_hiperparametros")
N_TEST = 10

# Rampa secuencial (un solo tono): más oscuro = red más grande
AZULES = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"

TITULOS = {
    "centro": "sin la célula central", "NO": "sin el vecino noroeste $(-1,-1)$", "N": "sin el vecino norte $(0,-1)$",
    "3x3": "con vecindad de $3\\times3$", "5x5": "con vecindad de $5\\times5$",
    "7x7": "con vecindad de $7\\times7$", "8x8": "con vecindad de $8\\times8$",
}


def referencia() -> list:
    return [np.load(f"CA/B3S23/aleatorias/test_{i}.npy") for i in range(N_TEST)]


def patrones_distintos(evo: np.ndarray) -> int:
    """Compara el centro de cada bloque de 3 × 3 tras un paso; los 529 bloques contienen 512 patrones únicos."""
    inicial, real = evo[0], b3s23(evo[0])
    vistos, distintos = set(), 0
    for i in range(0, inicial.shape[0], 3):
        for j in range(0, inicial.shape[1], 3):
            codigo = inicial[i:i + 3, j:j + 3].tobytes()
            if codigo in vistos:
                continue
            vistos.add(codigo)
            distintos += int(evo[1][i + 1, j + 1] != real[i + 1, j + 1])
    assert len(vistos) == 512
    return distintos


def metricas(exp: str, ref: list) -> list:
    carpeta = Path(EXPERIMENTOS[exp]["carpeta"])
    filas = {(r["act"], int(r["out_size"]), r["out_act"], int(r["capas"]), int(r["neuronas"])): r
             for r in csv.DictReader(open(carpeta / "resultados.csv"))}
    out = []
    for act, o, oa in CONFIGS:
        for capas, neuronas in TAMANOS:
            r = filas[(act, o, oa, capas, neuronas)]
            base = carpeta / nombre(capas, neuronas, act, o, oa)
            tests = [np.load(base / f"test_{i}.npy") for i in range(N_TEST)]
            error = np.mean([(t != r_).mean(axis=(1, 2)) for t, r_ in zip(tests, ref)], axis=0)
            densidad = np.mean([t.mean(axis=(1, 2)) for t in tests], axis=0)
            perdida = [float(x["loss"]) for x in csv.DictReader(open(base.with_suffix(".csv")))]
            out.append(dict(act=act, out_size=o, out_act=oa, capas=capas, neuronas=neuronas,
                            epocas=int(r["epocas_corridas"]), final=float(r["final"]), minimo=float(r["minimo"]),
                            patrones=patrones_distintos(np.load(str(base) + "_evo.npy")),
                            error=error, densidad=densidad, perdida=perdida))
    return out


ENCABEZADOS = ["Capa\\\\Oculta", "Capa\\\\Salida", "Épocas", "Pérdida", "Patrones\\\\distintos",
               "Error\\\\paso 1", "Error\\\\paso 50", "Densidad\\\\paso 50"]


def encabezado(texto: str) -> str:
    return f"\\textbf{{\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{texto}\\end{{tabular}}}}"


def tabla(exp: str, ms: list, dens_ref: float) -> str:
    cuerpo, previa = [], None
    for m in ms:
        config = (m["act"], m["out_size"], m["out_act"])
        if previa is not None and config != previa:
            cuerpo.append("\t\t\\midrule")
        previa = config
        cuerpo.append(
            f"\t\t{m['capas']}:{m['neuronas']}:{ACT_TXT[m['act']]} & {m['out_size']}:{ACT_TXT[m['out_act']]}"
            f" & {m['epocas']} & {m['final']:.6f} & {m['patrones']}"
            f" & {m['error'][1]:.4f} & {m['error'][-1]:.4f} & {m['densidad'][-1]:.4f} \\\\")
    # Cada tabla ocupa casi una página: empieza en una nueva para que [H] no la corte
    return "\n".join([
        "\\clearpage", "\\begin{table}[H]", "\t\\centering", "\t\\footnotesize", "\t\\setlength{\\tabcolsep}{2.5pt}",
        "\t\\begin{tabular}{llrrrrrr}", "\t\t\\toprule",
        "\t\t" + " & ".join(encabezado(t) for t in ENCABEZADOS) + " \\\\",
        "\t\t\\midrule", *cuerpo, "\t\t\\bottomrule", "\t\\end{tabular}",
        f"\t\\caption{{Regla B3/S23 {TITULOS[exp]}. Pérdida objetivo {LOSS_TARGET}, límite de épocas: {EPOCAS}, "
        f"tamaño del lote: {LOTE}, tasa de aprendizaje: {LR}, semilla aleatoria: {SEED}. "
        f"Pérdida mínima alcanzable: {ms[0]['minimo']:.6f}. "
        "Patrones: centros distintos a B3/S23 de los 512 patrones tras un paso. "
        "Error $k$: fracción de células distintas a B3/S23 en el paso $k$ y Densidad 50: fracción de células "
        f"vivas en el paso 50, ambas promedio de {N_TEST} condiciones iniciales aleatorias de $69\\times69$ "
        f"(B3/S23: {dens_ref:.4f}).}}",
        f"\t\\label{{tab:vecindad_{exp.lower()}}}", "\\end{table}", ""])


def figura(exp: str, ms: list, dens_ref: np.ndarray, ruta: Path):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ejes = plt.subplots(3, 4, figsize=(7.2, 6.4), sharey="row", constrained_layout=True)
    for c, (act, o, oa) in enumerate(CONFIGS):
        grupo = [m for m in ms if (m["act"], m["out_size"], m["out_act"]) == (act, o, oa)]
        ejes[0, c].set_title(f"{ACT_TXT[act]} & {o}:{ACT_TXT[oa]}", color=TINTA, fontsize=8)
        for m, color in zip(grupo, AZULES):
            etiqueta = f"{m['capas']}:{m['neuronas']}"
            ejes[0, c].plot(m["perdida"], color=color, lw=1.2, label=etiqueta)
            ejes[1, c].plot(m["error"], color=color, lw=1.2)
            ejes[2, c].plot(m["densidad"], color=color, lw=1.2)
        ejes[0, c].axhline(ms[0]["minimo"] if ms[0]["minimo"] > 0 else LOSS_TARGET, color=TINTA_2, lw=0.9,
                           ls=":", label="mínimo" if ms[0]["minimo"] > 0 else "objetivo")
        ejes[2, c].plot(dens_ref, color=TINTA, lw=1.0, ls="--", label="B3/S23")
        ejes[0, c].set_yscale("log")
        ejes[0, c].set_xscale("symlog", linthresh=10)
        ejes[0, c].set_xlim(0, EPOCAS)
        for f in range(3):
            ejes[f, c].grid(True, color=REJILLA, lw=0.6)
            ejes[f, c].set_axisbelow(True)
        ejes[0, c].set_xlabel("época")
        ejes[1, c].set_xlabel("paso")
        ejes[2, c].set_xlabel("paso")
    ejes[0, 0].set_ylabel("pérdida")
    ejes[1, 0].set_ylabel("error vs. B3/S23")
    ejes[2, 0].set_ylabel("densidad")
    manejadores, etiquetas = ejes[0, 0].get_legend_handles_labels()
    m2, e2 = ejes[2, 0].get_legend_handles_labels()
    fig.legend(manejadores + m2[-1:], etiquetas + e2[-1:], loc="outside upper center", ncol=8, frameon=False,
               title="capas:neuronas", title_fontsize=8)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def tabla_ajuste(archivo: str, titulo: str, label: str) -> str:
    """Pérdida final de cada configuración de activación por combinación de tasa y lote."""
    filas = list(csv.DictReader(open(AJUSTE / archivo)))
    lrs = sorted({float(r["lr"]) for r in filas})
    lotes = sorted({int(r["lote"]) for r in filas})
    cuerpo = []
    for lr in lrs:
        if cuerpo:
            cuerpo.append("\t\t\\midrule")
        for lote in lotes:
            g = {(r["act"], int(r["out_size"]), r["out_act"]): r for r in filas
                 if float(r["lr"]) == lr and int(r["lote"]) == lote}
            perdidas = " & ".join(f"{float(g[c]['final']):.6f}" for c in CONFIGS)
            segundos = np.mean([float(r["segundos"]) for r in g.values()])
            cuerpo.append(f"\t\t{lr:g} & {lote} & {perdidas} & {segundos:.0f} \\\\")
    columnas = ["Tasa", "Lote"] + [f"{ACT_TXT[a]} \\& \\\\{o}:{ACT_TXT[oa]}" for a, o, oa in CONFIGS] + ["Segundos"]
    return "\n".join([
        "\\begin{table}[H]", "\t\\centering", "\t\\small", "\t\\begin{tabular}{rrrrrrr}", "\t\t\\toprule",
        "\t\t" + " & ".join(encabezado(t) for t in columnas) + " \\\\",
        "\t\t\\midrule", *cuerpo, "\t\t\\bottomrule", "\t\\end{tabular}",
        f"\t\\caption{{{titulo} Vecindad de $3\\times3$ sin el vecino noroeste, 100 épocas, semilla 42. "
        "Pérdida final de cada configuración de activación; segundos: tiempo medio por entrenamiento. "
        "Pérdida mínima alcanzable: 0.121294.}",
        f"\t\\label{{{label}}}", "\\end{table}", ""])


def tablas_ajuste() -> str:
    return (tabla_ajuste("ajuste_16x16.csv", "Ajuste de tasa de aprendizaje y tamaño de lote con la red 16:16.",
                         "tab:vecindad_ajuste_16x16")
            + tabla_ajuste("ajuste_2x16.csv", "Ajuste de tasa de aprendizaje y tamaño de lote con la red 2:16.",
                           "tab:vecindad_ajuste_2x16"))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["ajuste"]:
        # Las tablas de ajuste están escritas dentro de Experimentacion/gol.tex: se imprimen para pegarlas
        print(tablas_ajuste())
        sys.exit()
    TEX.mkdir(parents=True, exist_ok=True)
    ref = referencia()
    dens_ref = np.mean([r.mean(axis=(1, 2)) for r in ref], axis=0)
    for exp in ["centro", "NO", "N", "3x3", "5x5", "7x7", "8x8"]:
        ms = metricas(exp, ref)
        (TEX / f"vecindad_{exp.lower()}.tex").write_text(tabla(exp, ms, dens_ref[-1]))
        if "tablas" not in sys.argv[1:]:
            figura(exp, ms, dens_ref, IMG / f"vecindad_{exp.lower()}.png")
        print(f"{exp}: tabla" + ("" if "tablas" in sys.argv[1:] else " y figura"), flush=True)
