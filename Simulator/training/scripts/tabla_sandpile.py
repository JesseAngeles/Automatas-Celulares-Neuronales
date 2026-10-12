"""Tablas LaTeX del entrenamiento de sandpile (sección 1.3 del experimento 1).

Lee los CSV de NCA/Sandpile/ e imprime las tablas para pegarlas en Documentation/Experimentacion/duplicacion/sandpile.tex:
    sandpile_ajuste_lr.tex       pérdida de las 9 configuraciones 16:16 con cuatro tasas de aprendizaje (100 épocas)
    sandpile_activaciones.tex    las 9 configuraciones 16:16 con 500 épocas
    sandpile_arquitecturas.tex   ReLU / 17:Linear con otros tamaños de red, tasas y pérdidas objetivo

Uso (desde Simulator/training): python scripts/tabla_sandpile.py
"""
import csv
from pathlib import Path

from barrido_sandpile import ACT_TXT, CARPETA, CONFIGS, EPOCAS, LOSS_TARGET, LOTE, LR, SEED

COMUN = "vecindad de Von Neumann, frontera abierta, datos de $25\\times25$ con 2000 pasos y un generador"


def leer(ruta) -> list:
    return list(csv.DictReader(open(ruta)))


def pct(x) -> str:
    return f"{100 * float(x):.3f}"


def num(x) -> str:
    """Decimal sin notación científica: 3e-05 → 0.00003."""
    return f"{float(x):.6f}".rstrip("0").rstrip(".")


def dos(texto: str) -> str:
    """Encabezado en dos renglones, cortado en el espacio más cercano a la mitad."""
    espacios = [i for i, c in enumerate(texto) if c == " "]
    corte = min(espacios, key=lambda i: abs(i - len(texto) / 2), default=-1)
    return texto if corte < 0 else f"\\shortstack{{{texto[:corte]}\\\\{texto[corte + 1:]}}}"


def tabla(columnas: str, encabezado: list, cuerpo: list, caption: str, label: str, superior: list = (),
          separacion: str = "") -> str:
    return "\n".join([
        "\\begin{table}[H]", "\t\\centering", "\t\\footnotesize",
        *([f"\t\\setlength{{\\tabcolsep}}{{{separacion}}}"] if separacion else []),
        f"\t\\begin{{tabular}}{{{columnas}}}", "\t\t\\toprule",
        *superior,
        "\t\t" + " & ".join(f"\\textbf{{{e}}}" for e in encabezado) + " \\\\", "\t\t\\midrule", *cuerpo,
        "\t\t\\bottomrule", "\t\\end{tabular}", f"\t\\caption{{{caption}}}", f"\t\\label{{{label}}}", "\\end{table}", ""])


def agrupadas(filas: list) -> list:
    """Agrupa por activación oculta con \\multirow; cada fila es (act, salida, columnas...)."""
    cuerpo = []
    for act in dict.fromkeys(f[0] for f in filas):
        grupo = [f for f in filas if f[0] == act]
        if cuerpo:
            cuerpo.append("\t\t\\midrule")
        for i, (_, out_act, *resto) in enumerate(grupo):
            primera = f"\\multirow{{{len(grupo)}}}{{*}}{{16:16:{ACT_TXT[act]}}}" if i == 0 else ""
            cuerpo.append(f"\t\t{primera} & 17:{ACT_TXT[out_act]} & " + " & ".join(resto) + " \\\\")
    return cuerpo


def ajuste_lr() -> str:
    rs = leer(CARPETA / "ajuste" / "ajuste_lr.csv")
    tasas = list(dict.fromkeys(r["lr"] for r in rs))
    perdida = {(r["act"], r["out_act"], r["lr"]): float(r["final"]) for r in rs}
    filas = [(a, s, *[f"{perdida[(a, s, lr)]:.6f}" for lr in tasas]) for a, s in CONFIGS]
    superior = [f"\t\t & & \\multicolumn{{{len(tasas)}}}{{c}}{{\\textbf{{Tasa de aprendizaje}}}} \\\\",
                f"\t\t\\cmidrule(lr){{3-{2 + len(tasas)}}}"]
    return tabla("ll" + "r" * len(tasas), ["Capa Oculta", "Capa Salida", *(num(t) for t in tasas)], agrupadas(filas),
                 f"Pérdida de la regla \\textit{{sandpile}} después de 100 épocas con cada tasa de aprendizaje: "
                 f"sin pérdida objetivo, tamaño del lote: {LOTE}, semilla aleatoria: {SEED}, {COMUN}.",
                 "tab:sp_ajuste_lr", superior)


def activaciones() -> str:
    rs = {(r["capa_oculta"].split(":")[2], r["capa_salida"].split(":")[1]): r
          for r in leer(CARPETA / "resultados.csv") if r["capa_oculta"].startswith("16:16")}
    filas = []
    for a, s in CONFIGS:
        r = rs[(ACT_TXT[a], ACT_TXT[s])]
        filas.append((a, s, f"{float(r['perdida_final']):.6f}", pct(r["exactitud"]), pct(r["exactitud_generador"])))
    return tabla("llrrr", ["Capa Oculta", "Capa Salida", "Pérdida", "Exactitud (\\%)", dos("Exactitud generador (\\%)")],
                 agrupadas(filas),
                 f"Regla \\textit{{sandpile}} con múltiples funciones de activación: pérdida objetivo {LOSS_TARGET}, "
                 f"límite de épocas: {EPOCAS}, tamaño del lote: {LOTE}, tasa de aprendizaje: {LR}, "
                 f"semilla aleatoria: {SEED}, {COMUN}.",
                 "tab:sp_activaciones")


def arquitecturas() -> str:
    # (capa oculta, lr, pérdida objetivo, registro del entrenamiento, pesos para leer la exactitud)
    corridas = []
    base = next(r for r in leer(CARPETA / "resultados.csv") if r["capa_oculta"] == "16:16:ReLU" and r["capa_salida"] == "17:Linear")
    corridas.append(("16:16:ReLU", LR, LOSS_TARGET, base["epocas"], base["perdida_final"], base["exactitud"], base["exactitud_generador"]))
    for r in sorted(leer(CARPETA / "ajuste" / "32x32" / "ajuste_lr.csv"), key=lambda r: -float(r["lr"])):
        corridas.append(("32:32:ReLU", r["lr"], r["loss_target"], r["epocas_corridas"], r["final"], r["exactitud"], r["exactitud_generador"]))
    fila_32 = next(r for r in leer(CARPETA / "resultados.csv") if r["capa_oculta"] == "32:32:ReLU")
    corridas.append(("32:32:ReLU", LR, LOSS_TARGET, fila_32["epocas"], fila_32["perdida_final"], fila_32["exactitud"], fila_32["exactitud_generador"]))
    corridas.sort(key=lambda c: (c[0] != "16:16:ReLU", -float(c[1])))
    for carpeta, objetivo in (("2000epocas", 0.001), ("objetivo0.0001", 0.0001)):
        corridas.append(("8:32:ReLU", 0.001, objetivo, *EXTRA[carpeta]))
    cuerpo = []
    for capa in dict.fromkeys(c[0] for c in corridas):
        grupo = [c for c in corridas if c[0] == capa]
        if cuerpo:
            cuerpo.append("\t\t\\midrule")
        for i, (_, lr, obj, ep, perd, ex, exg) in enumerate(grupo):
            primera = f"\\multirow{{{len(grupo)}}}{{*}}{{{capa}}}" if i == 0 and len(grupo) > 1 else (capa if i == 0 else "")
            cuerpo.append(f"\t\t{primera} & {num(lr)} & {num(obj)} & {ep} & {float(perd):.6f} & {pct(ex)} & {pct(exg)} \\\\")
    return tabla("lrrrrrr", ["Capa Oculta", dos("Tasa de aprendizaje"), dos("Pérdida objetivo"), "Épocas", "Pérdida",
                             dos("Exactitud (\\%)"), dos("Exactitud generador (\\%)")], cuerpo,
                 f"Regla \\textit{{sandpile}} con capa de salida 17:Linear y distintos tamaños de red: "
                 f"tamaño del lote: {LOTE}, semilla aleatoria: {SEED}, {COMUN}. Límite de épocas: {EPOCAS} "
                 f"(2000 en 8:32:ReLU).",
                 "tab:sp_arquitecturas", separacion="4pt")


def extra(carpeta: str) -> tuple:
    """Épocas, pérdida final y exactitudes de una corrida suelta de 8:32 (registro .csv y pesos .npz)."""
    from barrido_sandpile import cargar_muestras, exactitud
    ruta = CARPETA / carpeta / "8-32-relu_17-linear"
    log = leer(ruta.with_suffix(".csv"))
    e = exactitud(str(ruta.with_suffix(".npz")), MUESTRAS)
    return int(log[-1]["epoch"]) + 1, log[-1]["loss"], e["exactitud"], e["exactitud_generador"]


if __name__ == "__main__":
    from barrido_sandpile import cargar_muestras
    MUESTRAS = cargar_muestras()
    EXTRA = {c: extra(c) for c in ("2000epocas", "objetivo0.0001")}
    for nombre, contenido in (("sandpile_ajuste_lr.tex", ajuste_lr()), ("sandpile_activaciones.tex", activaciones()),
                              ("sandpile_arquitecturas.tex", arquitecturas())):
        print(f"% {nombre}\n{contenido}")
