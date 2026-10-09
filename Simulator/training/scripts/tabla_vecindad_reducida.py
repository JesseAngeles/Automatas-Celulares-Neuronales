"""Tabla LaTeX de las vecindades de 7 × 7 y 9 × 9 sin un vecino (solo red 16:16).

Lee NCA/Vecindad/{7x7,9x9}_sin_{NO,centro}/resultados.csv y escribe
Documentation/Experimentacion/datos/vecindad_reducida_ampliada.tex.

Uso (desde Simulator/training): python scripts/tabla_vecindad_reducida.py
"""
import csv
from pathlib import Path

from barrido_vecindad import ACT_TXT, CONFIGS, EPOCAS, EXPERIMENTOS, LOSS_TARGET, LOTE, LR, SEED

EXPS = ["7x7_NO", "9x9_NO", "7x7_centro", "9x9_centro"]   # agrupadas por vecino quitado
SALIDA = Path("../../Documentation/Experimentacion/datos/vecindad_reducida_ampliada.tex")


def filas(exp: str) -> tuple:
    rs = {(r["act"], int(r["out_size"]), r["out_act"]): r
          for r in csv.DictReader(open(Path(EXPERIMENTOS[exp]["carpeta"]) / "resultados.csv"))}
    cuerpo = []
    for i, config in enumerate(CONFIGS):
        r = rs[config]
        primera = f"\\multirow{{{len(CONFIGS)}}}{{*}}{{{EXPERIMENTOS[exp]['etiqueta']}}}" if i == 0 else ""
        cuerpo.append(f"\t\t{primera} & {r['capas']}:{r['neuronas']}:{ACT_TXT[r['act']]} & "
                      f"{r['out_size']}:{ACT_TXT[r['out_act']]} & {float(r['final']):.6f} \\\\")
    return cuerpo


if __name__ == "__main__":
    # Con 48 u 80 celdas casi cada patrón aparece una sola vez en los datos, así que la pérdida mínima que calcula
    # barrido_vecindad.perdida_minima sale casi en 0 por falta de muestras; se reporta la de 3 × 3 como referencia.
    cuerpo = []
    for exp in EXPS:
        if cuerpo:
            cuerpo.append("\t\t\\midrule")
        cuerpo += filas(exp)
    SALIDA.write_text("\n".join([
        "\\begin{table}[H]", "\t\\centering", "\t\\footnotesize", "\t\\begin{tabular}{lllr}", "\t\t\\toprule",
        "\t\t\\textbf{Vecindad} & \\textbf{Capa Oculta} & \\textbf{Capa Salida} & \\textbf{Pérdida} \\\\",
        "\t\t\\midrule", *cuerpo, "\t\t\\bottomrule", "\t\\end{tabular}",
        f"\t\\caption{{Regla B3/S23 con vecindades de $7\\times7$ y $9\\times9$ sin un vecino. Pérdida objetivo {LOSS_TARGET}, "
        f"límite de épocas: {EPOCAS}, tamaño del lote: {LOTE}, tasa de aprendizaje: {LR}, semilla aleatoria: {SEED}. "
        "Como referencia, la pérdida mínima alcanzable con la vecindad de $3\\times3$ sin el mismo vecino es "
        "0.121294 sin $(-1,-1)$ y 0.078650 sin $(0,0)$.}",
        "\t\\label{tab:vecindad_reducida_ampliada}", "\\end{table}", ""]))
    print(f"tabla → {SALIDA}")
