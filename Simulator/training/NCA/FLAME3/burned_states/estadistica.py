"""Comparación estadística de las configuraciones de barrido.csv que caben en el simulador (9×9 y dispersa41).

Cada bloque es una semilla en una prueba (p3 → p2 o p2 → p3); dentro de cada bloque se ordenan las configuraciones en
4 métricas y el puntaje del bloque es el promedio de esos 4 lugares (1 = mejor):
    error1   error de un paso ÷ el de la copia (menor es mejor)
    libre    error al final de la simulación libre ÷ el de la copia (menor es mejor)
    iou_q    IoU del área quemada al final de la simulación libre (mayor es mejor)
    iou_f    IoU de las células que alguna vez llegaron a 130 °C (mayor es mejor)
Con los puntajes se aplica la prueba de Friedman y la diferencia crítica de Nemenyi (Demšar 2006, α = 0.05): las
configuraciones cuyo lugar promedio está a menos de la diferencia crítica de la mejor no se distinguen de ella.
Se hacen dos rankings: "combinado" (las 4 métricas) y "area" (solo iou_q, el criterio visual del usuario).

Escribe estadistica.csv (una fila por configuración) y estadistica.txt en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/burned_states/estadistica.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CARPETA = Path(__file__).resolve().parent
VECINDADES = {"9x9": 81, "dispersa41": 41}
SEGUNDOS = {"6": 30, "3": 15, "1": 5}
METRICAS = {"error1": -1, "libre": -1, "iou_q": 1, "iou_f": 1}      # 1: mayor es mejor
ALFA = 0.05


def metricas(r: dict) -> dict:
    f = lambda c: float(r[c])
    return {"error1": f("mae_c") / f("copia_mae_c"), "libre": f("rollout_mae_c") / f("copia_rollout_mae_c"),
            "iou_q": f("rollout_iou_quemado"), "iou_f": f("rollout_iou_fuego_total")}


def main():
    filas = [r for r in csv.DictReader(open(CARPETA / "barrido.csv")) if r["vecindad"] in VECINDADES]
    datos = defaultdict(dict)                                       # config -> bloque -> métricas
    for r in filas:
        datos[(r["paso"], r["vecindad"], r["red"])][(r["ronda"], r["semilla"])] = metricas(r)
    configs = sorted(datos)
    bloques = sorted(set.intersection(*(set(d) for d in datos.values())))
    k, n = len(configs), len(bloques)

    q = stats.studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    rankings = {}
    for nombre, usadas in (("combinado", list(METRICAS)), ("area", ["iou_q"])):
        puntaje = np.zeros((n, k))                                  # lugar promedio de cada config en cada bloque
        for i, b in enumerate(bloques):
            for m in usadas:
                v = np.array([datos[c][b][m] for c in configs])
                puntaje[i] += stats.rankdata(-METRICAS[m] * v)
            puntaje[i] /= len(usadas)
        lugar = np.vstack([stats.rankdata(p) for p in puntaje])     # lugar final de cada config en cada bloque
        rankings[nombre] = (lugar.mean(0), *stats.friedmanchisquare(*lugar.T))

    lugar_medio = rankings["combinado"][0]
    orden = np.argsort(lugar_medio)
    salida = []
    for j in orden:
        c = configs[j]
        fila = {"paso_s": SEGUNDOS[c[0]], "vecindad": c[1], "vecinos": VECINDADES[c[1]], "red": c[2]}
        for nombre, (lm, _, _) in rankings.items():
            fila[f"lugar_{nombre}"] = round(lm[j], 2)
            fila[f"igual_a_la_mejor_{nombre}"] = lm[j] - lm.min() <= cd
        for m in METRICAS:
            for ronda in ("p3_a_p2", "p2_a_p3"):
                v = [datos[c][b][m] for b in bloques if b[0] == ronda]
                fila[f"{m}_{ronda}_media"] = round(float(np.mean(v)), 4)
                fila[f"{m}_{ronda}_desv"] = round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4)
            v = [datos[c][b][m] for b in bloques]
            fila[f"{m}_media"] = round(float(np.mean(v)), 4)
        salida.append(fila)
    with open(CARPETA / "estadistica.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(salida[0]))
        w.writeheader()
        w.writerows(salida)

    texto = [f"{k} configuraciones, {n} bloques (semilla × prueba): {bloques}",
             f"Nemenyi: q = {q:.3f}, diferencia crítica = {cd:.2f} lugares"]
    texto += [f"Friedman {nombre}: chi2 = {chi2:.2f}, p = {p:.2e}" for nombre, (_, chi2, p) in rankings.items()]
    texto += ["", "comb.   área"]
    for s in salida:
        texto.append(f"{s['lugar_combinado']:5.2f}{'*' if s['igual_a_la_mejor_combinado'] else ' '} "
                     f"{s['lugar_area']:5.2f}{'*' if s['igual_a_la_mejor_area'] else ' '} {s['paso_s']:2d} s "
                     f"{s['vecindad']:10s} {s['red']:6s} error1 {s['error1_media']:.3f} libre {s['libre_media']:.3f} "
                     f"iou_q {s['iou_q_media']:.3f} iou_f {s['iou_f_media']:.3f}")
    texto.append("* = no se distingue de la mejor (diferencia menor a la crítica)")
    (CARPETA / "estadistica.txt").write_text("\n".join(texto) + "\n")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
