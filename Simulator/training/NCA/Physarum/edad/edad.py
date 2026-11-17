"""Experimento 0 del plan (PLAN_ULTIMOS_ENTRENAMIENTOS.md): ¿el plasmodio recién llegado se retira más que el que
lleva tiempo?

Para cada paso de tiempo k (en imágenes de 30 min) y cada desfase o, toma la serie S[o::k] de las 20 cajas de
entrenamiento y mide

    P(retirada | edad) = P(no plasmodio en t + 1 | plasmodio en t con edad a)

donde la edad a es el número de pasos seguidos que la célula lleva con plasmodio hasta t (a = 1: llegó en el último
paso). Las células que ya tenían plasmodio al inicio de la serie no tienen edad conocida y no se cuentan.

    python NCA/Physarum/edad/edad.py      → edad.json junto a este script
"""
import json
import sys
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent
CA = CARPETA.parents[2] / "CA" / "Physarum" / "estados_cada30min_160"
PRUEBA = {f"physarum-{e}-dish5-r2c1" for e in ("001", "002", "008", "009")}
PL = 3
PASOS = {"30 min": 1, "2 h": 4, "5 h": 10, "6 h": 12}
GRUPOS = [(1, 1), (2, 3), (4, 10**9)]                       # edad en pasos: 1, 2–3, ≥ 4


def contar(S):
    """S (T, H, W) → por grupo de edad: [células con plasmodio, de ellas las que se retiran en el siguiente paso]."""
    p = S >= PL
    edad = np.zeros(S.shape[1:], np.int32)
    conocida = ~p[0]                                        # sin plasmodio al inicio: su edad se puede contar
    n = np.zeros((len(GRUPOS), 2), np.int64)
    for t in range(1, len(S) - 1):
        edad = np.where(p[t], edad + 1, 0)
        conocida = conocida | ~p[t]
        sale = p[t] & ~p[t + 1]
        for g, (a, b) in enumerate(GRUPOS):
            m = p[t] & conocida & (edad >= a) & (edad <= b)
            n[g, 0] += m.sum()
            n[g, 1] += (m & sale).sum()
    return n


def main():
    cajas = [f for f in sorted(CA.glob("physarum-*.npy")) if f.stem not in PRUEBA]
    series = [np.load(f) for f in cajas]
    res = {}
    for nombre, k in PASOS.items():
        n = np.zeros((len(GRUPOS), 2), np.int64)
        for S in series:
            for o in range(k):
                n += contar(S[o::k])
        prob = n[:, 1] / np.maximum(n[:, 0], 1)
        res[nombre] = {"celulas": n[:, 0].tolist(), "retiradas": n[:, 1].tolist(), "p_retirada": prob.tolist(),
                       "nuevo_vs_establecido": float(prob[0] / max(prob[-1], 1e-12))}
    (CARPETA / "edad.json").write_text(json.dumps({"grupos_edad_pasos": ["1", "2-3", ">=4"], "cajas": len(cajas),
                                                   "pasos": res}, indent=1))
    print(f"{'paso':>7} | {'edad 1':>14} | {'edad 2-3':>14} | {'edad >= 4':>14} | nuevo / establecido")
    for nombre, r in res.items():
        c = " | ".join(f"{p:6.1%} ({n / 1e6:5.2f} M)" for p, n in zip(r["p_retirada"], r["celulas"]))
        print(f"{nombre:>7} | {c} | {r['nuevo_vs_establecido']:.1f}×")


if __name__ == "__main__":
    sys.exit(main())
