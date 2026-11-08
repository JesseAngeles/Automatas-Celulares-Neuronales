"""4.2 — ¿El ortomosaico RGB de después de la quema confirma el área quemada que da la térmica?

Verdad térmica: celdas que alguna vez llegan a ≥ 135.9 °C en el escenario completo (con ignición) de 100×100.
Señales del RGB (../4_1_combustible/datos/<escenario>_100.npz): oscurecimiento (brillo antes − después), pérdida de
verdor (exg antes − después), verdor después y brillo después. Para cada señal:
    auc        probabilidad de que una celda quemada tenga la señal más "de quemado" que una no quemada
               (0.5 = no distingue; el signo se elige para que auc ≥ 0.5)
    iou        IoU entre la máscara térmica y la máscara RGB con el umbral que deja la misma fracción quemada
    corr_tmax  correlación con la temperatura máxima
Todo en todas las celdas y solo fuera de las copas (vegetación < 5 m), donde el suelo se ve desde arriba; sin
desplazar y con el desplazamiento de ../4_1_combustible/combustible.py.
Además compara la verdad térmica de 10 min que usa la evaluación (IoU q) con la del escenario completo.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/4_2_quemado_rgb/quemado_rgb.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
sys.path.insert(0, str(CARPETA.parent / "4_1_combustible"))
import comun_r as cr                                                      # noqa: E402
from combustible import DESPLAZAMIENTO, desplazar                         # noqa: E402

DATOS = CARPETA.parent / "4_1_combustible" / "datos"


def auc(senal, etiqueta):
    r = rankdata(senal)
    n1 = etiqueta.sum()
    n0 = len(etiqueta) - n1
    return float((r[etiqueta].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    cr.log(CARPETA, "Inicio 4.2 (área quemada del RGB posterior)")
    (CARPETA / "resultados.csv").unlink(missing_ok=True)
    for k in cr.ESC:
        completo = np.load(cr.TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo_celsius.npy")
        tmax = completo.max(0)
        quemado = tmax >= cr.UMBRAL_FUEGO
        t10 = cr.temperaturas()[k][0::cr.PASO][:round(cr.HORIZONTE / cr.PASO) + 1]
        q10 = (t10 >= cr.UMBRAL_FUEGO).any(0)
        cr.log(CARPETA, f"{k}: quemado en el escenario completo {quemado.mean():.3f}; en los 10 min que evalúa la "
                        f"simulación libre {q10.mean():.3f}")
        for desp in (False, True):
            m = dict(np.load(DATOS / f"{k}_100.npz"))
            if desp:
                m = {n: desplazar(v, *DESPLAZAMIENTO[k]) for n, v in m.items()}
            senales = {"oscurecimiento": m["oscurecimiento"], "perdida_verdor": m["exg_pre"] - m["exg_post"],
                       "verdor_post": m["exg_post"], "brillo_post": m["brillo_post"]}
            sin_copa = np.nan_to_num(m["altura"], nan=0) < 5
            for zona, mz in (("todas", np.ones_like(sin_copa)), ("sin_copa", sin_copa)):
                for nombre, v in senales.items():
                    ok = mz & np.isfinite(v)
                    y, x = quemado[ok], v[ok]
                    a = auc(x, y)
                    signo = 1 if a >= 0.5 else -1
                    x = signo * x
                    umbral = np.quantile(x, 1 - y.mean())
                    pred = x > umbral
                    fila = {"escenario": k, "desplazado": desp, "zona": zona, "senal": nombre,
                            "celdas": int(ok.sum()), "fraccion_quemada_termica": round(float(y.mean()), 4),
                            "auc": round(max(a, 1 - a), 4), "sentido": "alto = quemado" if signo > 0 else "bajo = quemado",
                            "iou": round(cr.iou(pred, y), 4),
                            "iou_azar": round(float(y.mean() ** 2 / (2 * y.mean() - y.mean() ** 2)), 4),
                            "corr_tmax": round(float(np.corrcoef(v[ok], tmax[ok])[0, 1]), 4)}
                    cr.escribir(CARPETA, fila)
                    cr.log(CARPETA, f"  {k} desp={desp} {zona} {nombre}: AUC {fila['auc']}, IoU {fila['iou']} "
                                    f"(azar {fila['iou_azar']}), corr T máx {fila['corr_tmax']}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
