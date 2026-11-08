"""4.1 — Mapa de combustible como canal fijo (estado = T + 15·clase, como ../../opciones/3_terreno).

Clases por celda (100×100) a partir de datos/<escenario>_100.npz (mapas.py):
    altura2         copa de árbol (vegetación > 5 m) o no
    altura3         < 0.5 m (suelo o pasto), 0.5–5 m (arbustos), > 5 m (copa)
    verdor2         verdor del ortomosaico previo (2g − r − b) arriba o abajo de la mediana del escenario
    altura_verdor4  altura2 × verdor2
    *_desp          lo mismo con el mapa desplazado (dy, dx) celdas: el desplazamiento que maximiza la correlación
                    entre la altura y la temperatura máxima del escenario completo (p1: +1, +2; p3: +4, +2). Usa la
                    térmica del propio escenario (también el de prueba) solo para alinear; sin desplazar es la
                    georreferencia original.
9×9, paso 6, cruda, 2 capas de 32, scripts/main.py (--out-size 15·clases), 5 semillas. En la evaluación la clase se
mantiene fija. Además, con la semilla 42, una simulación libre donde el estado completo (temperatura y clase) es el
argmax de la red, como lo haría la GUI: clase_cambia = fracción de celdas cuya clase cambió al final, y su error.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/4_1_combustible/combustible.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402

DESPLAZAMIENTO = {"p1": (1, 2), "p3": (4, 2)}


def desplazar(v: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """v'[i, j] = v[i + dy, j + dx] (lo que queda fuera se rellena con el borde)."""
    H, W = v.shape
    i = np.clip(np.arange(H) + dy, 0, H - 1)
    j = np.clip(np.arange(W) + dx, 0, W - 1)
    return v[np.ix_(i, j)]


def clases() -> dict:
    salida = {}
    for desp in (False, True):
        for k in cr.ESC:
            m = dict(np.load(CARPETA / "datos" / f"{k}_100.npz"))
            if desp:
                m = {n: desplazar(v, *DESPLAZAMIENTO[k]) for n, v in m.items()}
            alt = np.nan_to_num(m["altura"], nan=0.0)
            exg = m["exg_pre"]
            a2 = (alt > 5).astype(np.int32)
            a3 = np.digitize(alt, [0.5, 5]).astype(np.int32)
            v2 = (exg > np.nanmedian(exg)).astype(np.int32)
            suf = "_desp" if desp else ""
            for nombre, c, n in (("altura2", a2, 2), ("altura3", a3, 3), ("verdor2", v2, 2),
                                 ("altura_verdor4", a2 * 2 + v2, 4)):
                salida.setdefault(nombre + suf, {"n": n})[k] = c
    return salida


def rollout_libre(prob, nb, t, c, n):
    """Simulación libre con el argmax sobre los 15·n estados (la clase puede cambiar, como en la GUI)."""
    pasos = round(cr.HORIZONTE / cr.PASO)
    estado = (cr.estados(t[0]) + cr.NT * c).astype(np.float32)
    for _ in range(pasos):
        X, _ = cr.extract_patches(np.stack([estado, estado]), nb, True)
        estado = prob(X).argmax(1).cpu().numpy().reshape(estado.shape).astype(np.float32)
    e = estado.astype(int)
    real = t[0::cr.PASO][pasos]
    return {"gui_clase_cambia": round(float((e // cr.NT != c).mean()), 4),
            "gui_rollout_mae_c": round(float(np.abs(cr.REP[e % cr.NT] - real).mean()), 3)}


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 4.1 (mapa de combustible)")
    temps = cr.temperaturas()
    C = clases()
    for variante, d in C.items():
        n = d["n"]
        cr.log(CARPETA, f"{variante}: fracción por clase p1 {np.bincount(d['p1'].ravel(), minlength=n) / 1e4}, "
                        f"p3 {np.bincount(d['p3'].ravel(), minlength=n) / 1e4}")
        rutas = cr.secuencias({k: cr.estados(temps[k]) + cr.NT * d[k] for k in cr.ESC}, cr.PASO,
                              CARPETA / "datos", variante)
        for entrena, prueba in cr.RONDAS:
            ronda = cr.ronda_nombre(entrena, prueba)
            for s in cr.SEMILLAS:
                clave = {"experimento": "4_1_combustible", "variante": variante, "ronda": ronda, "semilla": s}
                if cr.hecho(CARPETA, clave):
                    continue
                (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                base = CARPETA / "redes" / ronda / f"{variante}_k6_r4_cruda_2x32_exp_s{s}"
                cr.log(CARPETA, f"{variante} {ronda} s{s}: entrenando con main.py")
                info = cr.cm.entrenar_main(dict(cr.BASE, semilla=s), [r for k in entrena for r in rutas[k]], base,
                                           cr.NT * n, CARPETA)
                prob, nb = cr.predictor([f"{base}.npz"], dev, cr.NT * n)
                canal = {k: d[k] for k in cr.ESC}
                fila = {**clave, "prueba": prueba, "lado": 100, "clases": n, **info,
                        **cr.evaluar(prob, nb, temps, prueba, dev, canal=canal, ncanal=n)}
                if s == 42:
                    fila.update(rollout_libre(prob, nb, temps[prueba], d[prueba], n))
                cr.escribir(CARPETA, fila)
                cr.log(CARPETA, f"    {cr.resumen(fila)}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
