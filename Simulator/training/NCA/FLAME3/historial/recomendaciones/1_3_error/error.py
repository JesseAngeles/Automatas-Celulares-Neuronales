"""1.3 — Dónde está el error de un paso y cuál es el techo del encendido.

categorias.csv: el error de un paso (paso 6, todas las secuencias con desfase) separado por lo que pasa de verdad con
cada celda entre t y t+1 (°C reales):
    fria_sigue       t < 130 y t+1 < 130
    enciende_fria    t < 83 y t+1 ≥ 130 (lo que mide "encendido")
    enciende_tibia   83 ≤ t < 130 y t+1 ≥ 130
    arde             t ≥ 130 y t+1 ≥ 130
    se_apaga         t ≥ 130 y t+1 < 130
  Para cada categoría: fracción de celdas, MAE en °C, parte del error total (suma de errores de la categoría / suma
  de todos) y exact, para la referencia (sin_p2, 5 semillas por separado), su ensamble, 200×200 con 9×9, la tabla
  sin vecinos de 1.1 y la copia.

techo.csv: de las celdas que se encienden (t < 83 → t+1 ≥ 130), qué fracción tenía alguna celda ≥ 130 °C a distancia
≤ d (Chebyshev, en celdas de 100×100) en t. Con d = 4 es lo que alcanza a ver la vecindad 9×9: el resto de los
encendidos no los puede predecir ninguna NCA de 9×9. También a 200×200 (d en celdas de 200×200).

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/1_3_error/error.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402

DISTANCIAS = [1, 2, 3, 4, 6, 8, 12, 16, 24]


def categorias(t_ini, t_fin):
    return {
        "fria_sigue": (t_ini < cr.FUEGO_C) & (t_fin < cr.FUEGO_C),
        "enciende_fria": (t_ini < cr.FRIO_C) & (t_fin >= cr.FUEGO_C),
        "enciende_tibia": (t_ini >= cr.FRIO_C) & (t_ini < cr.FUEGO_C) & (t_fin >= cr.FUEGO_C),
        "arde": (t_ini >= cr.FUEGO_C) & (t_fin >= cr.FUEGO_C),
        "se_apaga": (t_ini >= cr.FUEGO_C) & (t_fin < cr.FUEGO_C),
    }


def un_paso(prob, nb, t):
    """Estados predichos, °C reales en t y t+1, para todas las secuencias con desfase."""
    S = cr.estados(t)
    pred, ti, tf = [], [], []
    for o in range(cr.PASO):
        X, _ = cr.extract_patches(S[o::cr.PASO].astype(np.float32), nb, True)
        pred.append(cr.decodificar(prob(X), 1, "argmax"))
        ti.append(t[o::cr.PASO][:-1].ravel())
        tf.append(t[o::cr.PASO][1:].ravel())
    return np.concatenate(pred), np.concatenate(ti), np.concatenate(tf)


def filas_categoria(nombre, ronda, pred, ti, tf):
    err = np.abs(cr.REP[pred] - tf)
    total = err.sum()
    exacto = pred == cr.estados(tf)
    for cat, m in categorias(ti, tf).items():
        yield {"modelo": nombre, "ronda": ronda, "categoria": cat, "fraccion_celdas": round(float(m.mean()), 5),
               "mae_c": round(float(err[m].mean()), 3) if m.any() else np.nan,
               "parte_del_error": round(float(err[m].sum() / total), 4),
               "exact": round(float(exacto[m].mean()), 4) if m.any() else np.nan}
    yield {"modelo": nombre, "ronda": ronda, "categoria": "todas", "fraccion_celdas": 1.0,
           "mae_c": round(float(err.mean()), 3), "parte_del_error": 1.0, "exact": round(float(exacto.mean()), 4)}


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 1.3 (error por tipo de celda y techo del encendido)")
    for archivo in ("categorias.csv", "techo.csv"):
        (CARPETA / archivo).unlink(missing_ok=True)
    temps = {100: cr.temperaturas(100), 200: cr.temperaturas(200)}

    for entrena, prueba in cr.RONDAS:
        ronda = cr.ronda_nombre(entrena, prueba)
        t = temps[100][prueba]
        modelos = {f"referencia_s{s}": [cr.red_referencia(ronda, s)] for s in cr.SEMILLAS}
        modelos["ensamble_sin_p2"] = [cr.red_referencia(ronda, s) for s in cr.SEMILLAS]
        for nombre, redes in modelos.items():
            prob, nb = cr.predictor(redes, dev, cr.NT)
            for f in filas_categoria(nombre, ronda, *un_paso(prob, nb, t)):
                cr.escribir(CARPETA, f, "categorias.csv")
        secs = [s for k in entrena for s in cr.secs_offset(cr.estados(temps[100][k]))]
        prob, nb, _ = cr.tabla_transicion(secs, dev)
        pred, ti, tf = un_paso(prob, nb, t)
        for f in filas_categoria("tabla_sin_vecinos", ronda, pred, ti, tf):
            cr.escribir(CARPETA, f, "categorias.csv")
        for f in filas_categoria("copia", ronda, cr.estados(ti), ti, tf):
            cr.escribir(CARPETA, f, "categorias.csv")
        for s in cr.SEMILLAS:
            prob, nb = cr.predictor([cr.redes_200(ronda, s)], dev, cr.NT)
            for f in filas_categoria(f"200_9x9_s{s}", ronda, *un_paso(prob, nb, temps[200][prueba])):
                cr.escribir(CARPETA, f, "categorias.csv")
        cr.log(CARPETA, f"{ronda}: categorías listas")

    for lado in (100, 200):
        for k in cr.ESC:
            t = temps[lado][k]
            enc, dist_hits = 0, {d: 0 for d in DISTANCIAS}
            for o in range(cr.PASO):
                ts = t[o::cr.PASO]
                for a, b in zip(ts[:-1], ts[1:]):
                    m = (a < cr.FRIO_C) & (b >= cr.FUEGO_C)
                    if not m.any():
                        continue
                    enc += int(m.sum())
                    fuego = a >= cr.FUEGO_C
                    for d in DISTANCIAS:
                        dist_hits[d] += int((maximum_filter(fuego, size=2 * d + 1, mode="constant") & m).sum())
            for d in DISTANCIAS:
                cr.escribir(CARPETA, {"lado": lado, "escenario": k, "distancia_celdas": d,
                                      "vecindad": f"{2 * d + 1}x{2 * d + 1}", "encendidos": enc,
                                      "con_fuego_a_esa_distancia": dist_hits[d],
                                      "fraccion": round(dist_hits[d] / enc, 4)}, "techo.csv")
            cr.log(CARPETA, f"techo {lado} {k}: {enc} encendidos; con fuego en 9×9: "
                            f"{dist_hits[4] / enc:.3f}; en 17×17: {dist_hits[8] / enc:.3f}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
