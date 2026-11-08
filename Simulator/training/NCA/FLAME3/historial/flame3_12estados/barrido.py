"""Barrido de NCA sobre CA/FLAME3/seleccion2 con 12 estados (ver comun12.py).

Configuraciones (27): paso de 6, 3 o 1 frames (30, 15, 5 s) × red de 2 capas de 32, 3 de 64 o 4 de 256 (ReLU, salida
lineal) × vecindad 9×9, 13×13 o la dispersa de 41 celdas (la de NCA/FLAME3/Personal/test_01). Entrada cruda,
frontera abierta, 200 épocas, exp 1e-3 → 1e-5, lote 8192, scripts/main.py. 2 semillas (42, 1) y 2 rondas
({p1, p3} → p2 y {p1, p2} → p3). Cada red se evalúa con argmax y con el sesgo de encendido calibrado (dos filas).
Orden: paso 6 primero, luego 3 y 1; si se pasa de --horas, se detiene y lo anota en el registro.

Escribe datos/, redes/<ronda>/, resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/flame3_12estados/barrido.py --horas 4.5
    python NCA/FLAME3/historial/flame3_12estados/barrido.py --horas 2.5 --semillas 2 3 4   (semillas extra)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
import comun12 as c                                                       # noqa: E402

PASOS = [6, 3, 1]
SEMILLAS = [42, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas", type=float, default=4.5)
    ap.add_argument("--semillas", type=int, nargs="+", default=SEMILLAS)
    args = ap.parse_args()
    limite = time.time() + args.horas * 3600
    dev = torch.device("cuda")
    S, T = c.estados_y_temps()
    c.log(CARPETA, f"Inicio barrido 12 estados (límite {args.horas} h); estados: "
                   + ", ".join(f"{s}: {c.BORDES[s]:.1f}–{c.BORDES[s + 1]:.1f} °C" for s in range(c.NT)))
    (CARPETA / "datos").mkdir(exist_ok=True)
    rutas = {}
    for paso in PASOS:
        for k in c.ESC:
            rutas[(k, paso)] = []
            for o, s in enumerate(c.secuencias(S[k], paso)):
                p = CARPETA / "datos" / f"{k}_k{paso}_o{o}.npy"
                np.save(p, s)
                rutas[(k, paso)].append(p)
    for paso in PASOS:
        for red in c.REDES:
            for vec in c.VECINDADES:
                for s in args.semillas:
                    for entrena, prueba in c.RONDAS:
                        ronda = c.ronda_nombre(entrena, prueba)
                        clave = {"paso": paso, "red": red, "vecindad": vec, "ronda": ronda, "semilla": s}
                        if c.hecho(CARPETA, {**clave, "decodificador": "sesgo"}):
                            continue
                        if time.time() > limite:
                            c.log(CARPETA, f"Límite de tiempo: no se entrenó {clave} ni lo que sigue")
                            c.log(CARPETA, "Fin (por tiempo)")
                            return
                        (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                        base = CARPETA / "redes" / ronda / f"k{paso}_{vec}_{red}_s{s}"
                        info = c.entrenar([r for k in entrena for r in rutas[(k, paso)]], base, vec, red, s)
                        prob, nb = c.predictor(f"{base}.npz", dev)
                        secs = [q for k in entrena for q in c.secuencias(S[k], paso)]
                        tsecs = [q for k in entrena for q in c.secuencias(T[k], paso)]
                        cal = c.calibrar_sesgo(prob, nb, secs, tsecs)
                        for dec, p in (("argmax", prob), ("sesgo", c.con_sesgo(prob, cal["delta"]))):
                            f = {**clave, "decodificador": dec, "prueba": prueba, **info, **cal,
                                 **c.evaluar(p, nb, S[prueba], T[prueba], paso, prueba)}
                            c.escribir(CARPETA, f)
                        c.log(CARPETA, f"k{paso} {vec} {red} {ronda} s{s}: {info['tiempo_s']} s, delta {cal['delta']}; "
                                       f"mae {f['mae_c']} (copia {f['copia_mae_c']}), rollout sesgo {f['rollout_mae_c']} "
                                       f"(copia {f['copia_rollout_mae_c']}), llegada sesgo {f['llegada_mae_min']} "
                                       f"(copia {f['copia_llegada_mae_min']})")
    c.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
