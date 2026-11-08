"""Barrido de las 27 combinaciones del historial sobre BurnedStates (funciones de entrenar.py).

Paso de 6, 3 o 1 frames (30, 15, 5 s) × red de 2 capas de 32, 3 de 64 o 4 de 256 (ReLU, salida lineal de 18) ×
vecindad 9×9, 13×13 o la dispersa de 41 celdas, como NCA/FLAME3/historial/flame3_12estados, pero con one-hot de 18 estados y
frontera abierta. Semilla 42, rondas p3 → p2 y p2 → p3, 200 épocas, exp 1e-3 → 1e-5.
Ojo: el kernel de la GUI acepta hasta 128 vecinos (MAX_VECINOS), así que las de 13×13 (169) no se cargan ahí.

Escribe datos/, redes/barrido/<ronda>/, gif/barrido/<ronda>_k<paso>_<vecindad>_<red>.gif y barrido.csv en esta carpeta;
si una combinación ya está en barrido.csv se salta. Los GIF solo se hacen con la semilla 42.

Uso (desde Simulator/training):
    python NCA/FLAME3/burned_states/barrido.py
    python NCA/FLAME3/burned_states/barrido.py --semillas 1 2 3 4 --vecindades 9x9 dispersa41
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenar as e                                               # noqa: E402

PASOS = [6, 3, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, nargs="+", default=[42])
    ap.add_argument("--vecindades", nargs="+", default=list(e.VECINDADES))
    args = ap.parse_args()
    dev = torch.device("cuda")
    S = {k: np.load(e.DATOS / v / "evo.npy").astype(np.int32) for k, v in e.ESC.items()}
    T = {k: np.load(e.DATOS / v / "evo_celsius.npy").astype(np.float32) for k, v in e.ESC.items()}
    (e.CARPETA / "datos").mkdir(exist_ok=True)
    (e.CARPETA / "gif" / "barrido").mkdir(parents=True, exist_ok=True)
    datos = {}
    for paso in PASOS:
        for k in e.ESC:
            datos[(k, paso)] = []
            for o in range(paso):
                p = e.CARPETA / "datos" / f"{k}_k{paso}_o{o}.npy"
                np.save(p, S[k][o::paso])
                datos[(k, paso)].append(p)
    destino = e.CARPETA / "barrido.csv"
    hechos = {(r["ronda"], r["paso"], r["vecindad"], r["red"], r["semilla"]) for r in csv.DictReader(open(destino))} \
        if destino.exists() else set()
    e.log(f"Inicio barrido BurnedStates: {len(PASOS) * len(e.REDES) * len(args.vecindades)} combinaciones × "
          f"{len(e.RONDAS)} rondas, semillas {args.semillas}, one-hot {e.NE}, frontera abierta")
    for semilla in args.semillas:
        for paso in PASOS:
            for red in e.REDES:
                for vec in args.vecindades:
                    for entrena, prueba in e.RONDAS:
                        ronda = f"{entrena}_a_{prueba}"
                        if (ronda, str(paso), vec, red, str(semilla)) in hechos:
                            continue
                        (e.CARPETA / "redes" / "barrido" / ronda).mkdir(parents=True, exist_ok=True)
                        nombre = f"k{paso}_{vec}_{red}"
                        base = e.CARPETA / "redes" / "barrido" / ronda / f"{nombre}_onehot_s{semilla}"
                        info = e.entrenar(datos[(entrena, paso)], base, semilla, vec, red)
                        pred, nb = e.predictor(f"{base}.npz", dev)
                        f, Ts = e.evaluar(pred, nb, S[prueba], T[prueba], paso)
                        e.escribir({"ronda": ronda, "paso": paso, "vecindad": vec, "red": red, "semilla": semilla,
                                    **info, **f}, "barrido.csv")
                        if semilla == 42:
                            e.gif(Ts, S[prueba], e.CARPETA / "gif" / "barrido" / f"{ronda}_{nombre}.gif",
                                  f"{entrena} -> {prueba}, k{paso} {vec} {red}", paso=paso)
                        e.log(f"{ronda} {nombre} s{semilla}: {info['tiempo_s']} s; mae {f['mae_c']} "
                              f"(copia {f['copia_mae_c']}), rollout {f['rollout_mae_c']} "
                              f"(copia {f['copia_rollout_mae_c']}), iou quemado {f['rollout_iou_quemado']}, "
                              f"iou fuego {f['rollout_iou_fuego_total']}")
    e.log("Fin barrido")


if __name__ == "__main__":
    main()
