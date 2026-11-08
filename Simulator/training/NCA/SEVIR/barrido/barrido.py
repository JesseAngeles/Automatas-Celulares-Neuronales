"""Barrido de NCA sobre SEVIR con scripts/main.py (plan en PLAN.md). Todo en secuencia; lo que ya está en
resultados.csv se salta.

    Fase 1  pruebas pequeñas: 8 vecindades × 5 redes × cruda/one-hot, 40 tormentas completas, 30 épocas
    Fase 2  las 4 mejores de la fase 1 con más tormentas, sin pasar del límite de memoria de main.py:
            400 tormentas × 5 transiciones, 400 tormentas × todas las que quepan, 1,000 tormentas × las que quepan
    Fase 3  la mejor de la fase 2: semillas 1 y 2, 4 estados y 100 épocas

Datos: CA/SEVIR/todo_2018 + todo_2017 (las primeras 40 de todo_2018 son CA/SEVIR/entrenamiento); prueba =
CA/SEVIR/prueba_200 (200 tormentas de 2019; las primeras 20 son CA/SEVIR/prueba). Las transiciones elegidas se
guardan en CA/SEVIR/barrido/.

Escribe redes/fase*/<clave>.npz (+ _loss.csv), resultados.csv, ranking_fase1.csv, mejor.json, area_libre.csv,
area_real.csv, libre/<clave>.npy (fases 2 y 3) y registro.log en esta carpeta.

Uso (desde la raíz del proyecto):
    Simulator/GUI/venv/bin/python Simulator/training/NCA/SEVIR/barrido/barrido.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun as c                                                   # noqa: E402

CARPETA = c.CARPETA
DATOS = c.CA / "barrido"
REDES = {"2x32": [32] * 2, "3x64": [64] * 3, "8x64": [64] * 8, "16x64": [64] * 16, "16x128": [128] * 16}
ENTRADAS = {"cruda": False, "onehot": True}
EPOCAS = 30
EPOCAS_LARGO = 100
FINALISTAS = 4
ENTRENA = ["todo_2018", "todo_2017"]
LIMITE = 9.0e9            # bytes de muestras (float32) que main.py arma; 40 tormentas con 11×11 = 8.7 GB
CELDAS = 96 * 96


def log(msg: str):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(CARPETA / "registro.log", "a") as f:
        f.write(linea + "\n")


def hechos() -> dict:
    ruta = CARPETA / "resultados.csv"
    if not ruta.exists():
        return {}
    return {r["clave"]: r for r in csv.DictReader(open(ruta))}


def escribir(fila: dict, archivo="resultados.csv"):
    ruta = CARPETA / archivo
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def transiciones_max(n_vecinos: int) -> int:
    return int(LIMITE // (CELDAS * n_vecinos * 4))


def datos(tormentas: int, k: int, seg: str) -> Path:
    """Archivo con k transiciones (t, t + 1) de cada una de las primeras `tormentas`; k = 48 → evoluciones completas.
    Los pasos de cada tormenta salen de una permutación fija (semilla 0), así que k = 5 ⊂ k = 10 ⊂ ..."""
    ruta = DATOS / f"t{tormentas}_k{k}_e{seg}.npy"
    if not ruta.exists():
        DATOS.mkdir(parents=True, exist_ok=True)
        S = c.cargar(ENTRENA, tormentas, seg)
        if k >= S.shape[1] - 1:
            np.save(ruta, S)
        else:
            rng = np.random.default_rng(0)
            pasos = np.stack([rng.permutation(S.shape[1] - 1)[:k] for _ in range(tormentas)])
            par = np.stack([S[np.arange(tormentas)[:, None], pasos], S[np.arange(tormentas)[:, None], pasos + 1]], 2)
            np.save(ruta, par.reshape(-1, 2, *S.shape[2:]))
    return ruta


class Corrida:
    def __init__(self, dev):
        self.dev = dev
        self.prueba = {s: c.cargar(["prueba_200"], segmentacion=s) for s in c.SEGMENTACIONES}

    def correr(self, fase, vec, red, entrada, tormentas=40, k=48, seg="7", semilla=42, epocas=EPOCAS,
               libre=False) -> dict:
        clave = f"f{fase}_{vec}_{red}_{entrada}_t{tormentas}_k{k}_e{seg}_s{semilla}_ep{epocas}"
        previo = hechos().get(clave)
        if previo:
            return previo
        nb = c.VECINDADES[vec]
        estados = 7 if seg == "7" else 4
        ruta = CARPETA / "redes" / f"fase{fase}" / f"{clave}.npz"
        r = c.entrenar([datos(tormentas, k, seg)], nb, REDES[red], estados, ENTRADAS[entrada], epocas, ruta,
                       semilla)
        f, area_libre, area_real = c.evaluar(c.modelo(ruta, self.dev), nb, self.prueba[seg], self.dev,
                                             CARPETA / "libre" / f"{clave}.npy" if libre else None)
        fila = {"clave": clave, "fase": fase, "vecindad": vec, "n_vecinos": len(nb), "red": red, "entrada": entrada,
                "estados": estados, "tormentas": tormentas, "transiciones": tormentas * min(k, 48),
                "muestras": r["muestras"], "epocas": r["epocas"], "semilla": semilla, "perdida": r["perdida"],
                "tiempo_s": r["tiempo_s"], **f}
        escribir(fila)
        escribir({"clave": clave, **{f"m{5 * t}": round(a, 5) for t, a in enumerate(area_libre)}}, "area_libre.csv")
        if not (CARPETA / "area_real.csv").exists():
            escribir({f"m{5 * t}": round(a, 5) for t, a in enumerate(area_real)}, "area_real.csv")
        log(f"{clave}: pérdida {r['perdida']}, {r['tiempo_s']} s | un paso área {f['area']} (copia {f['copia_area']})"
            f" | libre 60 área {f['libre60_area']} (copia {f['copia_libre60_area']}), proporción 120 "
            f"{f['libre120_proporcion']}")
        torch.cuda.empty_cache()
        return {k_: str(v) for k_, v in fila.items()}


def puntaje(filas: list) -> list:
    """Promedio de los lugares en (1) área con tormenta a un paso, (2) y (3) área con tormenta en simulación libre a
    60 y 120 min, (4) qué tan lejos queda la cantidad de tormenta simulada a 120 min de la real (|log proporción|).
    Menor = mejor."""
    criterios = [lambda r: -float(r["area"]), lambda r: -float(r["libre60_area"]),
                 lambda r: -float(r["libre120_area"]), lambda r: abs(np.log(max(float(r["libre120_proporcion"]), 1e-6)))]
    lugares = np.zeros(len(filas))
    for crit in criterios:
        orden = np.argsort([crit(r) for r in filas], kind="stable")
        lugares[orden] += np.arange(1, len(filas) + 1)
    lugares /= len(criterios)
    return sorted(zip(lugares, filas), key=lambda x: x[0])


def niveles(n_vecinos: int) -> list:
    """(tormentas, transiciones por tormenta) de la fase 2 para una vecindad."""
    T = transiciones_max(n_vecinos)
    return [(400, 5), (400, min(48, T // 400)), (1000, min(48, T // 1000))]


def main():
    dev = torch.device("cuda")
    log("Inicio del barrido")
    run = Corrida(dev)

    f1 = [run.correr(1, v, red, e) for v in c.VECINDADES for red in REDES for e in ENTRADAS]
    orden = puntaje(f1)
    with open(CARPETA / "ranking_fase1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lugar_promedio", "vecindad", "red", "entrada", "area", "libre60_area", "libre120_area",
                    "libre120_proporcion", "perdida"])
        for p, r in orden:
            w.writerow([round(p, 2), r["vecindad"], r["red"], r["entrada"], r["area"], r["libre60_area"],
                        r["libre120_area"], r["libre120_proporcion"], r["perdida"]])
    finalistas = [r for _, r in orden[:FINALISTAS]]
    log("Fase 1 terminada; finalistas: " + ", ".join(f"{r['vecindad']} {r['red']} {r['entrada']}" for r in finalistas))

    f2 = []
    for r in finalistas:
        hecho = set()
        for t, k in niveles(int(r["n_vecinos"])):
            if (t, k) in hecho or k < 1:
                continue
            hecho.add((t, k))
            f2.append(run.correr(2, r["vecindad"], r["red"], r["entrada"], t, k, libre=True))
    mejor = puntaje(f2)[0][1]
    log(f"Fase 2 terminada; mejor: {mejor['vecindad']} {mejor['red']} {mejor['entrada']}, {mejor['tormentas']} "
        f"tormentas, {mejor['transiciones']} transiciones")
    (CARPETA / "mejor.json").write_text(json.dumps(mejor, indent=1, ensure_ascii=False))

    v, red, e = mejor["vecindad"], mejor["red"], mejor["entrada"]
    t, k = int(mejor["tormentas"]), int(mejor["transiciones"]) // int(mejor["tormentas"])
    for s in (1, 2):
        run.correr(3, v, red, e, t, k, semilla=s)
    run.correr(3, v, red, e, t, k, seg="4", libre=True)
    run.correr(3, v, red, e, t, k, epocas=EPOCAS_LARGO, libre=True)
    log("Fin del barrido")


if __name__ == "__main__":
    main()
