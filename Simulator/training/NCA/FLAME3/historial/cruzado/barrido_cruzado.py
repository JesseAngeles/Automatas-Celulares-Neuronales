"""Validación cruzada de NCA para FLAME 3: entrenar con dos escenarios y evaluar en el tercero.

Rondas: {p1, p2} → p3, {p1, p3} → p2, {p2, p3} → p1 (CA/FLAME3/seleccion1/flame_plot_N/evo.npy, 15 estados log).
En cada ronda se entrenan las mismas configuraciones con scripts/main.py (varios --data) y cada red se
evalúa en los tres escenarios: los dos de entrenamiento y el que no vio.

No modifica el simulador: solo llama a scripts/main.py y reutiliza funciones de scripts/barrido_flame.py.
Escribe en esta carpeta:
    resultados.csv            una fila por (ronda, configuración)
    registro.log              qué se está haciendo, con hora
    <ronda>/<nombre>.npz      pesos
    <ronda>/<nombre>_loss.csv curva de pérdida (epoch, loss, lr)

Métricas por escenario (mismo significado que en barrido_flame.py): loss, exact, exact_cambia,
encendido, mae, con sufijo _ent1 / _ent2 (escenarios de entrenamiento, en el orden de la columna
"entrena") y _prueba (el no visto); rollout20_mae en el no visto; copia_* = "el siguiente frame es igual".

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/cruzado/barrido_cruzado.py
"""
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TRAINING / "scripts"))
from main import extract_patches                                          # noqa: E402
from barrido_flame import cargar_red, predecir, metricas, FRIO, FUEGO      # noqa: E402

PY = str(TRAINING.parent / "GUI" / "venv" / "bin" / "python")
MAIN = str(TRAINING / "scripts" / "main.py")
CARPETA = Path(__file__).resolve().parent
ESC = {f"p{i}": TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{i}/evo.npy" for i in (1, 2, 3)}
RONDAS = [(("p1", "p2"), "p3"), (("p1", "p3"), "p2"), (("p2", "p3"), "p1")]
ESTADOS, PASOS_ROLLOUT = 15, 20

FIJO = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=300, lote=8192, frontera="abierta", semilla=42)


def configuraciones() -> list:
    confs = []
    for entrada in ("cruda", "onehot"):
        for radio in (1, 2, 3):
            for capas, neuronas in ((2, 32), (3, 128)):
                confs.append(dict(FIJO, entrada=entrada, radio=radio, capas=capas, neuronas=neuronas))
    confs.append(dict(FIJO, entrada="onehot", radio=3, capas=3, neuronas=512))   # contraste: red grande
    return confs


def nombre(c: dict) -> str:
    return f"r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}_{c['schedule']}_s{c['semilla']}"


def log(msg: str):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(CARPETA / "registro.log", "a") as f:
        f.write(linea + "\n")


def entrenar(c: dict, datos: list, base: Path) -> dict:
    cmd = [PY, "-u", MAIN, "--data", *map(str, datos), "--output", f"{base}.npz",
           "--radius", str(c["radio"]),
           "--layers", " ".join([f"{c['neuronas']}:relu"] * c["capas"]),
           "--out-size", str(ESTADOS), "--out-act", "linear",
           "--epochs", str(c["epocas"]), "--batch", str(c["lote"]), "--lr", str(c["lr"]),
           "--lr-schedule", c["schedule"], "--lr-min", str(c["lr_min"]),
           "--loss-target", "1e-9", "--print-every", "1", "--seed", str(c["semilla"]),
           "--log-csv", f"{base}_loss.csv"]
    if c["entrada"] == "onehot":
        cmd += ["--one-hot", str(ESTADOS)]
    if c["frontera"] == "abierta":
        cmd += ["--open-boundary"]
    log("    " + " ".join(cmd[1:]))
    salida = subprocess.run(cmd, capture_output=True, text=True, cwd=TRAINING)
    if salida.returncode != 0:
        log("    ERROR: " + salida.stderr[-1500:].replace("\n", " | "))
        raise RuntimeError("falló main.py")
    tiempo = float(re.search(r"Tiempo de entrenamiento: ([\d.]+) s", salida.stdout).group(1))
    filas = list(csv.DictReader(open(f"{base}_loss.csv")))
    return {"epocas_hechas": int(filas[-1]["epoch"]) + 1, "loss_final": float(filas[-1]["loss"]),
            "lr_final": float(filas[-1]["lr"]), "tiempo_s": round(tiempo, 1)}


def rollout(red, nb, evo, abierta, dev) -> float:
    estado = evo[0].astype(np.float32)
    for _ in range(PASOS_ROLLOUT):
        X, _ = extract_patches(np.stack([estado, estado]), nb, abierta)
        estado = predecir(red, X, dev).argmax(1).cpu().numpy().reshape(estado.shape).astype(np.float32)
    return float(np.abs(estado - evo[PASOS_ROLLOUT]).mean())


def evaluar(c: dict, base: Path, evos: dict, prueba: str, dev) -> dict:
    red, nb = cargar_red(f"{base}.npz", dev)
    abierta = c["frontera"] == "abierta"
    centro = int(np.where((nb == 0).all(1))[0][0])
    fila = {}
    entrenamiento = [k for k in evos if k != prueba]
    for clave, evo in evos.items():
        X, Y = extract_patches(evo.astype(np.float32), nb, abierta)
        m = metricas(predecir(red, X, dev), X[:, centro].astype(np.int64), Y.astype(np.int64), dev)
        # columnas iguales en todas las rondas: ent1 y ent2 son los escenarios de entrenamiento (en orden)
        etiqueta = "prueba" if clave == prueba else f"ent{entrenamiento.index(clave) + 1}"
        for k, v in m.items():
            fila[f"{k}_{etiqueta}"] = round(v, 5)
    fila["rollout20_mae_prueba"] = round(rollout(red, nb, evos[prueba], abierta, dev), 4)
    return fila


def copia(evo: np.ndarray) -> dict:
    x, y = evo[:-1].ravel().astype(np.int64), evo[1:].ravel().astype(np.int64)
    return {"copia_exact_prueba": round(float((x == y).mean()), 5),
            "copia_mae_prueba": round(float(np.abs(x - y).mean()), 5),
            "copia20_mae_prueba": round(float(np.abs(evo[0] - evo[PASOS_ROLLOUT]).mean()), 4)}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evos = {k: np.load(v) for k, v in ESC.items()}
    destino = CARPETA / "resultados.csv"
    hechos = {(r["ronda"], r["nombre"]) for r in csv.DictReader(open(destino))} if destino.exists() else set()
    confs = configuraciones()
    total = len(RONDAS) * len(confs)
    log(f"Inicio: {len(RONDAS)} rondas × {len(confs)} configuraciones = {total} entrenamientos ({dev})")

    i = 0
    for entrenamiento, prueba in RONDAS:
        ronda = f"{'+'.join(entrenamiento)}_a_{prueba}"
        (CARPETA / ronda).mkdir(exist_ok=True)
        log(f"Ronda {ronda}: entrena con {', '.join(entrenamiento)}, evalúa en {prueba}")
        for c in confs:
            i += 1
            nom = nombre(c)
            if (ronda, nom) in hechos:
                log(f"[{i}/{total}] {ronda} {nom}: ya está en resultados.csv, se omite")
                continue
            log(f"[{i}/{total}] {ronda} {nom}: entrenando")
            base = CARPETA / ronda / nom
            t0 = time.time()
            fila = {"ronda": ronda, "entrena": "+".join(entrenamiento), "prueba": prueba, "nombre": nom,
                    **c, "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
            fila.update(entrenar(c, [ESC[k] for k in entrenamiento], base))
            log(f"    entrenada: {fila['epocas_hechas']} épocas, loss {fila['loss_final']:.4f}, "
                f"{fila['tiempo_s']} s; evaluando")
            fila.update(evaluar(c, base, {k: evos[k] for k in entrenamiento + (prueba,)}, prueba, dev))
            fila.update(copia(evos[prueba]))
            nuevo = not destino.exists()
            with open(destino, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(fila))
                if nuevo:
                    w.writeheader()
                w.writerow(fila)
            log(f"    {prueba} (no vista): loss {fila['loss_prueba']:.4f}, exact {fila['exact_prueba']:.4f} "
                f"(copia {fila['copia_exact_prueba']:.4f}), exact_cambia {fila['exact_cambia_prueba']:.3f}, "
                f"encendido {fila['encendido_prueba']:.3f}, rollout20 {fila['rollout20_mae_prueba']:.3f} "
                f"(copia {fila['copia20_mae_prueba']:.3f})  [{time.time() - t0:.0f} s]")
    log("Fin")


if __name__ == "__main__":
    main()
