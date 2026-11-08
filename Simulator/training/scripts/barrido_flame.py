"""Barrido de configuraciones de NCA para FLAME 3 (flame_plot_1, 15 estados log).

Entrena con CA/FLAME3/seleccion1/flame_plot_1/evo.npy y evalúa cada red sobre flame_plot_1, flame_plot_2 y
flame_plot_3 (misma escala de estados). Una fila por configuración en
NCA/FLAME3/historial/flame_plot_1/experimentos/resultados.csv; junto a ella los pesos (<nombre>.npz) y la
curva de pérdida (<nombre>_loss.csv). Las configuraciones que ya están en el CSV se omiten.

Métricas por escenario (un paso, sobre todas las celdas de todas las transiciones):
    loss        entropía cruzada media
    exact       fracción de celdas con el estado exacto
    exact_cambia  exactitud solo en las celdas cuyo estado cambia entre t y t+1
    encendido   fracción de celdas que se encienden (estado < 6 → ≥ 8) que la red predice ≥ 8
    mae         error absoluto medio en número de estado
    copia_*     las mismas métricas para "el siguiente frame es igual al actual"
Simulación libre de 20 pasos desde el primer frame (rollout):
    rollout20_mae  error absoluto medio en número de estado en el paso 20
    copia20_mae    lo mismo si nada cambia (frame 0 contra frame 20)

Uso (desde Simulator/training):
    python scripts/barrido_flame.py
"""
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from main import extract_patches

PY = "../GUI/venv/bin/python"
TRAIN = "CA/FLAME3/seleccion1/flame_plot_1/evo.npy"
EVAL = {f"p{i}": f"CA/FLAME3/seleccion1/flame_plot_{i}/evo.npy" for i in (1, 2, 3)}
CARPETA = Path("NCA/FLAME3/historial/flame_plot_1/experimentos")
ESTADOS = 15
FUEGO, FRIO = 8, 6          # encendido: estado < FRIO en t y ≥ FUEGO en t+1
PASOS_ROLLOUT = 20

BASE = dict(radio=3, entrada="onehot", capas=3, neuronas=128, lr=1e-3, schedule="plateau",
            lr_min=1e-5, patience=20, factor=0.5, epocas=400, lote=8192, frontera="abierta", semilla=42)


def configuraciones() -> list:
    confs = []
    # 1. Vecindad × codificación de la entrada
    for radio in (1, 2, 3, 4):
        for entrada in ("cruda", "onehot"):
            confs.append(("vecindad_entrada", dict(BASE, radio=radio, entrada=entrada)))
    # 2. Tipo de decaimiento del learning rate
    for schedule in ("exp", "cosine", "none"):
        confs.append(("schedule", dict(BASE, schedule=schedule)))
    # 3. Arquitectura
    for capas, neuronas in ((2, 64), (4, 256), (6, 128), (3, 512)):
        confs.append(("arquitectura", dict(BASE, capas=capas, neuronas=neuronas)))
    # 4. Learning rate inicial
    for lr in (3e-3, 3e-4):
        confs.append(("lr_inicial", dict(BASE, lr=lr)))
    # 5. Frontera, lote y semilla
    confs.append(("frontera", dict(BASE, frontera="periodica")))
    confs.append(("lote", dict(BASE, lote=2048)))
    for semilla in (1, 2):
        confs.append(("semilla", dict(BASE, semilla=semilla)))
    return confs


def nombre(c: dict) -> str:
    return (f"r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}_{c['schedule']}_lr{c['lr']:g}"
            f"_b{c['lote']}_{c['frontera']}_s{c['semilla']}")


def entrenar(c: dict, base: Path) -> dict:
    cmd = [PY, "-u", "scripts/main.py", "--data", TRAIN, "--output", f"{base}.npz",
           "--radius", str(c["radio"]),
           "--layers", " ".join([f"{c['neuronas']}:relu"] * c["capas"]),
           "--out-size", str(ESTADOS), "--out-act", "linear",
           "--epochs", str(c["epocas"]), "--batch", str(c["lote"]), "--lr", str(c["lr"]),
           "--lr-schedule", c["schedule"], "--lr-min", str(c["lr_min"]),
           "--plateau-patience", str(c["patience"]), "--plateau-factor", str(c["factor"]),
           "--loss-target", "1e-9", "--print-every", "1", "--seed", str(c["semilla"]),
           "--log-csv", f"{base}_loss.csv"]
    if c["entrada"] == "onehot":
        cmd += ["--one-hot", str(ESTADOS)]
    if c["frontera"] == "abierta":
        cmd += ["--open-boundary"]
    salida = subprocess.run(cmd, capture_output=True, text=True)
    if salida.returncode != 0:
        raise RuntimeError(salida.stderr[-2000:])
    tiempo = float(re.search(r"Tiempo de entrenamiento: ([\d.]+) s", salida.stdout).group(1))
    filas = list(csv.DictReader(open(f"{base}_loss.csv")))
    return {"epocas_hechas": int(filas[-1]["epoch"]) + 1, "loss_final": float(filas[-1]["loss"]),
            "lr_final": float(filas[-1]["lr"]), "tiempo_s": round(tiempo, 1)}


def cargar_red(path: str, dev) -> tuple:
    w = np.load(path)
    n = len(w["layer_out"])
    capas = [(torch.tensor(w[f"weights_{i}"], device=dev), torch.tensor(w[f"biases_{i}"], device=dev))
             for i in range(n)]
    acts = w["activations"].tolist()
    estados = int(w["one_hot_states"])

    def red(x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.one_hot(x.long(), estados).flatten(1).float() if estados else x
        for i, (W, b) in enumerate(capas):
            h = h @ W.T + b
            h = {0: torch.relu, 1: torch.sigmoid, 2: torch.tanh, 3: lambda v: v}[acts[i]](h)
        return h
    return red, w["neighbourhood"]


def predecir(red, X: np.ndarray, dev, lote: int = 1 << 18) -> torch.Tensor:
    salidas = []
    with torch.no_grad():
        for s in range(0, len(X), lote):
            salidas.append(red(torch.from_numpy(X[s:s + lote]).to(dev)))
    return torch.cat(salidas)


def metricas(logits: torch.Tensor, x0: np.ndarray, y: np.ndarray, dev) -> dict:
    y_t = torch.from_numpy(y).long().to(dev)
    x_t = torch.from_numpy(x0).long().to(dev)
    pred = logits.argmax(1)
    cambia = y_t != x_t
    enciende = (x_t < FRIO) & (y_t >= FUEGO)
    return {
        "loss": torch.nn.functional.cross_entropy(logits, y_t).item(),
        "exact": (pred == y_t).float().mean().item(),
        "exact_cambia": (pred[cambia] == y_t[cambia]).float().mean().item(),
        "encendido": (pred[enciende] >= FUEGO).float().mean().item() if enciende.any() else float("nan"),
        "mae": (pred - y_t).abs().float().mean().item(),
    }


def rollout(red, nb: np.ndarray, evo: np.ndarray, abierta: bool, dev) -> float:
    estado = evo[0].astype(np.float32)
    for _ in range(PASOS_ROLLOUT):
        X, _ = extract_patches(np.stack([estado, estado]), nb, abierta)
        estado = predecir(red, X, dev).argmax(1).cpu().numpy().reshape(estado.shape).astype(np.float32)
    return float(np.abs(estado - evo[PASOS_ROLLOUT]).mean())


def evaluar(c: dict, base: Path, evos: dict, dev) -> dict:
    red, nb = cargar_red(f"{base}.npz", dev)
    abierta = c["frontera"] == "abierta"
    centro = int(np.where((nb == 0).all(1))[0][0])
    fila = {}
    for clave, evo in evos.items():
        X, Y = extract_patches(evo.astype(np.float32), nb, abierta)
        x0 = X[:, centro].astype(np.int64)
        for k, v in metricas(predecir(red, X, dev), x0, Y.astype(np.int64), dev).items():
            fila[f"{k}_{clave}"] = round(v, 5)
        if clave in ("p1", "p3"):
            fila[f"rollout{PASOS_ROLLOUT}_mae_{clave}"] = round(rollout(red, nb, evo, abierta, dev), 4)
    return fila


def referencia(evos: dict) -> dict:
    """Métricas de copiar el frame anterior (no dependen de la red)."""
    fila = {}
    for clave, evo in evos.items():
        x, y = evo[:-1].ravel().astype(np.int64), evo[1:].ravel().astype(np.int64)
        enciende = (x < FRIO) & (y >= FUEGO)
        fila[f"copia_exact_{clave}"] = round(float((x == y).mean()), 5)
        fila[f"copia_encendido_{clave}"] = round(float((x[enciende] >= FUEGO).mean()), 5)
        fila[f"copia_mae_{clave}"] = round(float(np.abs(x - y).mean()), 5)
        if clave in ("p1", "p3"):
            fila[f"copia{PASOS_ROLLOUT}_mae_{clave}"] = round(float(np.abs(evo[0] - evo[PASOS_ROLLOUT]).mean()), 4)
    return fila


def main():
    CARPETA.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evos = {k: np.load(v) for k, v in EVAL.items()}
    ref = referencia(evos)
    destino = CARPETA / "resultados.csv"
    hechos = {r["nombre"] for r in csv.DictReader(open(destino))} if destino.exists() else set()

    confs = configuraciones()
    for i, (grupo, c) in enumerate(confs, 1):
        nom = nombre(c)
        if nom in hechos:
            print(f"[{i}/{len(confs)}] {nom}: ya está, se omite")
            continue
        hechos.add(nom)
        base = CARPETA / nom
        t0 = time.time()
        print(f"[{i}/{len(confs)}] {grupo}: {nom}", flush=True)
        fila = {"grupo": grupo, "nombre": nom, **c, "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
        fila.update(entrenar(c, base))
        fila.update(evaluar(c, base, evos, dev))
        fila.update(ref)
        nuevo = not destino.exists()
        with open(destino, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fila))
            if nuevo:
                w.writeheader()
            w.writerow(fila)
        print(f"    loss {fila['loss_final']:.4f}  exact_cambia p1 {fila['exact_cambia_p1']:.3f} "
              f"p3 {fila['exact_cambia_p3']:.3f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
