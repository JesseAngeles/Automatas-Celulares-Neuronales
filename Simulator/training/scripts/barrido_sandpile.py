"""Barrido de activaciones para sandpile (sección 1.3 del experimento 1).

Red 16:16 (16 capas de 16 neuronas) con activación oculta ReLU, Sigmoid o TanH y capa de salida de 17 neuronas
con Linear, Sigmoid o TanH: 9 configuraciones. Vecindad de Von Neumann y frontera abierta (--open-boundary),
igual que la simulación de notebooks/sandpile.nb.

Uso (desde Simulator/training):
    python scripts/barrido_sandpile.py prueba        (ajuste corto de lr y objetivo de pérdida)
    python scripts/barrido_sandpile.py entrenar      (las 9 configuraciones → NCA/Sandpile/resultados.csv)
"""
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from main import extract_patches

PY = "../GUI/venv/bin/python"
DATA = "CA/Sandpile/evo.mat"
CARPETA = Path("NCA/Sandpile")
VECINDAD = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
CAPAS, NEURONAS, SALIDA = 16, 16, 17
GENERADOR = 8                  # estados >= 8 son generadores

# Hiperparámetros elegidos con la prueba corta (ver NCA/Sandpile/ajuste/README.md)
LR, LOTE, EPOCAS, LOSS_TARGET, SEED, HILOS = 0.0003, 4096, 500, 0.001, 42, 5
CADA = 50                      # cada cuántas épocas se reporta la pérdida en resultados.csv

ACTIVACIONES = ["relu", "sigmoid", "tanh"]
SALIDAS = ["linear", "sigmoid", "tanh"]
CONFIGS = [(a, s) for a in ACTIVACIONES for s in SALIDAS]
ACT_TXT = {"relu": "ReLU", "tanh": "TanH", "sigmoid": "Sigmoid", "linear": "Linear"}
ACT_TORCH = {0: torch.relu, 1: torch.sigmoid, 2: torch.tanh, 3: lambda x: x}


def entrada(w, x: torch.Tensor) -> torch.Tensor:
    """Entrada de la primera capa: los estados tal cual o en one-hot si la red se entrenó con --one-hot."""
    estados = int(w["one_hot_states"]) if "one_hot_states" in w else 0
    return torch.nn.functional.one_hot(x.long(), estados).flatten(1).float() if estados else x


def nombre(act: str, out_act: str) -> str:
    return f"{CAPAS}-{NEURONAS}-{act}_{SALIDA}-{out_act}"


def cargar_muestras():
    import scipy.io
    datos = scipy.io.loadmat(DATA)["evolution"].astype(np.float32)
    X, Y = extract_patches(datos, np.array(VECINDAD), open_boundary=True)
    patrones, inv = np.unique(X, axis=0, return_inverse=True)
    return patrones, inv.ravel(), Y.astype(np.int64)


def exactitud(pesos: str, muestras) -> dict:
    """Fracción de celdas cuyo estado siguiente predice bien la red (argmax), en total y en generadores."""
    patrones, inv, Y = muestras
    w = np.load(pesos)
    x = entrada(w, torch.from_numpy(patrones))
    with torch.no_grad():
        for i, act in enumerate(w["activations"]):
            x = x @ torch.from_numpy(w[f"weights_{i}"]).T + torch.from_numpy(w[f"biases_{i}"])
            x = ACT_TORCH[int(act)](x)
    pred = x.argmax(1).numpy()[inv]
    bien = pred == Y
    gen = Y >= GENERADOR
    return dict(exactitud=bien.mean(), exactitud_generador=bien[gen].mean(),
                patrones_mal=len(np.unique(inv[~bien])))


def entrenar(job: dict) -> dict:
    out = Path(job["salida"])
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".csv")
    cmd = [PY, "-W", "ignore", "scripts/main.py", "--data", DATA, "--open-boundary",
           "--neighbourhood", " ".join(f"{dx},{dy}" for dx, dy in VECINDAD),
           "--layers", " ".join([f"{NEURONAS}:{job['act']}"] * CAPAS),
           "--out-size", str(SALIDA), "--out-act", job["out_act"],
           "--epochs", str(job["epocas"]), "--batch", str(job["lote"]), "--lr", str(job["lr"]),
           "--loss-target", str(job["loss_target"]), "--seed", str(SEED), "--print-every", "1",
           "--output", str(out), "--log-csv", str(log)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    res = dict(job, segundos=time.perf_counter() - t0, error="")
    if r.returncode != 0 or not out.exists():
        res["error"] = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"código {r.returncode}"
        print(f"ERROR {out}: {res['error']}", flush=True)
        return res
    filas_log = list(csv.DictReader(open(log)))
    perdidas = {int(f["epoch"]): float(f["loss"]) for f in filas_log}
    # main.py repite la última época al converger: las épocas corridas son la última + 1
    res.update(final=float(filas_log[-1]["loss"]), epocas_corridas=int(filas_log[-1]["epoch"]) + 1,
               perdidas=perdidas)
    print(f"{out}  final={res['final']:.6f}  épocas={res['epocas_corridas']}  {res['segundos']:.0f}s", flush=True)
    return res


def evaluar(resultados: list) -> None:
    muestras = cargar_muestras()
    for r in resultados:
        if not r["error"]:
            r.update(exactitud(r["salida"], muestras))


def cmd_prueba():
    """lr con 100 épocas sin objetivo de pérdida; después objetivo de pérdida con el lr elegido."""
    carpeta = CARPETA / "ajuste"
    jobs = [dict(act=a, out_act=s, lr=lr, lote=LOTE, epocas=100, loss_target=0.0,
                 salida=str(carpeta / f"lr{lr}" / f"{nombre(a, s)}.npz"))
            for lr in (0.003, 0.001, 0.0003, 0.0001) for a, s in CONFIGS]
    with ThreadPoolExecutor(HILOS) as ex:
        resultados = list(ex.map(entrenar, jobs))
    evaluar(resultados)
    campos = ["act", "out_act", "lr", "lote", "epocas", "epocas_corridas", "final", "exactitud",
              "exactitud_generador", "patrones_mal", "segundos", "salida", "error"]
    with open(carpeta / "ajuste_lr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(resultados)
    print(f"Resumen → {carpeta / 'ajuste_lr.csv'}", flush=True)


def cmd_entrenar():
    jobs = [dict(act=a, out_act=s, lr=LR, lote=LOTE, epocas=EPOCAS, loss_target=LOSS_TARGET,
                 salida=str(CARPETA / f"{nombre(a, s)}.npz")) for a, s in CONFIGS]
    with ThreadPoolExecutor(HILOS) as ex:
        resultados = list(ex.map(entrenar, jobs))
    evaluar(resultados)
    marcas = list(range(0, EPOCAS, CADA)) + [EPOCAS - 1]
    campos = (["capa_oculta", "capa_salida", "epocas", "perdida_final", "exactitud", "exactitud_generador"]
              + [f"perdida_{e}" for e in marcas])
    with open(CARPETA / "resultados.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(campos)
        for r in resultados:
            if r["error"]:
                w.writerow([f"{CAPAS}:{NEURONAS}:{ACT_TXT[r['act']]}", f"{SALIDA}:{ACT_TXT[r['out_act']]}", "error"])
                continue
            w.writerow([f"{CAPAS}:{NEURONAS}:{ACT_TXT[r['act']]}", f"{SALIDA}:{ACT_TXT[r['out_act']]}",
                        r["epocas_corridas"], r["final"], r["exactitud"], r["exactitud_generador"]]
                       + [r["perdidas"].get(e, "") for e in marcas])
    print(f"Resumen → {CARPETA / 'resultados.csv'}", flush=True)


if __name__ == "__main__":
    {"prueba": cmd_prueba, "entrenar": cmd_entrenar}[sys.argv[1]]()
