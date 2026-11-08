"""NCA sobre tormentas de SEVIR (VIL en 7 estados, ver la sección 2 de notebooks/SEVIR.ipynb).

Validación cruzada por año: se entrena con 40 tormentas de 2018 (CA/SEVIR/entrenamiento) y se prueba con 20 de 2019
(CA/SEVIR/prueba). Células de 4 km, paso de 5 min, 49 frames por tormenta (4 h).

Configuraciones: vecindad 5×5 y 9×9, capas ocultas 2:32 y 3:64 ReLU, salida lineal de 7 estados (CrossEntropy),
one-hot de 7 estados, frontera abierta, 30 épocas, lote 8192, lr exp 1e-3 → 1e-5, semilla 42, scripts/main.py.

Métricas en las tormentas de prueba (todas contra la copia, "el siguiente paso es igual al actual"):
    un paso: exactitud (estado correcto), exactitud en células con lluvia (real o predicha ≥ 1),
             coincidencia del área con tormenta (estado ≥ 2, VIL ≥ 74) entre la predicción y el real
    simulación libre desde el frame 0: exactitud y coincidencia del área con tormenta a los 30, 60 y 120 min,
             proporción del área con tormenta en cada paso (para la gráfica)

Coincidencia del área = |predicha ∩ real| / |predicha ∪ real|.

Escribe redes/, resultados.csv, area_libre.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/SEVIR/entrenar.py
"""
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[1]
sys.path.insert(0, str(TRAINING / "scripts"))
from main import extract_patches                                   # noqa: E402
from barrido_flame import cargar_red                               # noqa: E402

PY = sys.executable
MAIN = TRAINING / "scripts" / "main.py"
ENTRENA = TRAINING / "CA" / "SEVIR" / "entrenamiento" / "evo.npy"
PRUEBA = TRAINING / "CA" / "SEVIR" / "prueba" / "evo.npy"
NE = 7
TORMENTA = 2
VECINDADES = {"5x5": 2, "9x9": 4}
REDES = {"2x32": (2, 32), "3x64": (3, 64)}
EPOCAS = 30
SEMILLA = 42
MINUTOS = (30, 60, 120)


def log(msg: str):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(CARPETA / "registro.log", "a") as f:
        f.write(linea + "\n")


def escribir(fila: dict, archivo: str = "resultados.csv"):
    destino = CARPETA / archivo
    nuevo = not destino.exists()
    with open(destino, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def entrenar(base: Path, radio: int, capas: int, neuronas: int) -> dict:
    cmd = [PY, "-u", str(MAIN), "--data", str(ENTRENA), "--output", f"{base}.npz",
           "--radius", str(radio), "--layers", " ".join([f"{neuronas}:relu"] * capas),
           "--out-size", str(NE), "--out-act", "linear", "--one-hot", str(NE),
           "--epochs", str(EPOCAS), "--batch", "8192", "--lr", "1e-3", "--lr-schedule", "exp", "--lr-min", "1e-5",
           "--loss-target", "1e-9", "--print-every", "1", "--seed", str(SEMILLA), "--open-boundary",
           "--log-csv", f"{base}_loss.csv"]
    t0 = time.perf_counter()
    salida = subprocess.run(cmd, capture_output=True, text=True, cwd=TRAINING)
    if salida.returncode != 0:
        raise RuntimeError("falló main.py: " + salida.stderr[-1500:])
    filas = list(csv.DictReader(open(f"{base}_loss.csv")))
    return {"epocas": int(filas[-1]["epoch"]) + 1, "perdida": round(float(filas[-1]["loss"]), 5),
            "tiempo_s": round(time.perf_counter() - t0, 1)}


def predictor(ruta, dev):
    red, nb = cargar_red(str(ruta), dev)

    def paso(estados: np.ndarray) -> np.ndarray:
        """estados (B, H, W) → siguiente paso de cada uno."""
        B, H, W = estados.shape
        X = np.concatenate([extract_patches(np.stack([e, e]).astype(np.float32), nb, True)[0] for e in estados])
        salidas = []
        with torch.no_grad():
            for s in range(0, len(X), 1 << 18):
                salidas.append(red(torch.from_numpy(X[s:s + (1 << 18)]).to(dev)).argmax(1))
        return torch.cat(salidas).cpu().numpy().astype(np.int8).reshape(B, H, W)
    return paso


def coincidencia(p: np.ndarray, r: np.ndarray) -> float:
    a, b = p >= TORMENTA, r >= TORMENTA
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def un_paso(p: np.ndarray, r: np.ndarray) -> dict:
    lluvia = (p >= 1) | (r >= 1)
    return {"exactitud": round(float((p == r).mean()), 5),
            "exactitud_lluvia": round(float((p == r)[lluvia].mean()), 5),
            "area_tormenta": round(coincidencia(p, r), 5)}


def evaluar(paso, S: np.ndarray) -> tuple:
    """S (N, T, H, W). Un paso desde cada frame real y simulación libre desde el frame 0."""
    N, T = S.shape[:2]
    P = np.stack([paso(S[:, t]) for t in range(T - 1)], 1)          # P[:, t] predice S[:, t + 1]
    f = un_paso(P, S[:, 1:])
    f.update({f"copia_{k}": v for k, v in un_paso(S[:, :-1], S[:, 1:]).items()})
    estado, L = S[:, 0], [S[:, 0]]
    for _ in range(T - 1):
        estado = paso(estado)
        L.append(estado)
    L = np.stack(L, 1)
    for m in MINUTOS:
        k = m // 5
        f[f"libre_{m}_exactitud"] = round(float((L[:, k] == S[:, k]).mean()), 5)
        f[f"libre_{m}_area_tormenta"] = round(coincidencia(L[:, k], S[:, k]), 5)
        f[f"copia_libre_{m}_exactitud"] = round(float((S[:, 0] == S[:, k]).mean()), 5)
        f[f"copia_libre_{m}_area_tormenta"] = round(coincidencia(S[:, 0], S[:, k]), 5)
    return f, L


def main():
    dev = torch.device("cuda")
    S = np.load(PRUEBA).astype(np.int8)
    log(f"Inicio: entrenamiento {np.load(ENTRENA, mmap_mode='r').shape}, prueba {S.shape}, {NE} estados, "
        f"{EPOCAS} épocas, semilla {SEMILLA}")
    (CARPETA / "redes").mkdir(exist_ok=True)
    (CARPETA / "libre").mkdir(exist_ok=True)
    np.save(CARPETA / "libre" / "real.npy", S)
    area = {"minuto": [5 * t for t in range(S.shape[1])],
            "real": [round(float((S[:, t] >= TORMENTA).mean()), 5) for t in range(S.shape[1])],
            "copia": [round(float((S[:, 0] >= TORMENTA).mean()), 5)] * S.shape[1]}
    for vec, radio in VECINDADES.items():
        for nombre, (capas, neuronas) in REDES.items():
            clave = f"{vec}_{nombre}"
            base = CARPETA / "redes" / clave
            info = entrenar(base, radio, capas, neuronas)
            f, L = evaluar(predictor(f"{base}.npz", dev), S)
            np.save(CARPETA / "libre" / f"{clave}.npy", L)
            area[clave] = [round(float((L[:, t] >= TORMENTA).mean()), 5) for t in range(L.shape[1])]
            escribir({"vecindad": vec, "red": nombre, **info, **f})
            log(f"{clave}: {info['tiempo_s']} s, pérdida {info['perdida']}; exactitud {f['exactitud']} "
                f"(copia {f['copia_exactitud']}), área {f['area_tormenta']} (copia {f['copia_area_tormenta']}); "
                f"libre 60 min área {f['libre_60_area_tormenta']} (copia {f['copia_libre_60_area_tormenta']})")
    with open(CARPETA / "area_libre.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(area))
        w.writerows(zip(*area.values()))
    log("Fin")


if __name__ == "__main__":
    main()
