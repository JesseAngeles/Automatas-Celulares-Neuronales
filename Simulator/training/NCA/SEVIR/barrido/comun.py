"""Vecindades, datos, entrenamiento (scripts/main.py) y evaluación de NCA sobre SEVIR.

main.py arma todas las vecindades en RAM (float32, muestras × vecinos) y las sube a la GPU: con 40 tormentas completas
y 121 vecinos son 8.7 GB (18 GB de RAM como máximo). Para usar más tormentas sin pasar de ahí, datos.py toma solo
algunas transiciones de cada tormenta (archivos (N, 2, 96, 96): main.py las trata como N evoluciones de 2 imágenes).
"""
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[2]
sys.path.insert(0, str(TRAINING / "scripts"))
from barrido_flame import cargar_red                               # noqa: E402

PY = sys.executable
MAIN = TRAINING / "scripts" / "main.py"

CA = TRAINING / "CA" / "SEVIR"
TORMENTA = 2                       # estado ≥ 2 ⇔ VIL ≥ 74
MINUTOS = (30, 60, 120, 240)


# ── Vecindades (pares dx, dy como en main.py; todas ≤ 128 células, el límite del núcleo de la GUI) ───────────────

def cuadrada(r: int) -> list:
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


def circular(r: float) -> list:
    R = int(r)
    return [(dx, dy) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if dx * dx + dy * dy <= r * r]


def flame41() -> list:
    """Vecindad dispersa de 41 células de FLAME 3 (NCA/FLAME3/Personal/test_01.npz): 5×5 + 16 puntos lejanos."""
    w = np.load(TRAINING / "NCA" / "FLAME3" / "Personal" / "test_01.npz")
    return [tuple(int(v) for v in p) for p in w["neighbourhood"]]


def estrella(radios=(4, 6, 8, 11)) -> list:
    """Propuesta: núcleo 5×5 y 8 rayos (cada 45°) con puntos a 4, 6, 8 y 11 células (16–44 km).
    El núcleo ve el movimiento de un paso (≈ 1 célula / 5 min); los rayos ven la tormenta que se acerca desde lejos
    en cualquier dirección sin pagar las 23² células de un cuadrado de ese alcance."""
    nb = cuadrada(2)
    for r in radios:
        for k in range(8):
            a = k * np.pi / 4
            p = (int(round(r * np.cos(a))), int(round(r * np.sin(a))))
            if p not in nb:
                nb.append(p)
    return nb


def anillos(radios, puntos) -> list:
    """Dispersa circular: centro y anillos de radio r con n puntos repartidos por igual (los anillos impares girados
    medio paso para no alinear los puntos)."""
    nb = [(0, 0)]
    for i, (r, n) in enumerate(zip(radios, puntos)):
        giro = np.pi / n if i % 2 else 0.0
        for k in range(n):
            a = giro + 2 * np.pi * k / n
            p = (int(round(r * np.cos(a))), int(round(r * np.sin(a))))
            if p not in nb:
                nb.append(p)
    return nb


# Vecindades de anillos (anillos.py, después del barrido; no entran en la fase 1)
ANILLOS = {
    "anillos6": anillos((1, 2, 4, 6), (8, 12, 16, 20)),                     # 53, alcance 6
    "anillos11": anillos((1, 2, 4, 6, 8, 11), (8, 12, 16, 16, 16, 16)),     # 81, alcance 11
}

VECINDADES = {
    "5x5": cuadrada(2), "9x9": cuadrada(4), "11x11": cuadrada(5),
    "circ3": circular(3), "circ5": circular(5), "circ6": circular(6),
    "flame41": flame41(), "estrella57": estrella(),
}

# 4 estados: une los 7 de SEVIR en sin lluvia / lluvia / tormenta (74–160) / tormenta intensa (≥ 160)
SEGMENTACIONES = {"7": None, "4": np.array([0, 1, 2, 2, 3, 3, 3], dtype=np.int8)}


def cargar(nombres, n=None, segmentacion="7") -> np.ndarray:
    """Une CA/SEVIR/<nombre>/evo.npy (N, 49, 96, 96) int8; n = primeras n tormentas del total."""
    S = np.concatenate([np.load(CA / x / "evo.npy") for x in nombres])
    if n:
        S = S[:n]
    m = SEGMENTACIONES[segmentacion]
    return m[S] if m is not None else S


def entrenar(datos: list, nb, capas, estados: int, one_hot: bool, epocas: int, salida: Path, semilla=42) -> dict:
    """Entrena con scripts/main.py (frontera abierta, salida lineal, CrossEntropy, lr exp 1e-3 → 1e-5)."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", str(MAIN), "--data", *map(str, datos), "--output", str(salida),
           "--neighbourhood", " ".join(f"{dx},{dy}" for dx, dy in nb),
           "--layers", " ".join(f"{c}:relu" for c in capas), "--out-size", str(estados), "--out-act", "linear",
           "--epochs", str(epocas), "--batch", "8192", "--lr", "1e-3", "--lr-schedule", "exp", "--lr-min", "1e-5",
           "--loss-target", "1e-9", "--print-every", "1", "--seed", str(semilla), "--open-boundary",
           "--log-csv", str(salida.with_name(salida.stem + "_loss.csv"))]
    if one_hot:
        cmd += ["--one-hot", str(estados)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=TRAINING)
    if r.returncode != 0:
        raise RuntimeError("falló main.py: " + r.stderr[-1500:])
    muestras = int(re.search(r"Muestras: ([\d,]+)", r.stdout).group(1).replace(",", ""))
    filas = list(csv.DictReader(open(salida.with_name(salida.stem + "_loss.csv"))))
    return {"perdida": round(float(filas[-1]["loss"]), 5), "epocas": int(filas[-1]["epoch"]) + 1,
            "muestras": muestras, "tiempo_s": round(time.perf_counter() - t0, 1)}


def modelo(ruta: Path, dev):
    """Red guardada por main.py, como la usa la GUI (barrido_flame.cargar_red)."""
    return cargar_red(str(ruta), dev)[0]


# ── Evaluación (frontera abierta, argmax, igual que el núcleo de la GUI) ────────────────────────────────────────

@torch.no_grad()
def paso_red(modelo, nb, dev):
    off = torch.tensor(nb, device=dev)
    r = int(off.abs().max())

    def paso(E: torch.Tensor) -> torch.Tensor:
        """E (B, H, W) int8 en la GPU → siguiente paso."""
        B, H, W = E.shape
        Pd = torch.nn.functional.pad(E, (r, r, r, r))
        X = torch.stack([Pd[:, r + dy:r + dy + H, r + dx:r + dx + W] for dx, dy in nb.tolist()], -1)
        X = X.reshape(-1, len(nb)).float()
        out = torch.cat([modelo(X[s:s + (1 << 19)]).argmax(1) for s in range(0, len(X), 1 << 19)])
        return out.to(torch.int8).view(B, H, W)
    return paso


def coincidencia(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a >= TORMENTA, b >= TORMENTA
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else 1.0


@torch.no_grad()
def evaluar(modelo, nb, S: np.ndarray, dev, guardar_libre: Path = None) -> dict:
    """S (N, T, H, W) de prueba. Un paso desde cada imagen real y simulación libre desde la imagen 0, contra la
    predicción sin cambio (copia)."""
    paso = paso_red(modelo, np.array(nb), dev)
    St = torch.from_numpy(S).to(dev)
    N, T = S.shape[:2]
    bien = lluvia_bien = lluvia_tot = 0
    inter = union = c_inter = c_union = 0
    c_bien = c_lluvia_bien = c_lluvia_tot = 0
    for t in range(T - 1):
        P, R, C = paso(St[:, t]), St[:, t + 1], St[:, t]
        bien += (P == R).sum().item()
        c_bien += (C == R).sum().item()
        ll = (P >= 1) | (R >= 1)
        lluvia_bien += ((P == R) & ll).sum().item()
        lluvia_tot += ll.sum().item()
        llc = (C >= 1) | (R >= 1)
        c_lluvia_bien += ((C == R) & llc).sum().item()
        c_lluvia_tot += llc.sum().item()
        a, b, c = P >= TORMENTA, R >= TORMENTA, C >= TORMENTA
        inter += (a & b).sum().item(); union += (a | b).sum().item()
        c_inter += (c & b).sum().item(); c_union += (c | b).sum().item()
    total = N * (T - 1) * S.shape[2] * S.shape[3]
    f = {"exactitud": bien / total, "exactitud_lluvia": lluvia_bien / lluvia_tot, "area": inter / union,
         "copia_exactitud": c_bien / total, "copia_exactitud_lluvia": c_lluvia_bien / c_lluvia_tot,
         "copia_area": c_inter / c_union}
    E, L = St[:, 0], [St[:, 0].cpu()]
    area_libre = [(E >= TORMENTA).float().mean().item()]
    for t in range(1, T):
        E = paso(E)
        L.append(E.cpu())
        area_libre.append((E >= TORMENTA).float().mean().item())
    L = torch.stack(L, 1)
    area_real = (St >= TORMENTA).float().mean((0, 2, 3)).tolist()
    for m in MINUTOS:
        k = m // 5
        f[f"libre{m}_exactitud"] = float((L[:, k].numpy() == S[:, k]).mean())
        f[f"libre{m}_area"] = coincidencia(L[:, k], torch.from_numpy(S[:, k]))
        f[f"copia_libre{m}_exactitud"] = float((S[:, 0] == S[:, k]).mean())
        f[f"copia_libre{m}_area"] = coincidencia(torch.from_numpy(S[:, 0]), torch.from_numpy(S[:, k]))
        # proporción del área con tormenta simulada respecto a la real (1 = misma cantidad de tormenta)
        f[f"libre{m}_proporcion"] = area_libre[k] / area_real[k]
    if guardar_libre is not None:
        guardar_libre.parent.mkdir(parents=True, exist_ok=True)
        np.save(guardar_libre, L.numpy())
    return {k: round(v, 5) for k, v in f.items()}, area_libre, area_real
