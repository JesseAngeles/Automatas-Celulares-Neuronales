"""Funciones comunes para las NCA de CA/FLAME3/seleccion2 con 12 estados (estados 0–3 de log15 unidos).

Datos: CA/FLAME3/seleccion2/flame_plot_{1,2,3}/estados12/evo.npy (estados) y evo_celsius.npy (°C), 100×100, 5 s por frame.
    p1 = flame_plot_1 (plot 1, IRX 528–654, casi sin fuego), p2 = flame_plot_2 (plot 1, IRX 662–824),
    p3 = flame_plot_3 (plot 2, IRX 1189–1346).
Validación cruzada: p1 siempre se entrena (casi no tiene fuego); rondas {p1, p3} → p2 y {p1, p2} → p3.
Representante de cada estado: centro geométrico de su rango en °C (los cortes son iguales en log °C).
Fuego: estado ≥ 5 (≥ 137.6 °C).

Métricas (escenario no visto):
    un paso (todas las secuencias con desfase o = 0…paso−1):
        exact, mae_c, encendido (de las celdas < 83 °C que llegan a ≥ 130 °C, fracción que se predice ≥ 130 °C),
        precision_encendido, copia_* ("el siguiente paso es igual"), cuantizacion_mae_c
    simulación libre desde el primer frame durante 120 frames (10 min):
        rollout_mae_c, rollout_iou_quemado y las de propagación (solo cuenta el fuego nuevo):
        llegada_mae_min, llegada_sesgo_min, area_quemada_rel, iou_quemado_nuevo, fss_quemado_{1,3,9}, fss_fuego_3
    gif_*: lo mismo para la simulación libre desde el frame 26 de p2 hasta el final (la de los GIF), solo con p2.
Decodificadores: argmax (como el simulador) y sesgo (delta sumado a la salida de los estados de fuego, calibrado en
el escenario de entrenamiento para predecir tantos encendidos como los reales).
"""
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[3]
sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(TRAINING / "NCA" / "FLAME3" / "historial" / "cruzado"))
from main import extract_patches                                   # noqa: E402
from barrido_flame import cargar_red                               # noqa: E402

PY = sys.executable
MAIN = TRAINING / "scripts" / "main.py"
DATOS = TRAINING / "CA" / "FLAME3" / "seleccion2"
ESC = {"p1": "flame_plot_1", "p2": "flame_plot_2", "p3": "flame_plot_3"}
RONDAS = [(("p1", "p3"), "p2"), (("p1", "p2"), "p3")]
HORIZONTE = 120
FRIO_C, FUEGO_C = 83.0, 130.0
INICIO_GIF = 26

_meta = json.load(open(DATOS / "flame_plot_1" / "estados12" / "evo.json"))
NT = _meta["k"]
BORDES = np.array(_meta["edges_celsius"])
CORTES = BORDES[1:-1]
REP = np.sqrt(BORDES[:-1] * BORDES[1:])
FUEGO_ESTADO = int(np.searchsorted(BORDES, 135.0))          # 5: 137.6–170.0 °C
PALETA = TRAINING.parent / _meta["palette"]

DISPERSA = np.array([[int(v) for v in p.split(",")] for p in (
    "-5,-5 -1,-5 0,-5 1,-5 5,-5 -2,-2 -1,-2 0,-2 1,-2 2,-2 -5,-1 -2,-1 -1,-1 0,-1 1,-1 2,-1 5,-1 -5,0 -2,0 -1,0 "
    "0,0 1,0 2,0 5,0 -5,1 -2,1 -1,1 0,1 1,1 2,1 5,1 -2,2 -1,2 0,2 1,2 2,2 -5,5 -1,5 0,5 1,5 5,5").split()])
VECINDADES = {"9x9": 4, "13x13": 6, "dispersa41": None}
REDES = {"2x32": (2, 32), "3x64": (3, 64), "4x256": (4, 256)}


def ronda_nombre(entrena, prueba) -> str:
    return f"{'+'.join(entrena)}_a_{prueba}"


def estados_y_temps() -> tuple:
    S = {k: np.load(DATOS / v / "estados12" / "evo.npy").astype(np.int32) for k, v in ESC.items()}
    T = {k: np.load(DATOS / v / "estados12" / "evo_celsius.npy").astype(np.float32) for k, v in ESC.items()}
    for k in ESC:
        assert (S[k] == np.digitize(T[k], CORTES)).all(), f"{k}: los estados no coinciden con los cortes"
    return S, T


def secuencias(S: np.ndarray, paso: int) -> list:
    return [S[o::paso] for o in range(paso)]


# ── Registro ─────────────────────────────────────────────────────────────────

def log(carpeta: Path, msg: str):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(carpeta / "registro.log", "a") as f:
        f.write(linea + "\n")


def escribir(carpeta: Path, fila: dict, archivo: str = "resultados.csv"):
    """Agrega la fila; si trae columnas nuevas, reescribe el archivo con la unión de columnas."""
    destino = carpeta / archivo
    filas, campos = [], []
    if destino.exists():
        with open(destino) as f:
            lector = csv.DictReader(f)
            campos = list(lector.fieldnames or [])
            filas = list(lector)
    nuevos = [k for k in fila if k not in campos]
    if destino.exists() and not nuevos:
        with open(destino, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=campos).writerow(fila)
        return
    with open(destino, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos + nuevos)
        w.writeheader()
        w.writerows(filas + [fila])


def hecho(carpeta: Path, clave: dict, archivo: str = "resultados.csv") -> bool:
    destino = carpeta / archivo
    if not destino.exists():
        return False
    return any(all(str(r.get(k)) == str(v) for k, v in clave.items()) for r in csv.DictReader(open(destino)))


# ── Entrenamiento con scripts/main.py ────────────────────────────────────────

def entrenar(datos: list, base: Path, vecindad: str, red: str, semilla: int, epocas: int = 200) -> dict:
    capas, neuronas = REDES[red]
    cmd = [PY, "-u", str(MAIN), "--data", *map(str, datos), "--output", f"{base}.npz",
           "--layers", " ".join([f"{neuronas}:relu"] * capas), "--out-size", str(NT), "--out-act", "linear",
           "--epochs", str(epocas), "--batch", "8192", "--lr", "1e-3", "--lr-schedule", "exp", "--lr-min", "1e-5",
           "--loss-target", "1e-9", "--print-every", "1", "--seed", str(semilla), "--open-boundary",
           "--log-csv", f"{base}_loss.csv"]
    if VECINDADES[vecindad] is None:
        cmd += ["--neighbourhood", " ".join(f"{dx},{dy}" for dx, dy in DISPERSA)]
    else:
        cmd += ["--radius", str(VECINDADES[vecindad])]
    t0 = time.perf_counter()
    salida = subprocess.run(cmd, capture_output=True, text=True, cwd=TRAINING)
    if salida.returncode != 0:
        raise RuntimeError("falló main.py: " + salida.stderr[-1500:])
    filas = list(csv.DictReader(open(f"{base}_loss.csv")))
    return {"epocas_hechas": int(filas[-1]["epoch"]) + 1, "loss_final": round(float(filas[-1]["loss"]), 6),
            "tiempo_s": round(time.perf_counter() - t0, 1)}


# ── Predicción ───────────────────────────────────────────────────────────────

def predictor(ruta, dev):
    red, nb = cargar_red(str(ruta), dev)

    def prob(X: np.ndarray) -> torch.Tensor:
        salidas = []
        with torch.no_grad():
            for s in range(0, len(X), 1 << 18):
                p = torch.softmax(red(torch.from_numpy(X[s:s + (1 << 18)]).to(dev)), 1)
                if p.shape[1] < NT:
                    p = torch.nn.functional.pad(p, (0, NT - p.shape[1]))
                salidas.append(p)
        return torch.cat(salidas)
    return prob, nb


def con_sesgo(prob, delta: float):
    def p2(X):
        p = prob(X)
        f = torch.ones(p.shape[1], device=p.device)
        f[FUEGO_ESTADO:] = float(np.exp(delta))
        q = p * f
        return q / q.sum(1, keepdim=True)
    return p2


def calibrar_sesgo(prob, nb, secs: list, tsecs: list, lo: float = -3.0, hi: float = 12.0) -> dict:
    lp, reales = [], 0
    for s, t in zip(secs, tsecs):
        X, _ = extract_patches(s.astype(np.float32), nb, True)
        frio = (t[:-1] < FRIO_C).ravel()
        reales += int((frio & (t[1:].ravel() >= FUEGO_C)).sum())
        lp.append(torch.log(prob(X[frio]).clamp_min(1e-12)))
    lp = torch.cat(lp)
    fuego = torch.tensor(REP >= FUEGO_C, device=lp.device)
    m = torch.zeros(NT, device=lp.device)
    m[FUEGO_ESTADO:] = 1

    def n(d):
        return int(fuego[(lp + d * m).argmax(1)].sum())
    n0 = n(0.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if n(mid) < reales else (lo, mid)
    d = (lo + hi) / 2
    return {"delta": round(d, 4), "encendidos_reales_ent": reales, "predichos_ent_sin_sesgo": n0}


def guardar_con_sesgo(origen: Path, destino: Path, delta: float):
    z = dict(np.load(origen))
    ult = len(z["layer_out"]) - 1
    b = z[f"biases_{ult}"].copy()
    b[FUEGO_ESTADO:] += delta
    z[f"biases_{ult}"] = b
    np.savez(destino, **z)


def paso_red(prob, nb, estado: np.ndarray) -> np.ndarray:
    X, _ = extract_patches(np.stack([estado, estado]).astype(np.float32), nb, True)
    return prob(X).argmax(1).cpu().numpy().reshape(estado.shape).astype(np.int32)


def trayectoria(prob, nb, S0: np.ndarray, pasos: int) -> list:
    estado, Ts = S0.astype(np.int32), []
    for _ in range(pasos):
        estado = paso_red(prob, nb, estado)
        Ts.append(estado)
    return Ts


# ── Métricas ─────────────────────────────────────────────────────────────────

def iou(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def fss(pred, real, n: int) -> float:
    pf = uniform_filter(pred.astype(np.float64), size=n, mode="constant") if n > 1 else pred.astype(np.float64)
    po = uniform_filter(real.astype(np.float64), size=n, mode="constant") if n > 1 else real.astype(np.float64)
    ref = (pf ** 2).mean() + (po ** 2).mean()
    return float(1 - ((pf - po) ** 2).mean() / ref) if ref > 0 else 1.0


def llegada(mascaras: list) -> np.ndarray:
    inf = len(mascaras) + 1
    a = np.full(mascaras[0].shape, inf, dtype=np.int32)
    for k in range(len(mascaras), 0, -1):
        a[mascaras[k - 1]] = k
    return a


def propagacion(Ts: list, t0: np.ndarray, paso: int, prefijo: str = "") -> dict:
    """Ts: estados predichos en los pasos 1…n; t0: °C reales en los pasos 0…n."""
    n = len(Ts)
    real_f = [t0[k] >= BORDES[FUEGO_ESTADO] for k in range(n + 1)]
    pred_f = [T >= FUEGO_ESTADO for T in Ts]
    nuevas = ~real_f[0]
    inf = n + 1
    a_r, a_p = llegada(real_f[1:]), llegada(pred_f)
    qr, qp = nuevas & (a_r < inf), nuevas & (a_p < inf)
    ambas, alguna = qr & qp, qr | qp
    mn = paso * 5 / 60
    dif = (a_p - a_r).astype(np.float64)
    quemado_r = np.logical_or.reduce(real_f)
    quemado_p = np.logical_or.reduce([real_f[0]] + pred_f)
    f = {"rollout_mae_c": round(float(np.abs(REP[Ts[-1]] - t0[n]).mean()), 3),
         "rollout_iou_quemado": round(iou(quemado_p, quemado_r), 4),
         "llegada_mae_min": round(float(np.abs(dif[alguna]).mean() * mn), 4) if alguna.any() else np.nan,
         "llegada_sesgo_min": round(float(dif[ambas].mean() * mn), 4) if ambas.any() else np.nan,
         "area_quemada_rel": round(float(qp.sum() / qr.sum()), 4) if qr.any() else np.nan,
         "iou_quemado_nuevo": round(iou(qp, qr), 4),
         "fss_fuego_3": round(float(np.mean([fss(pred_f[k], real_f[k + 1], 3) for k in range(n)])), 4)}
    for w in (1, 3, 9):
        f[f"fss_quemado_{w}"] = round(fss(qp, qr, w), 4)
    return {prefijo + k: v for k, v in f.items()}


def evaluar(prob, nb, S: np.ndarray, t: np.ndarray, paso: int, clave: str) -> dict:
    centro = int(np.where((nb == 0).all(1))[0][0])
    pred, act, real, real_t = [], [], [], []
    for o in range(paso):
        X, _ = extract_patches(S[o::paso].astype(np.float32), nb, True)
        pred.append(prob(X).argmax(1).cpu().numpy())
        act.append(X[:, centro].astype(np.int64))
        real.append(t[o::paso][1:].ravel())
        real_t.append(t[o::paso][:-1].ravel())
    pred, act, real, real_t = map(np.concatenate, (pred, act, real, real_t))
    sig = np.digitize(real, CORTES)
    frio = real_t < FRIO_C
    enciende = frio & (real >= FUEGO_C)
    pf = frio & (REP[pred] >= FUEGO_C)
    f = {"exact": round(float((pred == sig).mean()), 5), "mae_c": round(float(np.abs(REP[pred] - real).mean()), 4),
         "encendido": round(float((REP[pred][enciende] >= FUEGO_C).mean()), 4) if enciende.any() else np.nan,
         "precision_encendido": round(float((real[pf] >= FUEGO_C).mean()), 4) if pf.any() else np.nan,
         "predichos_encendido": int(pf.sum()),
         "copia_exact": round(float((act == sig).mean()), 5),
         "copia_mae_c": round(float(np.abs(REP[act] - real).mean()), 4),
         "cuantizacion_mae_c": round(float(np.abs(REP[sig] - real).mean()), 4)}
    pasos = max(1, round(HORIZONTE / paso))
    t0 = t[0::paso][:pasos + 1]
    f.update(propagacion(trayectoria(prob, nb, S[0], pasos), t0, paso))
    f.update({f"copia_{k}": v for k, v in propagacion([S[0]] * pasos, t0, paso).items()})
    if clave == "p2":
        tg = t[INICIO_GIF::paso]
        pg = len(tg) - 1
        f.update(propagacion(trayectoria(prob, nb, S[INICIO_GIF], pg), tg, paso, "gif_"))
        f.update({f"copia_gif_{k}": v for k, v in propagacion([S[INICIO_GIF]] * pg, tg, paso).items()})
    return f
