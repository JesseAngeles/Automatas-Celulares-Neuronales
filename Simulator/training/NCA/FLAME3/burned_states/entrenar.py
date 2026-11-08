"""NCA sobre CA/FLAME3/seleccion2/BurnedStates (12 estados de temperatura + 6 quemados, ver estados_quemado en notebooks/FLAME3.ipynb).

Validación cruzada: se entrena con un plot y se prueba en el otro (p3 → p2 y p2 → p3).
p2 = BurnedStates/flame_plot_1 (plot 1, IRX 662–777, 116 frames), p3 = BurnedStates/flame_plot_2 (plot 2, IRX 1189–1254,
66 frames).

Configuración: la mejor del barrido de 12 estados (NCA/FLAME3/historial/flame3_12estados/resumen.csv, menor media geométrica de
mae, rollout y llegada relativos a la copia): paso de 6 frames (30 s), vecindad 9×9, 2 capas de 32 ReLU, salida lineal
de 18 estados (CrossEntropy). Ahora con one-hot de 18 estados (81 × 18 = 1458 entradas) y frontera abierta. 200 épocas,
lr exp 1e-3 → 1e-5, lote 8192, semillas 42, 1, 2, 3, 4, scripts/main.py.

Representante de cada estado en °C: centro geométrico de su rango; el de un estado quemado es el del rango que une
(p. ej. 12: 25.3–72.9 °C). Fuego: representante ≥ 130 °C.

Métricas en el plot no visto (todas contra la copia, "el siguiente paso es igual"):
    un paso (las 6 secuencias con desfase o = 0…5):
        exact (18 estados), mae_c, quemado_acc (acierta si está quemada o no),
        nuevo_quemado_{recall,precision} (celdas sin quemar que pasan a quemadas),
        encendido / precision_encendido (celdas < 83 °C que llegan a ≥ 130 °C)
    simulación libre desde el frame 0 hasta el final del plot:
        rollout_mae_c (último paso), rollout_mae_c_medio (promedio de los pasos), rollout_quemado_acc e
        rollout_iou_quemado (bandera de quemado en el último paso), rollout_iou_fuego_total (alguna vez ≥ 130 °C)

Escribe redes/<ronda>/, resultados.csv, resumen.csv, gif/ y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/burned_states/entrenar.py
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
from PIL import Image, ImageDraw

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[2]
sys.path.insert(0, str(TRAINING / "scripts"))
from main import extract_patches                                   # noqa: E402
from barrido_flame import cargar_red                               # noqa: E402

PY = sys.executable
MAIN = TRAINING / "scripts" / "main.py"
DATOS = TRAINING / "CA" / "FLAME3" / "seleccion2" / "BurnedStates"
ESC = {"p2": "flame_plot_1", "p3": "flame_plot_2"}
RONDAS = [("p3", "p2"), ("p2", "p3")]
SEMILLAS = [42, 1, 2, 3, 4]
PASO, CAPAS = 6, (2, 32)
DISPERSA = "-5,-5 -1,-5 0,-5 1,-5 5,-5 -2,-2 -1,-2 0,-2 1,-2 2,-2 -5,-1 -2,-1 -1,-1 0,-1 1,-1 2,-1 5,-1 -5,0 -2,0 \
-1,0 0,0 1,0 2,0 5,0 -5,1 -2,1 -1,1 0,1 1,1 2,1 5,1 -2,2 -1,2 0,2 1,2 2,2 -5,5 -1,5 0,5 1,5 5,5"   # NCA/FLAME3/Personal/test_01
VECINDADES = {"9x9": 4, "13x13": 6, "dispersa41": None}
REDES = {"2x32": (2, 32), "3x64": (3, 64), "4x256": (4, 256)}
FRIO_C, FUEGO_C = 83.0, 130.0

_meta = json.loads((DATOS / "flame_plot_1" / "evo.json").read_text())
NE = _meta["k"]                                                    # 18
NT = _meta["burned_states"]["temperature_k"]                       # 12
BORDES = np.array(_meta["edges_celsius"])
GRUPOS = _meta["burned_states"]["groups"]
REP = np.array([np.sqrt(BORDES[s] * BORDES[s + 1]) for s in range(NT)]
               + [np.sqrt(BORDES[g[0]] * BORDES[g[-1] + 1]) for g in GRUPOS])
PALETA = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]
                   for h in (TRAINING.parent / _meta["palette"]).read_text().split()], dtype=np.uint8)


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


def entrenar(datos: list, base: Path, semilla: int, vecindad: str = "9x9", red: str = "2x32") -> dict:
    capas, neuronas = REDES[red]
    vec = ["--neighbourhood", DISPERSA] if VECINDADES[vecindad] is None else ["--radius", str(VECINDADES[vecindad])]
    cmd = [PY, "-u", str(MAIN), "--data", *map(str, datos), "--output", f"{base}.npz",
           *vec, "--layers", " ".join([f"{neuronas}:relu"] * capas),
           "--out-size", str(NE), "--out-act", "linear", "--one-hot", str(NE),
           "--epochs", "200", "--batch", "8192", "--lr", "1e-3", "--lr-schedule", "exp", "--lr-min", "1e-5",
           "--loss-target", "1e-9", "--print-every", "1", "--seed", str(semilla), "--open-boundary",
           "--log-csv", f"{base}_loss.csv"]
    t0 = time.perf_counter()
    salida = subprocess.run(cmd, capture_output=True, text=True, cwd=TRAINING)
    if salida.returncode != 0:
        raise RuntimeError("falló main.py: " + salida.stderr[-1500:])
    filas = list(csv.DictReader(open(f"{base}_loss.csv")))
    return {"epocas_hechas": int(filas[-1]["epoch"]) + 1, "loss_final": round(float(filas[-1]["loss"]), 6),
            "tiempo_s": round(time.perf_counter() - t0, 1)}


def predictor(ruta, dev):
    red, nb = cargar_red(str(ruta), dev)

    def pred(X: np.ndarray) -> np.ndarray:
        salidas = []
        with torch.no_grad():
            for s in range(0, len(X), 1 << 18):
                salidas.append(red(torch.from_numpy(X[s:s + (1 << 18)]).to(dev)).argmax(1))
        return torch.cat(salidas).cpu().numpy().astype(np.int32)
    return pred, nb


def paso_red(pred, nb, estado: np.ndarray) -> np.ndarray:
    X, _ = extract_patches(np.stack([estado, estado]).astype(np.float32), nb, True)
    return pred(X).reshape(estado.shape)


def iou(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def un_paso(p: np.ndarray, a: np.ndarray, s: np.ndarray, t: np.ndarray, t_ant: np.ndarray) -> dict:
    """p: predicho, a: actual, s: siguiente real (estados); t, t_ant: °C reales siguiente y actual."""
    frio = t_ant < FRIO_C
    enciende = frio & (t >= FUEGO_C)
    pf = frio & (REP[p] >= FUEGO_C)
    nuevo_r = (a < NT) & (s >= NT)
    nuevo_p = (a < NT) & (p >= NT)
    return {"exact": round(float((p == s).mean()), 5), "mae_c": round(float(np.abs(REP[p] - t).mean()), 4),
            "quemado_acc": round(float(((p >= NT) == (s >= NT)).mean()), 5),
            "nuevo_quemado_recall": round(float(nuevo_p[nuevo_r].mean()), 4) if nuevo_r.any() else np.nan,
            "nuevo_quemado_precision": round(float(nuevo_r[nuevo_p].mean()), 4) if nuevo_p.any() else np.nan,
            "encendido": round(float((REP[p][enciende] >= FUEGO_C).mean()), 4) if enciende.any() else np.nan,
            "precision_encendido": round(float((t[pf] >= FUEGO_C).mean()), 4) if pf.any() else np.nan}


def libre(Ts: list, S: np.ndarray, t: np.ndarray) -> dict:
    """Ts: estados simulados en los pasos 1…n; S, t: estados y °C reales en los pasos 0…n."""
    n = len(Ts)
    fuego_r = np.logical_or.reduce([t[k] >= FUEGO_C for k in range(n + 1)])
    fuego_p = np.logical_or.reduce([t[0] >= FUEGO_C] + [REP[T] >= FUEGO_C for T in Ts])
    return {"rollout_mae_c": round(float(np.abs(REP[Ts[-1]] - t[n]).mean()), 3),
            "rollout_mae_c_medio": round(float(np.mean([np.abs(REP[Ts[k]] - t[k + 1]).mean() for k in range(n)])), 3),
            "rollout_quemado_acc": round(float(((Ts[-1] >= NT) == (S[n] >= NT)).mean()), 4),
            "rollout_iou_quemado": round(iou(Ts[-1] >= NT, S[n] >= NT), 4),
            "rollout_iou_fuego_total": round(iou(fuego_p, fuego_r), 4)}


def evaluar(pred, nb, S: np.ndarray, T: np.ndarray, paso: int = PASO) -> tuple:
    centro = int(np.where((nb == 0).all(1))[0][0])
    P, A, Sg, Tg, Ta = [], [], [], [], []
    for o in range(paso):
        X, Y = extract_patches(S[o::paso].astype(np.float32), nb, True)
        P.append(pred(X))
        A.append(X[:, centro].astype(np.int32))
        Sg.append(Y.astype(np.int32))
        Tg.append(T[o::paso][1:].ravel())
        Ta.append(T[o::paso][:-1].ravel())
    P, A, Sg, Tg, Ta = map(np.concatenate, (P, A, Sg, Tg, Ta))
    f = un_paso(P, A, Sg, Tg, Ta)
    f.update({f"copia_{k}": v for k, v in un_paso(A, A, Sg, Tg, Ta).items()})
    s0, t0 = S[0::paso], T[0::paso]
    estado, Ts = s0[0], []
    for _ in range(len(s0) - 1):
        estado = paso_red(pred, nb, estado)
        Ts.append(estado)
    f.update(libre(Ts, s0, t0))
    f.update({f"copia_{k}": v for k, v in libre([s0[0]] * len(Ts), s0, t0).items()})
    return f, Ts


def gif(Ts: list, S: np.ndarray, destino: Path, titulo: str, zoom: int = 4, paso: int = PASO):
    s0 = S[0::paso]
    frames = []
    for k, real in enumerate(s0):
        sim = s0[0] if k == 0 else Ts[k - 1]
        a, b = (PALETA[x].repeat(zoom, 0).repeat(zoom, 1) for x in (real, sim))
        h, w = a.shape[:2]
        img = Image.new("RGB", (2 * w + 10, h + 18), "white")
        img.paste(Image.fromarray(a), (0, 18))
        img.paste(Image.fromarray(b), (w + 10, 18))
        d = ImageDraw.Draw(img)
        d.text((2, 3), f"real  t = {k * paso * 5} s", fill="black")
        d.text((w + 12, 3), f"NCA {titulo}", fill="black")
        frames.append(img)
    frames[0].save(destino, save_all=True, append_images=frames[1:], duration=max(100, 400 * paso // PASO), loop=0)


def main():
    dev = torch.device("cuda")
    S = {k: np.load(DATOS / v / "evo.npy").astype(np.int32) for k, v in ESC.items()}
    T = {k: np.load(DATOS / v / "evo_celsius.npy").astype(np.float32) for k, v in ESC.items()}
    datos = {k: sorted((DATOS / v / "entrenamiento").glob(f"k{PASO}_o*.npy")) for k, v in ESC.items()}
    log(f"Inicio: BurnedStates {NE} estados, paso {PASO}, 9x9, {CAPAS[0]}x{CAPAS[1]}, one-hot {NE}, frontera abierta; "
        + ", ".join(f"{k}: {S[k].shape}, estados {S[k].min()}–{S[k].max()}" for k in S))
    (CARPETA / "gif").mkdir(exist_ok=True)
    for entrena, prueba in RONDAS:
        ronda = f"{entrena}_a_{prueba}"
        (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
        for s in SEMILLAS:
            base = CARPETA / "redes" / ronda / f"k{PASO}_9x9_2x32_onehot_s{s}"
            info = entrenar(datos[entrena], base, s)
            pred, nb = predictor(f"{base}.npz", dev)
            f, Ts = evaluar(pred, nb, S[prueba], T[prueba])
            escribir({"ronda": ronda, "semilla": s, **info, **f})
            gif(Ts, S[prueba], CARPETA / "gif" / f"{ronda}_s{s}.gif", f"{entrena} -> {prueba}, semilla {s}")
            log(f"{ronda} s{s}: {info['tiempo_s']} s, loss {info['loss_final']}; exact {f['exact']} "
                f"(copia {f['copia_exact']}), mae {f['mae_c']} (copia {f['copia_mae_c']}), quemado_acc "
                f"{f['quemado_acc']} (copia {f['copia_quemado_acc']}), rollout {f['rollout_mae_c']} "
                f"(copia {f['copia_rollout_mae_c']})")
    filas = list(csv.DictReader(open(CARPETA / "resultados.csv")))
    metr = [k for k in filas[0] if k not in ("ronda", "semilla")]
    for ronda in dict.fromkeys(r["ronda"] for r in filas):
        g = [r for r in filas if r["ronda"] == ronda]
        fila = {"ronda": ronda, "n": len(g)}
        for k in metr:
            v = np.array([float(r[k]) for r in g])
            fila[k] = round(float(np.nanmean(v)), 4)
            fila[f"{k}_sd"] = round(float(np.nanstd(v)), 4)
        escribir(fila, "resumen.csv")
    log("Fin")


if __name__ == "__main__":
    main()
