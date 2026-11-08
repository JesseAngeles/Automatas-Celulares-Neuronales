"""Funciones comunes para probar, por separado, las opciones de mejora de la NCA de FLAME 3.

Referencia (opción 0): ../sin_p2 — log15 (escala 24.7–604.6 °C), p1 → p3 y p3 → p1 sin ignición, vecindad 9×9,
paso de 6 frames (30 s), entrada cruda, 2 capas de 32 ReLU, 200 épocas, exp 1e-3 → 1e-5, frontera abierta.
Cada opción cambia una sola cosa respecto a la referencia y escribe en su carpeta resultados.csv y registro.log.

Evaluación común (una fila por red o conjunto de redes y escenario):
  un paso, sobre todas las secuencias con desfase o = 0…paso−1:
    exact_temp            fracción de celdas con el estado de temperatura exacto
    mae_c                 °C: representante del estado predicho vs °C real
    encendido             de las celdas < 83 °C en t que llegan a ≥ 130 °C en t+1, fracción que la red predice ≥ 130 °C
    precision_encendido   de las celdas < 83 °C en t que la red predice ≥ 130 °C, fracción que de verdad llega
    predichos_encendido   cuántas celdas predice que se encienden
    copia_*               lo mismo para "el siguiente paso es igual"
  simulación libre desde el primer frame (desfase 0) durante HORIZONTE = 120 frames (10 min) o lo más cercano:
    rollout_mae_c         °C al final de la simulación
    rollout_iou_fuego     intersección / unión de las celdas ≥ 135.9 °C predichas y reales, promedio de todos los pasos
    rollout_iou_quemado   intersección / unión de las celdas que alguna vez llegaron a ≥ 135.9 °C (predichas y reales)
                          desde el primer frame hasta el final
    copia_rollout_*       lo mismo si nada cambia (el primer frame se queda igual)
Decodificadores: argmax (el estado más probable, como la NCA del simulador), muestreo (se sortea el estado con las
probabilidades de la red) y esperado (temperatura esperada Σ p · representante, discretizada de nuevo).
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

OPC = Path(__file__).resolve().parent
FLAME = OPC.parent
TRAINING = OPC.parents[3]
sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(FLAME / "cruzado"))
from main import extract_patches, build_model, save_weights      # noqa: E402,F401
from barrido_flame import cargar_red                              # noqa: E402
import barrido_cruzado as bc                                       # noqa: E402

ESC = ["p1", "p3"]
RONDAS = [(("p1",), "p3"), (("p3",), "p1")]
INICIO = {"p1": 133, "p3": 19}
SEMILLAS = [42, 1, 2, 3, 4]
FRIO_C, FUEGO_C = 83.0, 130.0
HORIZONTE = 120
PASO = 6
BASE = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=200, lote=8192, frontera="abierta",
            entrada="cruda", radio=4, capas=2, neuronas=32, semilla=42)

_d = json.load(open(FLAME / "sin_p2" / "datos" / "log15" / "discretizacion.json"))
CORTES = np.array(_d["cortes_c"])
REP = np.array(_d["representantes_c"])
NT = len(REP)
UMBRAL_FUEGO = float(CORTES[7])                    # 135.9 °C: inicio del estado 8
REP_T = torch.tensor(REP, dtype=torch.float32)


def ronda_nombre(entrena, prueba) -> str:
    return f"{'+'.join(entrena)}_a_{prueba}"


def red_referencia(ronda: str, semilla: int, carpeta: str = "sin_p2", prefijo: str = "log15") -> Path:
    return FLAME / carpeta / "semillas" / ronda / f"{prefijo}_k6_r4_cruda_2x32_exp_s{semilla}.npz"


def temperaturas() -> dict:
    return {k: np.load(TRAINING / f"CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{k[1]}/evo_celsius.npy") for k in ESC}


def estados(t: np.ndarray) -> np.ndarray:
    return np.digitize(t, CORTES).astype(np.int32)


# ── Registro ─────────────────────────────────────────────────────────────────

def log(carpeta: Path, msg: str):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(carpeta / "registro.log", "a") as f:
        f.write(linea + "\n")


def escribir(carpeta: Path, fila: dict, archivo: str = "resultados.csv"):
    destino = carpeta / archivo
    nuevo = not destino.exists()
    with open(destino, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def hecho(carpeta: Path, clave: dict, archivo: str = "resultados.csv") -> bool:
    destino = carpeta / archivo
    if not destino.exists():
        return False
    return any(all(str(r.get(k)) == str(v) for k, v in clave.items()) for r in csv.DictReader(open(destino)))


def resumen(fila: dict) -> str:
    return (f"exact_temp {fila['exact_temp']:.4f} (copia {fila['copia_exact_temp']:.4f}), mae {fila['mae_c']:.2f} °C "
            f"(copia {fila['copia_mae_c']:.2f}), encendido {fila['encendido']:.3f} (precisión "
            f"{fila['precision_encendido']}), rollout {fila['rollout_mae_c']:.2f} °C (copia {fila['copia_rollout_mae_c']:.2f}), "
            f"IoU fuego {fila['rollout_iou_fuego']:.3f} (copia {fila['copia_rollout_iou_fuego']:.3f}), "
            f"IoU quemado {fila['rollout_iou_quemado']:.3f} (copia {fila['copia_rollout_iou_quemado']:.3f})")


# ── Entrenamiento con scripts/main.py ────────────────────────────────────────

def secuencias(S: dict, paso: int, carpeta: Path, prefijo: str) -> dict:
    """Guarda las secuencias con desfase de cada escenario y regresa {escenario: [rutas]}."""
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for k, s in S.items():
        rutas[k] = []
        for o in range(paso):
            destino = carpeta / f"{prefijo}_{k}_k{paso}_o{o}.npy"
            np.save(destino, s[o::paso].astype(np.int32))
            rutas[k].append(destino)
    return rutas


def entrenar_main(c: dict, datos: list, base: Path, n_estados: int, carpeta: Path) -> dict:
    bc.CARPETA = carpeta              # bc.log escribe en carpeta/registro.log
    bc.ESTADOS = n_estados            # --out-size (y --one-hot si la entrada es one-hot)
    return bc.entrenar(c, datos, base)


# ── Predicción ───────────────────────────────────────────────────────────────

def predictor(rutas: list, dev, n_estados: int):
    """Promedio de las probabilidades (softmax) de una o varias redes; regresa (prob, vecindad)."""
    redes = [cargar_red(str(r), dev) for r in rutas]
    nb = redes[0][1]
    assert all(np.array_equal(nb, v) for _, v in redes)

    def prob(X: np.ndarray) -> torch.Tensor:
        salidas = []
        with torch.no_grad():
            for s in range(0, len(X), 1 << 18):
                x = torch.from_numpy(X[s:s + (1 << 18)]).to(dev)
                p = sum(torch.softmax(red(x), 1) for red, _ in redes) / len(redes)
                if p.shape[1] < n_estados:          # main.py quita las clases que no aparecen en el entrenamiento
                    p = torch.nn.functional.pad(p, (0, n_estados - p.shape[1]))
                salidas.append(p)
        return torch.cat(salidas)
    return prob, nb


def decodificar(p: torch.Tensor, ncanal: int, decodificador: str, gen: torch.Generator = None) -> np.ndarray:
    """Probabilidades sobre estados T + NT·canal → estado de temperatura T predicho."""
    pt = p.view(len(p), ncanal, NT).sum(1)
    if decodificador == "argmax":
        T = pt.argmax(1)
    elif decodificador == "muestreo":
        T = torch.multinomial(pt.clamp_min(0) + 1e-12, 1, generator=gen).squeeze(1)
    elif decodificador == "esperado":
        temp = pt @ REP_T.to(pt.device)
        T = torch.bucketize(temp, torch.tensor(CORTES, dtype=torch.float32, device=pt.device), right=True)
    else:
        raise ValueError(decodificador)
    return T.cpu().numpy()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


# ── Evaluación ───────────────────────────────────────────────────────────────

def evaluar(prob, nb, temps: dict, clave: str, dev, paso: int = PASO, canal: dict = None, ncanal: int = 1,
            decodificador: str = "argmax", repeticiones: int = 1, semilla: int = 0) -> dict:
    """Métricas de un escenario. canal: {escenario: arreglo (H, W) estático} que se suma como NT · canal."""
    gen = torch.Generator(device=dev).manual_seed(semilla)
    t = temps[clave]
    c = canal[clave] if canal else 0
    S = estados(t) + NT * c
    centro = int(np.where((nb == 0).all(1))[0][0])

    # un paso
    acum = []
    for _ in range(repeticiones if decodificador == "muestreo" else 1):
        pred, actual, real, real_t = [], [], [], []
        for o in range(paso):
            X, _ = extract_patches(S[o::paso].astype(np.float32), nb, True)
            pred.append(decodificar(prob(X), ncanal, decodificador, gen))
            actual.append(X[:, centro].astype(np.int64) % NT)
            real.append(t[o::paso][1:].ravel())
            real_t.append(t[o::paso][:-1].ravel())
        pred, actual = np.concatenate(pred), np.concatenate(actual)
        real, real_t = np.concatenate(real), np.concatenate(real_t)
        sig = estados(real)
        frio = real_t < FRIO_C
        enciende = frio & (real >= FUEGO_C)
        pred_fuego = frio & (REP[pred] >= FUEGO_C)
        acum.append({
            "exact_temp": float((pred == sig).mean()),
            "mae_c": float(np.abs(REP[pred] - real).mean()),
            "encendido": float((REP[pred][enciende] >= FUEGO_C).mean()),
            "precision_encendido": float((real[pred_fuego] >= FUEGO_C).mean()) if pred_fuego.any() else np.nan,
            "predichos_encendido": float(pred_fuego.sum()),
        })
    fila = {k: round(float(np.nanmean([a[k] for a in acum])), 5) for k in acum[0]}
    fila["predichos_encendido"] = int(fila["predichos_encendido"])
    fila.update({
        "copia_exact_temp": round(float((actual == sig).mean()), 5),
        "copia_mae_c": round(float(np.abs(REP[actual] - real).mean()), 4),
        "cuantizacion_mae_c": round(float(np.abs(REP[sig] - real).mean()), 4),
    })

    # simulación libre
    pasos = round(HORIZONTE / paso)
    S0, t0 = S[0::paso], t[0::paso]
    assert len(t0) > pasos, f"{clave}: la secuencia no alcanza {pasos} pasos de {paso} frames"
    real_fin = t0[pasos]
    fuego_reales = [t0[k] >= UMBRAL_FUEGO for k in range(1, pasos + 1)]
    quemado_real = np.logical_or.reduce([t0[0] >= UMBRAL_FUEGO] + fuego_reales)
    maes, ious, ious_q, campos = [], [], [], []
    for _ in range(repeticiones):
        estado = S0[0].astype(np.float32)
        quemado = (S0[0] % NT) >= 8
        ious_paso = []
        for k in range(pasos):
            X, _ = extract_patches(np.stack([estado, estado]), nb, True)
            T = decodificar(prob(X), ncanal, decodificador, gen).reshape(estado.shape)
            estado = (T + NT * c).astype(np.float32)
            ious_paso.append(iou(T >= 8, fuego_reales[k]))
            quemado |= T >= 8
        T_fin = estado.astype(int) % NT
        campos.append(REP[T_fin])
        maes.append(float(np.abs(REP[T_fin] - real_fin).mean()))
        ious.append(np.mean(ious_paso))
        ious_q.append(iou(quemado, quemado_real))
    T_ini = S0[0] % NT
    fuego_ini = T_ini >= 8
    fila.update({
        "rollout_frames": pasos * paso,
        "rollout_mae_c": round(float(np.mean(maes)), 3),
        "rollout_mae_c_std": round(float(np.std(maes)), 3),
        "rollout_mae_c_campo_medio": round(float(np.abs(np.mean(campos, 0) - real_fin).mean()), 3),
        "rollout_iou_fuego": round(float(np.mean(ious)), 4),
        "rollout_iou_quemado": round(float(np.mean(ious_q)), 4),
        "copia_rollout_mae_c": round(float(np.abs(REP[T_ini] - real_fin).mean()), 3),
        "copia_rollout_iou_fuego": round(float(np.mean([iou(fuego_ini, f) for f in fuego_reales])), 4),
        "copia_rollout_iou_quemado": round(iou(fuego_ini, quemado_real), 4),
    })
    return fila
