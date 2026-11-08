"""Funciones comunes para los experimentos recomendados (ver README.md de esta carpeta).

Parte de ../opciones/comun.py (misma referencia, mismos escenarios y la misma evaluación de un paso y simulación libre)
y agrega:

Métricas de propagación (recomendación 1.2), sobre la simulación libre de HORIZONTE = 120 frames desde el primer frame.
Las celdas que ya arden en el primer frame real (≥ 135.9 °C) se excluyen: solo cuenta el fuego nuevo.
    llegada_mae_min        error medio del tiempo de llegada (primer paso ≥ 135.9 °C) en minutos, sobre las celdas que
                           se queman en la simulación o en la realidad; si una nunca se quema cuenta como un paso
                           después del final
    llegada_mae_ambos_min  lo mismo solo sobre las celdas que se queman en las dos
    llegada_sesgo_min      promedio de (llegada predicha − real) en las celdas que se queman en las dos; > 0 = tarde
    area_quemada_rel       celdas nuevas quemadas en la simulación / en la realidad (1 = misma área)
    iou_quemado_nuevo      intersección / unión de las celdas nuevas quemadas
    fss_quemado_n          Fractions Skill Score del área nueva quemada con ventanas de n×n celdas (n = 1, 3, 5, 9);
                           1 = perfecto, 0 = sin coincidencia. Con n > 1 tolera desplazamientos chicos
    fss_fuego_n            FSS de las celdas ≥ 135.9 °C en cada paso, promedio de los pasos
    copia_*                lo mismo para "nada cambia"
A 200×200 las ventanas se escalan (n → 2n − 1) para cubrir la misma área en metros que a 100×100.

Entrenador en GPU (entrenar_gpu): arma las vecindades por lote en la GPU a partir de las secuencias de estados, así
la memoria no crece con el tamaño de la vecindad. Mismo modelo (main.build_model), Adam, exp 1e-3 → 1e-5, frontera
abierta (relleno con 0) y semillas que main.py. Opciones: orientaciones al azar (8, grupo diedral), ruido en la
entrada, etiquetas ordinales, destilación de un ensamble, validación con los últimos frames y weight decay.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter

REC = Path(__file__).resolve().parent
sys.path.insert(0, str(REC.parent / "opciones"))
import comun as cm                                                      # noqa: E402
from comun import (NT, REP, CORTES, UMBRAL_FUEGO, FRIO_C, FUEGO_C, HORIZONTE, PASO, BASE, ESC, RONDAS, SEMILLAS,  # noqa
                   INICIO, estados, log, hecho, ronda_nombre, red_referencia, predictor, decodificar,
                   extract_patches, build_model, save_weights, iou, secuencias)


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

FLAME = REC.parent
TRAINING = cm.TRAINING
OPC = cm.OPC
FUEGO_ESTADO = 8                                      # estados 8–14 = ≥ 135.9 °C
VENTANAS = (1, 3, 5, 9)


def temperaturas(lado: int = 100) -> dict:
    if lado == 100:
        return cm.temperaturas()
    return {k: np.load(OPC / "7_resolucion200" / "datos" / f"{k}_celsius_200.npy") for k in ESC}


def redes_200(ronda: str, semilla: int) -> Path:
    return OPC / "7_resolucion200" / "redes" / ronda / f"200_k6_r4_cruda_2x32_exp_s{semilla}.npz"


# ── Métricas de propagación ──────────────────────────────────────────────────

def fss(pred: np.ndarray, real: np.ndarray, n: int) -> float:
    if n > 1:
        pf = uniform_filter(pred.astype(np.float64), size=n, mode="constant")
        po = uniform_filter(real.astype(np.float64), size=n, mode="constant")
    else:
        pf, po = pred.astype(np.float64), real.astype(np.float64)
    ref = (pf ** 2).mean() + (po ** 2).mean()
    return float(1 - ((pf - po) ** 2).mean() / ref) if ref > 0 else 1.0


def llegada(mascaras: list) -> np.ndarray:
    """Primer índice (1…) en que cada celda está en la máscara; len(mascaras) + 1 si nunca."""
    inf = len(mascaras) + 1
    a = np.full(mascaras[0].shape, inf, dtype=np.int32)
    for k in range(len(mascaras), 0, -1):
        a[mascaras[k - 1]] = k
    return a


def metricas_propagacion(Ts: list, t0: np.ndarray, paso: int, escala: int = 1) -> dict:
    """Ts: estados de temperatura predichos en los pasos 1…n; t0: °C reales en los pasos 0…n (cada `paso` frames)."""
    pasos = len(Ts)
    fuego_real = [t0[k] >= UMBRAL_FUEGO for k in range(pasos + 1)]
    fuego_pred = [T >= FUEGO_ESTADO for T in Ts]
    nuevas = ~fuego_real[0]
    inf = pasos + 1
    a_r, a_p = llegada(fuego_real[1:]), llegada(fuego_pred)
    qr, qp = nuevas & (a_r < inf), nuevas & (a_p < inf)
    ambas, alguna = qr & qp, qr | qp
    minutos = paso * 5 / 60
    dif = (a_p - a_r).astype(np.float64)
    f = {
        "llegada_mae_min": round(float(np.abs(dif[alguna]).mean() * minutos), 4) if alguna.any() else np.nan,
        "llegada_mae_ambos_min": round(float(np.abs(dif[ambas]).mean() * minutos), 4) if ambas.any() else np.nan,
        "llegada_sesgo_min": round(float(dif[ambas].mean() * minutos), 4) if ambas.any() else np.nan,
        "area_quemada_rel": round(float(qp.sum() / qr.sum()), 4) if qr.any() else np.nan,
        "iou_quemado_nuevo": round(iou(qp, qr), 4),
    }
    for n in VENTANAS:
        m = escala * (n - 1) + 1
        f[f"fss_quemado_{n}"] = round(fss(qp, qr, m), 4)
        f[f"fss_fuego_{n}"] = round(float(np.mean([fss(fuego_pred[k], fuego_real[k + 1], m) for k in range(pasos)])), 4)
    return f


def trayectoria(prob, nb, S0: np.ndarray, pasos: int, c=0, ncanal: int = 1, decodificador: str = "argmax",
                gen=None) -> list:
    estado = S0.astype(np.float32)
    Ts = []
    for _ in range(pasos):
        X, _ = extract_patches(np.stack([estado, estado]), nb, True)
        T = decodificar(prob(X), ncanal, decodificador, gen).reshape(estado.shape)
        estado = (T + NT * c).astype(np.float32)
        Ts.append(T)
    return Ts


def propagacion(prob, nb, temps: dict, clave: str, dev, paso: int = PASO, canal: dict = None, ncanal: int = 1,
                decodificador: str = "argmax", repeticiones: int = 1, semilla: int = 0, escala: int = 1) -> dict:
    gen = torch.Generator(device=dev).manual_seed(semilla + 1)
    t = temps[clave]
    c = canal[clave] if canal else 0
    S0 = estados(t[0]) + NT * c
    pasos = round(HORIZONTE / paso)
    t0 = t[0::paso][:pasos + 1]
    filas = [metricas_propagacion(trayectoria(prob, nb, S0, pasos, c, ncanal, decodificador, gen), t0, paso, escala)
             for _ in range(repeticiones)]
    f = {k: round(float(np.nanmean([x[k] for x in filas])), 4) for k in filas[0]}
    copia = metricas_propagacion([estados(t[0])] * pasos, t0, paso, escala)
    f.update({f"copia_{k}": v for k, v in copia.items()})
    return f


def evaluar(prob, nb, temps: dict, clave: str, dev, paso: int = PASO, escala: int = 1, **kw) -> dict:
    """Evaluación de ../opciones/comun.py más las métricas de propagación."""
    return {**cm.evaluar(prob, nb, temps, clave, dev, paso, **kw),
            **propagacion(prob, nb, temps, clave, dev, paso, escala=escala, **kw)}


def resumen(f: dict) -> str:
    return (cm.resumen(f) + f"; llegada {f['llegada_mae_min']} min (copia {f['copia_llegada_mae_min']}), "
            f"área {f['area_quemada_rel']}, FSS quemado 3 {f['fss_quemado_3']}, 9 {f['fss_quemado_9']}")


# ── Variantes del predictor ──────────────────────────────────────────────────

def con_sesgo(prob, delta: float):
    """Suma delta a la salida (logit) de los estados de fuego (T = 8–14, en cualquier canal) antes del argmax: p_s ∝ p_s · e^delta."""
    factor = None

    def p2(X):
        nonlocal factor
        p = prob(X)
        if factor is None or factor.device != p.device:
            factor = torch.ones(p.shape[1], device=p.device)
            factor[torch.arange(p.shape[1], device=p.device) % NT >= FUEGO_ESTADO] = float(np.exp(delta))
        q = p * factor
        return q / q.sum(1, keepdim=True)
    return p2


def permutaciones_d4(nb: np.ndarray) -> np.ndarray:
    """(8, vecinos): columnas de la vecindad que corresponden a cada rotación/reflejo de la vecindad."""
    idx = {(int(dx), int(dy)): i for i, (dx, dy) in enumerate(nb)}
    trans = [lambda x, y: (x, y), lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x),
             lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (y, x), lambda x, y: (-y, -x)]
    return np.array([[idx[g(int(dx), int(dy))] for dx, dy in nb] for g in trans])


def con_orientaciones(prob, nb, n: int = 8):
    """Promedio de las probabilidades sobre las n primeras orientaciones (4 = rotaciones, 8 = rotaciones y reflejos)."""
    perms = permutaciones_d4(nb)[:n]

    def p2(X):
        return sum(prob(np.ascontiguousarray(X[:, pm])) for pm in perms) / len(perms)
    return p2


def tabla_transicion(secs: list, dev):
    """P(estado siguiente | estado actual) contada en las secuencias de entrenamiento: la NCA sin vecinos óptima."""
    cuenta = np.zeros((NT, NT))
    for s in secs:
        np.add.at(cuenta, (s[:-1].ravel(), s[1:].ravel()), 1)
    vistos = cuenta.sum(1) > 0
    P = np.where(vistos[:, None], cuenta / np.maximum(cuenta.sum(1, keepdims=True), 1), np.eye(NT))
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)

    def prob(X):
        return Pt[torch.from_numpy(X[:, 0].astype(np.int64)).to(dev)]
    return prob, np.array([[0, 0]], dtype=np.int32), P


def calibrar_sesgo(prob, nb, secs: list, temps_secs: list, lo: float = -3.0, hi: float = 12.0) -> dict:
    """delta para que, en las secuencias de entrenamiento, se predigan tantos encendidos (frío → ≥ 130 °C) como hay."""
    ps, reales = [], 0
    for s, t in zip(secs, temps_secs):
        X, _ = extract_patches(s.astype(np.float32), nb, True)
        frio = (t[:-1] < FRIO_C).ravel()
        reales += int((frio & (t[1:].ravel() >= FUEGO_C)).sum())
        p = prob(X[frio])
        ps.append(torch.log(p.view(len(p), -1, NT).sum(1).clamp_min(1e-12)))     # como decodificar: suma los canales
    lp = torch.cat(ps)
    fuego_rep = torch.tensor(REP >= FUEGO_C, device=lp.device)
    masc = torch.zeros(NT, device=lp.device)
    masc[FUEGO_ESTADO:] = 1

    def predichos(d):
        return int(fuego_rep[(lp + d * masc).argmax(1)].sum())
    n0 = predichos(0.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        if predichos(mid) < reales:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return {"delta": round(d, 4), "encendidos_reales_ent": reales, "predichos_ent_sin_sesgo": n0,
            "predichos_ent_con_sesgo": predichos(d)}


def guardar_con_sesgo(origen: Path, destino: Path, delta: float):
    """Copia del .npz con delta sumado al sesgo de la última capa en los estados 8–14 (lo usa igual la GUI)."""
    z = dict(np.load(origen))
    ult = len(z["layer_out"]) - 1
    b = z[f"biases_{ult}"].copy()
    b[FUEGO_ESTADO:] += delta
    z[f"biases_{ult}"] = b
    np.savez(destino, **z)


# ── Entrenador en GPU ────────────────────────────────────────────────────────

def entrenar_gpu(secs: list, nb: np.ndarray, c: dict, base: Path, dev, aug: str = None, ruido: float = 0.0,
                 suavizado: float = 1.0, maestros: list = None, val: float = 0.0, wd: float = 0.0) -> dict:
    """secs: lista de secuencias de estados (T, H, W); las transiciones son t → t+1 dentro de cada secuencia.
    suavizado: peso de la etiqueta real (1 = entropía cruzada normal; el resto se reparte entre los estados ±1).
    val: fracción de las últimas transiciones de cada secuencia que se aparta como validación; con val > 0 se guarda
    la época con menor pérdida de validación."""
    np.random.seed(c["semilla"])
    torch.manual_seed(c["semilla"])
    torch.cuda.manual_seed_all(c["semilla"])
    r = int(np.abs(nb).max())
    H, W = secs[0].shape[1:]
    marcos, ent, va = [], [], []
    base_idx = 0
    for s in secs:
        n_tr = len(s) - 1
        n_va = int(np.ceil(val * n_tr)) if val > 0 else 0
        ent += range(base_idx, base_idx + n_tr - n_va)
        va += range(base_idx + n_tr - n_va, base_idx + n_tr)
        marcos.append(s)
        base_idx += len(s)
    G = torch.from_numpy(np.concatenate(marcos).astype(np.uint8)).to(dev)
    F = torch.nn.functional.pad(G, (r, r, r, r))
    dx = torch.tensor(nb[:, 0], device=dev).long()
    dy = torch.tensor(nb[:, 1], device=dev).long()
    perms = torch.tensor(permutaciones_d4(nb), device=dev) if aug == "d4" else None
    ent_t = torch.tensor(ent, device=dev).long()
    va_t = torch.tensor(va, device=dev).long()
    HW = H * W

    def lote(fuentes, idx):
        f = fuentes[idx // HW]
        p = idx % HW
        y, x = p // W, p % W
        X = F[f[:, None], y[:, None] + r + dy[None], x[:, None] + r + dx[None]].float()
        Y = G[f + 1, y, x].long()
        return X, Y

    capas = [(c["neuronas"], "relu")] * c["capas"]
    modelo = build_model(len(nb), capas, NT, "linear").to(dev)
    opt = (torch.optim.AdamW(modelo.parameters(), lr=c["lr"], weight_decay=wd) if wd > 0
           else torch.optim.Adam(modelo.parameters(), lr=c["lr"]))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(c["lr_min"] / c["lr"]) ** (1 / (c["epocas"] - 1)))
    profes = None
    if maestros:
        profes = [cm.cargar_red(str(m), dev)[0] for m in maestros]
    logsm = torch.nn.functional.log_softmax
    N = len(ent_t) * HW
    mejor, mejor_val, mejor_epoca = None, np.inf, -1
    t0 = time.perf_counter()
    with open(f"{base}_loss.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["epoch", "loss", "val_loss", "lr"])
        for epoca in range(c["epocas"]):
            modelo.train()
            perm = torch.randperm(N, device=dev)
            total, lotes = 0.0, 0
            for s in range(0, N, c["lote"]):
                X, Y = lote(ent_t, perm[s:s + c["lote"]])
                if profes is not None:
                    with torch.no_grad():
                        q = sum(torch.softmax(m(X), 1) for m in profes) / len(profes)
                        if q.shape[1] < NT:
                            q = torch.nn.functional.pad(q, (0, NT - q.shape[1]))
                if perms is not None:
                    X = X.gather(1, perms[torch.randint(0, 8, (len(X),), device=dev)])
                if ruido > 0:
                    cambia = torch.rand_like(X) < ruido
                    signo = torch.randint(0, 2, X.shape, device=dev).float() * 2 - 1
                    X = (X + cambia * signo).clamp(0, NT - 1)
                lsm = logsm(modelo(X), 1)
                if profes is not None:
                    perdida = -(q * lsm).sum(1).mean()
                elif suavizado < 1:
                    q = torch.zeros_like(lsm)
                    q.scatter_(1, Y[:, None], suavizado)
                    resto = (1 - suavizado) / 2
                    for d in (-1, 1):
                        v = Y + d
                        ok = (v >= 0) & (v < NT)
                        q[ok, v[ok]] += resto
                        q[~ok, Y[~ok]] += resto
                    perdida = -(q * lsm).sum(1).mean()
                else:
                    perdida = torch.nn.functional.nll_loss(lsm, Y)
                opt.zero_grad()
                perdida.backward()
                opt.step()
                total += perdida.item()
                lotes += 1
            vl = np.nan
            if len(va_t):
                modelo.eval()
                with torch.no_grad():
                    Nv, acum = len(va_t) * HW, 0.0
                    for s in range(0, Nv, 1 << 16):
                        idx = torch.arange(s, min(s + (1 << 16), Nv), device=dev)
                        X, Y = lote(va_t, idx)
                        acum += torch.nn.functional.cross_entropy(modelo(X), Y, reduction="sum").item()
                    vl = acum / Nv
                if vl < mejor_val:
                    mejor_val, mejor_epoca = vl, epoca
                    mejor = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            w.writerow([epoca, total / lotes, vl, opt.param_groups[0]["lr"]])
            sched.step()
    if mejor is not None:
        modelo.load_state_dict(mejor)
    save_weights(f"{base}.npz", modelo, nb, capas, "linear", 0)
    return {"epocas_hechas": c["epocas"], "loss_final": round(total / lotes, 6),
            "val_mejor": round(float(mejor_val), 6) if mejor is not None else np.nan,
            "mejor_epoca": mejor_epoca, "tiempo_s": round(time.perf_counter() - t0, 1)}


def vecindad(radio: int) -> np.ndarray:
    """Moore de radio r en el mismo orden que main.build_neighbourhood."""
    return np.array([(dx, dy) for dy in range(-radio, radio + 1) for dx in range(-radio, radio + 1)], dtype=np.int32)


def secs_offset(S: np.ndarray, paso: int = PASO) -> list:
    return [S[o::paso] for o in range(paso)]


# ── Bucles comunes ───────────────────────────────────────────────────────────

def correr_entrenador(carpeta: Path, experimento: str, variante: str, dev, lado: int = 100, radio: int = 4,
                      semillas=SEMILLAS, c_extra: dict = None, maestros_fn=None, **kw_ent):
    """Entrena con entrenar_gpu en cada ronda y semilla, evalúa en el escenario no visto y escribe resultados.csv."""
    temps = temperaturas(lado)
    for entrena, prueba in RONDAS:
        ronda = ronda_nombre(entrena, prueba)
        secs = [s for k in entrena for s in secs_offset(estados(temps[k]))]
        for s in semillas:
            clave = {"experimento": experimento, "variante": variante, "ronda": ronda, "semilla": s}
            if hecho(carpeta, clave):
                continue
            (carpeta / "redes" / ronda).mkdir(parents=True, exist_ok=True)
            base = carpeta / "redes" / ronda / f"{variante}_s{s}"
            c = dict(BASE, semilla=s, **(c_extra or {}))
            info = entrenar_gpu(secs, vecindad(radio), c, base, dev,
                                maestros=maestros_fn(ronda) if maestros_fn else None, **kw_ent)
            prob, nb = predictor([f"{base}.npz"], dev, NT)
            fila = {**clave, "prueba": prueba, "lado": lado, "radio": radio, "capas": c["capas"],
                    "neuronas": c["neuronas"], **info, **evaluar(prob, nb, temps, prueba, dev, escala=lado // 100)}
            escribir(carpeta, fila)
            log(carpeta, f"{variante} {ronda} s{s}: {resumen(fila)}")


def evaluar_ensamble(carpeta: Path, experimento: str, variante: str, redes_fn, dev, lado: int = 100,
                     orientaciones: int = 0):
    """Promedio de las redes redes_fn(ronda); opcionalmente también de 4 u 8 orientaciones de la entrada."""
    temps = temperaturas(lado)
    for entrena, prueba in RONDAS:
        ronda = ronda_nombre(entrena, prueba)
        clave = {"experimento": experimento, "variante": variante, "ronda": ronda, "semilla": "ensamble"}
        if hecho(carpeta, clave):
            continue
        redes = redes_fn(ronda)
        prob, nb = predictor(redes, dev, NT)
        if orientaciones:
            prob = con_orientaciones(prob, nb, orientaciones)
        fila = {**clave, "prueba": prueba, "lado": lado, "redes": len(redes), "orientaciones": orientaciones,
                **evaluar(prob, nb, temps, prueba, dev, escala=lado // 100)}
        escribir(carpeta, fila)
        log(carpeta, f"{variante} {ronda}: {resumen(fila)}")
