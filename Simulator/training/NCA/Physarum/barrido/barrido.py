"""Barrido nocturno de NCA para Physarum, con prioridad en que funcionen en la GUI (argmax, ≤ 128 vecinos).

Cada configuración:
1. Datos: de CA/Physarum/<ca>/ toma las 20 cajas de entrenamiento (todas menos dish5), opcionalmente une estados
   (`mapa`) y toma una imagen de cada `paso` (paso de tiempo = 30 min × paso). De cada caja, `por_caja` transiciones
   repartidas en el tiempo (las que quepan en ~8.6 GB de muestras float32 de main.py), archivos (N, 2, H, W).
2. Entrena con scripts/main.py (one-hot o lineal, CrossEntropy, lr exp 1e-3 → 1e-5, frontera abierta).
3. Sesgo: suma una constante b a las salidas de plasmodio (como cambiar el sesgo de la última capa, así que la GUI lo
   usa tal cual). b se elige en simulación libre con argmax en 4 cajas de ENTRENAMIENTO (dish0 de cada experimento)
   maximizando el puntaje medio en el tiempo (IoU de la máscara de plasmodio × área menor / área mayor; el IoU solo
   premiaba llenar la caja); rejilla 0–6 cada 0.5 y luego ±0.375.
4. Prueba en las 4 dish5 (no vistas): simulación libre desde la primera imagen con argmax, con b = 0 y con b*;
   un paso con b*. Guarda red_gui.npz (con b* en el sesgo), libre_<caja>.npy y una fila en resultados.csv.
   Con "vector": True, después de b* calibra un sesgo por estado (búsqueda coordenada, mismas cajas y criterio).

    python barrido.py <fase>        (fases en FASES; se salta las configuraciones que ya están en resultados.csv)
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[2]
sys.path.insert(0, str(TRAINING / "scripts"))
from barrido_flame import cargar_red                                       # noqa: E402

PY = sys.executable
MAIN = TRAINING / "scripts" / "main.py"
CA = TRAINING / "CA" / "Physarum"
RESULTADOS = CARPETA / "resultados_v3.csv"   # resultados.csv: fase 1 calibrada con IoU (criterio viejo); _v2: puntaje
                                               # con 4 cajas de calibración; _v3: puntaje con 8 cajas
REGISTRO = CARPETA / "registro.log"

PRUEBA = [f"physarum-{e}-dish5-r2c1" for e in ("001", "002", "008", "009")]
CALIBRACION = [f"physarum-{e}-dish{d}-r0c{d}" for e in ("001", "002", "008", "009") for d in (0, 1)]
PRESUPUESTO = 8.6e9                                    # bytes de muestras float32 (cabe en RAM y VRAM)
SESGOS = np.arange(0, 6.01, 0.5)
HORAS = (6, 12, 24, 48)


# ── Vecindades (pares dx, dy; todas ≤ 128 células, el límite del núcleo de la GUI) ─────────────────────────────

def cuadrada(r):
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


def circular(r):
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= r * r]


def anillos(radios, puntos):
    """Centro y anillos de radio r con n puntos (los impares girados medio paso), como en SEVIR."""
    nb = [(0, 0)]
    for i, (r, n) in enumerate(zip(radios, puntos)):
        giro = np.pi / n if i % 2 else 0.0
        for k in range(n):
            a = giro + 2 * np.pi * k / n
            p = (int(round(r * np.cos(a))), int(round(r * np.sin(a))))
            if p not in nb:
                nb.append(p)
    return nb


VECINDADES = {
    "anillos3": anillos((1, 2, 3), (8, 12, 16)),
    "anillos4": anillos((1, 2, 3, 4), (8, 12, 16, 20)),
    "anillos6": anillos((1, 2, 4, 6), (8, 12, 16, 20)),
    "anillos8": anillos((1, 2, 4, 6, 8), (8, 12, 16, 20, 24)),
    "anillos10": anillos((1, 2, 4, 6, 8, 10), (8, 12, 16, 20, 24, 26)),
    "5x5": cuadrada(2), "7x7": cuadrada(3), "9x9": cuadrada(4), "11x11": cuadrada(5),
    "circ4": circular(4), "circ6": circular(6),
}
assert all(len(v) <= 128 for v in VECINDADES.values())

# Uniones de los 6 estados (borde, fondo, avena, plasmodio tenue, medio, intenso)
MAPAS = {
    "3": [0, 1, 1, 2, 2, 2],        # sin avena (avena → fondo), plasmodio de un nivel
    "4": [0, 1, 2, 3, 3, 3],        # plasmodio de un nivel
    "5": [0, 1, 2, 3, 4, 4],        # tenue | medio + intenso
    "5b": [0, 1, 2, 3, 3, 4],       # tenue + medio | intenso
}

BASE = {"ca": "estados_cada30min_160", "mapa": None, "vecindad": "anillos6", "capas": "16x64", "one_hot": True,
        "paso": 1, "epocas": 30, "semilla": 42, "vector": False, "desfases": False, "aumento": False}


def var(nombre, **cambios):
    return nombre, {**BASE, **cambios}


FASES = {
    # Fase 2: lo mejor de la fase 1 (4 estados, paso 2 h, vecindades grandes, 8:128) combinado
    "fase2": [
        var("e4_2h", mapa="4", paso=4),
        *[var(f"e4_2h_{v}", mapa="4", paso=4, vecindad=v) for v in ("11x11", "anillos10", "circ6", "9x9", "anillos8")],
        var("e4_2h_8x128", mapa="4", paso=4, capas="8x128"),
        var("e4_2h_11x11_8x128", mapa="4", paso=4, vecindad="11x11", capas="8x128"),
        var("e4_2h_anillos10_8x128", mapa="4", paso=4, vecindad="anillos10", capas="8x128"),
        var("e4_1h", mapa="4", paso=2), var("e4_3h", mapa="4", paso=6), var("e4_4h", mapa="4", paso=8),
        var("e4_1h_11x11", mapa="4", paso=2, vecindad="11x11"),
        var("e3_2h", mapa="3", paso=4), var("e3_2h_11x11", mapa="3", paso=4, vecindad="11x11"),
        var("e6_2h", paso=4), var("e6_2h_11x11", paso=4, vecindad="11x11"),
        var("e4_2h_100ep", mapa="4", paso=4, epocas=100),
        var("e4_2h_vector", mapa="4", paso=4, vector=True),
        var("e4_2h_11x11_vector", mapa="4", paso=4, vecindad="11x11", vector=True),
    ],
    # Fase 2b: pasos más largos (2 h → 4 h mejoró) y combinaciones a 4 h
    "fase2b": [
        var("e4_6h", mapa="4", paso=12), var("e4_8h", mapa="4", paso=16),
        *[var(f"e4_4h_{v}", mapa="4", paso=8, vecindad=v) for v in ("circ6", "9x9", "11x11", "anillos4", "circ4")],
        var("e4_4h_8x128", mapa="4", paso=8, capas="8x128"),
        var("e3_4h", mapa="3", paso=8), var("e6_4h", paso=8), var("e5b_4h", mapa="5b", paso=8),
        var("e8_4h", ca="estados_cada30min_8estados_160", paso=8),
        var("e4_4h_100ep", mapa="4", paso=8, epocas=100),
        var("e4_4h_vector", mapa="4", paso=8, vector=True),
        var("e4_6h_circ6", mapa="4", paso=12, vecindad="circ6"),
    ],
    # Fase 3: más datos a pasos largos (todos los desfases de 30 min) y simetrías del cuadrado
    "fase3": [
        var("e4_4h_d", mapa="4", paso=8, desfases=True),
        var("e4_4h_da", mapa="4", paso=8, desfases=True, aumento=True),
        var("e4_4h_a", mapa="4", paso=8, aumento=True),
        var("e4_2h_d", mapa="4", paso=4, desfases=True),
        var("e4_2h_da", mapa="4", paso=4, desfases=True, aumento=True),
        var("e4_4h_circ6_da", mapa="4", paso=8, vecindad="circ6", desfases=True, aumento=True),
        var("e4_4h_9x9_da", mapa="4", paso=8, vecindad="9x9", desfases=True, aumento=True),
        var("e4_4h_8x128_da", mapa="4", paso=8, capas="8x128", desfases=True, aumento=True),
        var("e6_4h_da", paso=8, desfases=True, aumento=True),
        var("e3_4h_da", mapa="3", paso=8, desfases=True, aumento=True),
        var("e4_4h_320_da", ca="estados_cada30min", mapa="4", paso=8, desfases=True, aumento=True),
        var("e4_6h_da", mapa="4", paso=12, desfases=True, aumento=True),
    ],
    # Fase 4: combinaciones de lo mejor (4–6 h, 9×9 / circ6, desfases + aumento) y semillas de las 4 mejores
    "fase4": [
        var("e4_6h_9x9_da", mapa="4", paso=12, vecindad="9x9", desfases=True, aumento=True),
        var("e4_6h_circ6_da", mapa="4", paso=12, vecindad="circ6", desfases=True, aumento=True),
        var("e4_5h_9x9_da", mapa="4", paso=10, vecindad="9x9", desfases=True, aumento=True),
        var("e4_4h_9x9_da_100ep", mapa="4", paso=8, vecindad="9x9", desfases=True, aumento=True, epocas=100),
        var("e6_6h_da", paso=12, desfases=True, aumento=True),
        var("e6_6h_9x9_da", paso=12, vecindad="9x9", desfases=True, aumento=True),
        *[var(f"{n}_s{s}", semilla=s, **c) for s in (43, 44) for n, c in (
            ("e4_4h_9x9_da", dict(mapa="4", paso=8, vecindad="9x9", desfases=True, aumento=True)),
            ("e4_6h_da", dict(mapa="4", paso=12, desfases=True, aumento=True)),
            ("e4_6h_circ6", dict(mapa="4", paso=12, vecindad="circ6")),
            ("e4_4h_9x9", dict(mapa="4", paso=8, vecindad="9x9")))],
    ],
    # Fase 4b: semillas de las dos mejores nuevas (6 estados a 6 h con 9×9; 4 estados a 5 h con 9×9)
    "fase4b": [
        *[var(f"{n}_s{s}", semilla=s, **c) for s in (43, 44) for n, c in (
            ("e6_6h_9x9_da", dict(paso=12, vecindad="9x9", desfases=True, aumento=True)),
            ("e4_5h_9x9_da", dict(mapa="4", paso=10, vecindad="9x9", desfases=True, aumento=True)))],
    ],
    # Fase 5: alrededor de 6 h + 9×9 + desfases + aumento (redes, estados, vecindades, entrada)
    "fase5": [
        *[var(f"e4_6h_9x9_da_{c}", mapa="4", paso=12, vecindad="9x9", desfases=True, aumento=True, capas=c)
          for c in ("4x64", "8x64", "16x32", "24x64")],
        *[var(f"e{m}_6h_9x9_da", mapa=m, paso=12, vecindad="9x9", desfases=True, aumento=True) for m in ("3", "5", "5b")],
        var("e8_6h_9x9_da", ca="estados_cada30min_8estados_160", paso=12, vecindad="9x9", desfases=True, aumento=True),
        *[var(f"e6_6h_{v}_da", paso=12, vecindad=v, desfases=True, aumento=True) for v in ("circ6", "11x11")],
        *[var(f"e6_{h}h_9x9_da", paso=2 * h, vecindad="9x9", desfases=True, aumento=True) for h in (4, 5)],
        *[var(f"e4_6h_{v}_da", mapa="4", paso=12, vecindad=v, desfases=True, aumento=True) for v in ("7x7", "11x11")],
        var("e4_6h_9x9_da_lineal", mapa="4", paso=12, vecindad="9x9", desfases=True, aumento=True, one_hot=False),
    ],
    # Fase 6: alrededor de la mejor (4 estados, 5 h, 9×9, desfases + aumento)
    "fase6": [
        *[var(f"e4_5h_{v}_da", mapa="4", paso=10, vecindad=v, desfases=True, aumento=True)
          for v in ("circ6", "11x11", "anillos6")],
        var("e4_5h_9x9_da_4x64", mapa="4", paso=10, vecindad="9x9", desfases=True, aumento=True, capas="4x64"),
        var("e4_5h_9x9_da_p1", ca="estados_cada30min_160_p1", mapa="4", paso=10, vecindad="9x9", desfases=True,
            aumento=True),
        var("e4_5h_9x9_da_p3", ca="estados_cada30min_160_p3", mapa="4", paso=10, vecindad="9x9", desfases=True,
            aumento=True),
        var("e4_5h_res80_da", ca="estados_cada30min_80", mapa="4", paso=10, vecindad="5x5", desfases=True,
            aumento=True),
    ],
    # Fase 7: una red de 6 estados más robusta (vecindades, 5 h, red chica, semillas)
    "fase7": [
        var("e6_6h_circ6_da_s43", paso=12, vecindad="circ6", desfases=True, aumento=True, semilla=43),
        var("e6_6h_circ6_da_s44", paso=12, vecindad="circ6", desfases=True, aumento=True, semilla=44),
        var("e6_6h_anillos6_da", paso=12, desfases=True, aumento=True),
        var("e6_5h_circ6_da", paso=10, vecindad="circ6", desfases=True, aumento=True),
        var("e6_6h_9x9_da_4x64", paso=12, vecindad="9x9", desfases=True, aumento=True, capas="4x64"),
        var("e4_5h_9x9_da_4x64_s43", mapa="4", paso=10, vecindad="9x9", desfases=True, aumento=True, capas="4x64",
            semilla=43),
        var("e4_5h_9x9_da_4x64_s44", mapa="4", paso=10, vecindad="9x9", desfases=True, aumento=True, capas="4x64",
            semilla=44),
    ],
    # Fase 7b: semillas de la mejor de 6 estados (5 h, circ6)
    "fase7b": [var(f"e6_5h_circ6_da_s{s}", paso=10, vecindad="circ6", desfases=True, aumento=True, semilla=s)
               for s in (43, 44)],
    # Fase 1: un factor a la vez respecto de la base (160 × 160, 6 estados, anillos6, 16:64 one-hot, 30 min)
    "fase1": [
        var("base"),
        var("est3", mapa="3"), var("est4", mapa="4"), var("est5", mapa="5"), var("est5b", mapa="5b"),
        var("est8", ca="estados_cada30min_8estados_160"),
        *[var(f"vec_{v}", vecindad=v) for v in ("anillos3", "anillos4", "anillos8", "anillos10", "5x5", "7x7",
                                                  "9x9", "11x11", "circ4", "circ6")],
        var("red_4x64", capas="4x64"), var("red_8x64", capas="8x64"), var("red_8x128", capas="8x128"),
        var("red_16x128", capas="16x128"),
        var("lineal", one_hot=False),
        var("paso1h", paso=2), var("paso2h", paso=4),
        var("persist1", ca="estados_cada30min_160_p1"), var("persist3", ca="estados_cada30min_160_p3"),
        var("res80", ca="estados_cada30min_80"),
    ],
}


# ── Datos ──────────────────────────────────────────────────────────────────────────────────────────────────────

def cargar(cfg, caja):
    S = np.load(CA / cfg["ca"] / f"{caja}.npy")[::cfg["paso"]]
    return np.array(MAPAS[cfg["mapa"]], np.int8)[S] if cfg["mapa"] else S


def info(cfg):
    """(número de estados, primer estado de plasmodio)."""
    if cfg["mapa"]:
        m = MAPAS[cfg["mapa"]]
        return max(m) + 1, m[3]
    return len(json.loads((CA / cfg["ca"] / "evo.json").read_text())["estados"]), 3


def cajas_entrenamiento(cfg):
    return [p.stem for p in sorted((CA / cfg["ca"]).glob("physarum-*.npy")) if p.stem not in PRUEBA]


def d4(E, i):
    """Una de las 8 simetrías del cuadrado (rotaciones y reflejos) sobre los dos últimos ejes."""
    E = np.rot90(E, i % 4, axes=(-2, -1))
    return np.ascontiguousarray(E[..., ::-1] if i >= 4 else E)


def datos(cfg):
    """desfases: con paso k usa los pares (t, t + k) para todo t (no solo t múltiplo de k); aumento: a cada par le
    aplica una simetría del cuadrado al azar."""
    cajas = cajas_entrenamiento(cfg)
    S0 = cargar(cfg, cajas[0])
    n = len(VECINDADES[cfg["vecindad"]])
    por_caja = int(PRESUPUESTO // (len(cajas) * S0.shape[1] * S0.shape[2] * n * 4))
    clave = (f"{cfg['ca']}_{cfg['mapa'] or 'x'}_k{cfg['paso']}_n{por_caja}" + ("_d" if cfg.get("desfases") else "")
             + ("_a" if cfg.get("aumento") else ""))
    carpeta = CARPETA / "datos" / clave
    if not (carpeta / "listo").exists():
        carpeta.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        k = cfg["paso"]
        for caja in cajas:
            if cfg.get("desfases"):
                S = cargar({**cfg, "paso": 1}, caja)
                pares = [(t, t + k) for t in range(len(S) - k)]
            else:
                S = cargar(cfg, caja)
                pares = [(t, t + 1) for t in range(len(S) - 1)]
            if por_caja < len(pares):
                b = np.linspace(0, len(pares), por_caja + 1).astype(int)
                pares = [pares[rng.integers(x, max(x + 1, y))] for x, y in zip(b[:-1], b[1:])]
            X = np.stack([S[[a, c]] for a, c in pares])
            if cfg.get("aumento"):
                X = np.stack([d4(x, int(rng.integers(8))) for x in X])
            np.save(carpeta / f"{caja}.npy", X)
        (carpeta / "listo").touch()
    return sorted(carpeta.glob("physarum-*.npy")), por_caja


# ── Entrenamiento ──────────────────────────────────────────────────────────────────────────────────────────────

def capas(cfg):
    k, n = cfg["capas"].split("x")
    return [int(n)] * int(k)


def entrenar(nombre, cfg, archivos):
    salida = CARPETA / "redes" / nombre
    salida.mkdir(parents=True, exist_ok=True)
    n_estados, _ = info(cfg)
    nb = VECINDADES[cfg["vecindad"]]
    cmd = [PY, "-u", str(MAIN), "--data", *map(str, archivos), "--output", str(salida / "red.npz"),
           "--neighbourhood", " ".join(f"{dx},{dy}" for dx, dy in nb),
           "--layers", " ".join(f"{c}:relu" for c in capas(cfg)), "--out-size", str(n_estados), "--out-act", "linear",
           "--epochs", str(cfg["epocas"]), "--batch", "8192", "--lr", "1e-3", "--lr-schedule", "exp",
           "--lr-min", "1e-5", "--loss-target", "1e-9", "--print-every", "1", "--seed", str(cfg["semilla"]),
           "--open-boundary", "--log-csv", str(salida / "perdida.csv")]
    if cfg["one_hot"]:
        cmd += ["--one-hot", str(n_estados)]
    t0 = time.perf_counter()
    with open(salida / "entrenamiento.log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=TRAINING)
    if r.returncode != 0:
        raise RuntimeError(f"falló main.py en {nombre}")
    texto = (salida / "entrenamiento.log").read_text()
    muestras = int(texto.split("Muestras: ")[1].split()[0].replace(",", ""))
    filas = list(csv.DictReader(open(salida / "perdida.csv")))
    return {"perdida": float(filas[-1]["loss"]), "muestras": muestras, "t_entrenar_s": time.perf_counter() - t0}


# ── Evaluación (frontera abierta, argmax, como el núcleo de la GUI) ────────────────────────────────────────────

class Red:
    def __init__(self, ruta, dev, pl):
        self.red, nb = cargar_red(str(ruta), dev)
        self.nb = [tuple(map(int, p)) for p in nb]
        self.r = max(max(abs(dx), abs(dy)) for dx, dy in self.nb)
        self.pl = pl

    @torch.no_grad()
    def logits(self, E):
        """E (B, H, W) int8 → (B, H, W, estados)."""
        B, H, W = E.shape
        r = self.r
        Pd = torch.nn.functional.pad(E, (r, r, r, r))
        X = torch.stack([Pd[:, r + dy:r + dy + H, r + dx:r + dx + W] for dx, dy in self.nb], -1).reshape(-1, len(self.nb))
        return torch.cat([self.red(X[s:s + (1 << 18)].float())                     # float: entrada lineal
                          for s in range(0, len(X), 1 << 18)]).view(B, H, W, -1)

    def vectores(self, sesgos, C):
        """Escalares (b a las salidas de plasmodio) o vectores (uno por estado) → (B, C)."""
        s = np.asarray(sesgos, np.float32)
        if s.ndim == 1:
            v = np.zeros((len(s), C), np.float32)
            v[:, self.pl:] = s[:, None]
            s = v
        return s

    @torch.no_grad()
    def libre(self, S0, T, sesgos):
        """Simulación libre con argmax desde S0 (H, W), una por sesgo (escalar o vector por estado)
        → (len(sesgos), T, H, W) int8 en la GPU."""
        E = S0.unsqueeze(0).repeat(len(sesgos), 1, 1)
        b = None
        ev = [E]
        for _ in range(1, T):
            L = self.logits(E)
            if b is None:
                b = torch.from_numpy(self.vectores(sesgos, L.shape[-1])).to(L.device).view(len(sesgos), 1, 1, -1)
            E = (L + b).argmax(-1).to(torch.int8)
            ev.append(E)
        return torch.stack(ev, 1)


def iou_t(real, sim, pl):
    """IoU de la máscara de plasmodio por paso: real (T, H, W), sim (B, T, H, W) → (B, T)."""
    a, b = (real >= pl).unsqueeze(0), sim >= pl
    inter, union = (a & b).flatten(2).sum(-1).float(), (a | b).flatten(2).sum(-1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def puntaje_t(real, sim, pl):
    """IoU × (área menor / área mayor) por paso → (B, T). Congelarse o llenar la caja dan ≈ 0.06 en prueba."""
    a = (real >= pl).flatten(1).sum(-1).float().unsqueeze(0)
    b = (sim >= pl).flatten(2).sum(-1).float()
    r = torch.where(torch.maximum(a, b) > 0, torch.minimum(a, b) / torch.maximum(a, b).clamp(min=1),
                    torch.ones_like(b))
    return iou_t(real, sim, pl) * r


def calibrar(red, cfg, dev, pl):
    def puntaje(sesgos):
        tot = np.zeros(len(sesgos))
        for caja in CALIBRACION:
            S = torch.from_numpy(cargar(cfg, caja)).to(dev)
            tot += puntaje_t(S, red.libre(S[0], len(S), sesgos), pl)[:, 1:].mean(1).cpu().numpy()
        return tot / len(CALIBRACION)
    p = puntaje(SESGOS)
    b0 = SESGOS[p.argmax()]
    finos = np.array([b0 + d for d in (-0.375, -0.25, -0.125, 0.125, 0.25, 0.375) if b0 + d >= 0])
    pf = puntaje(finos)
    todos = np.concatenate([SESGOS, finos])
    pt = np.concatenate([p, pf])
    return float(todos[pt.argmax()]), float(pt.max()), float(p[0])


def puntaje_vectores(red, cfg, dev, pl, V):
    tot = np.zeros(len(V))
    for caja in CALIBRACION:
        S = torch.from_numpy(cargar(cfg, caja)).to(dev)
        tot += puntaje_t(S, red.libre(S[0], len(S), V), pl)[:, 1:].mean(1).cpu().numpy()
    return tot / len(CALIBRACION)


def calibrar_vector(red, cfg, dev, pl, b, rondas=((1.5, 0.75), (0.375, 0.1875))):
    """Sesgo por estado (búsqueda coordenada desde el escalar b): para cada estado ≠ borde prueba ±d y se queda con
    lo mejor; dos rondas, la segunda con pasos menores."""
    C, _ = info(cfg)
    v = red.vectores([b], C)[0]
    mejor = puntaje_vectores(red, cfg, dev, pl, [v])[0]
    for pasos in rondas:
        for c in range(1, C):
            cands = []
            for d in pasos:
                for sgn in (-1, 1):
                    w = v.copy()
                    w[c] += sgn * d
                    cands.append(w)
            p = puntaje_vectores(red, cfg, dev, pl, cands)
            if p.max() > mejor:
                mejor, v = float(p.max()), cands[int(p.argmax())]
    return v, mejor


@torch.no_grad()
def probar(red, nombre, cfg, dev, pl, b):
    salida = CARPETA / "redes" / nombre
    horas_paso = 0.5 * cfg["paso"]
    m = {"puntaje_b0": [], "puntaje": [], "iou_b0": [], "iou": [], "area_real_fin": [], "area_red_fin": [],
         "error_area": []}
    for h in HORAS:
        m[f"iou_{h}h"] = []
    m["iou_fin"] = []
    uno = {k: 0.0 for k in ("n", "ok", "copia", "nuevo", "nuevo_ok", "queda", "falso")}
    bt = torch.from_numpy(b).to(dev)
    for caja in PRUEBA:
        S = torch.from_numpy(cargar(cfg, caja)).to(dev)
        dentro = S[0] != 0
        n = dentro.sum().item()
        L = red.libre(S[0], len(S), np.stack([np.zeros_like(b), b]))
        I = iou_t(S, L, pl).cpu().numpy()
        P = puntaje_t(S, L, pl).cpu().numpy()
        m["puntaje_b0"].append(P[0, 1:].mean())
        m["puntaje"].append(P[1, 1:].mean())
        np.save(salida / f"libre_{caja}.npy", L[1].cpu().numpy())
        m["iou_b0"].append(I[0, 1:].mean())
        m["iou"].append(I[1, 1:].mean())
        for h in HORAS:
            k = int(round(h / horas_paso))
            m[f"iou_{h}h"].append(I[1, min(k, len(S) - 1)])
        m["iou_fin"].append(I[1, -1])
        a_real = ((S >= pl) & dentro).flatten(1).sum(-1).float().cpu().numpy() / n
        a_red = ((L[1] >= pl) & dentro).flatten(1).sum(-1).float().cpu().numpy() / n
        m["area_real_fin"].append(a_real[-1])
        m["area_red_fin"].append(a_red[-1])
        m["error_area"].append(np.abs(a_red - a_real).mean())
        for t in range(len(S) - 1):                                   # un paso desde la imagen real, con b
            lg = red.logits(S[t:t + 1])[0] + bt
            p = lg.argmax(-1)[dentro]
            x, y = S[t][dentro], S[t + 1][dentro]
            nuevo = (x < pl) & (y >= pl)
            queda = (x < pl) & (y < pl)
            uno["n"] += len(y)
            uno["ok"] += (p == y).sum().item()
            uno["copia"] += (x == y).sum().item()
            uno["nuevo"] += nuevo.sum().item()
            uno["nuevo_ok"] += (p[nuevo] >= pl).sum().item()
            uno["queda"] += queda.sum().item()
            uno["falso"] += (p[queda] >= pl).sum().item()
    fila = {k: float(np.mean(v)) for k, v in m.items()}
    fila.update({"un_paso": uno["ok"] / uno["n"], "copia": uno["copia"] / uno["n"],
                 "nuevo_detectado": uno["nuevo_ok"] / max(uno["nuevo"], 1),
                 "falso_plasmodio": uno["falso"] / max(uno["queda"], 1)})
    return fila


def red_gui(nombre, b):
    """red.npz con el vector b (uno por estado) sumado al sesgo de la última capa → red_gui.npz."""
    salida = CARPETA / "redes" / nombre
    w = dict(np.load(salida / "red.npz"))
    ult = len(w["layer_out"]) - 1
    w[f"biases_{ult}"] = (w[f"biases_{ult}"] + b).astype(np.float32)
    np.savez(salida / "red_gui.npz", **w)


# ── Ciclo ──────────────────────────────────────────────────────────────────────────────────────────────────────

CAMPOS = ["nombre", "fase", "ca", "mapa", "estados", "vecindad", "vecinos", "capas", "one_hot", "paso_min", "epocas",
          "semilla", "por_caja", "muestras", "perdida", "t_entrenar_s", "t_evaluar_s", "vector", "sesgo",
          "calibracion", "calibracion_b0", "puntaje_b0", "puntaje", "iou_b0", "iou", "iou_6h", "iou_12h", "iou_24h", "iou_48h", "iou_fin",
          "area_real_fin", "area_red_fin", "error_area", "un_paso", "copia", "nuevo_detectado", "falso_plasmodio"]


def hechas():
    if not RESULTADOS.exists():
        return set()
    return {f["nombre"] for f in csv.DictReader(open(RESULTADOS))}


def registrar(msg):
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with open(REGISTRO, "a") as f:
        f.write(linea + "\n")


def correr(nombre, cfg, fase):
    n_estados, _ = info(cfg)
    archivos, por_caja = datos(cfg)
    registrar(f"{nombre}: entrenando ({n_estados} estados, {cfg['vecindad']}, {cfg['capas']}, "
              f"{'one-hot' if cfg['one_hot'] else 'lineal'}, paso {30 * cfg['paso']} min, {por_caja}/caja)")
    ent = entrenar(nombre, cfg, archivos)
    evaluar(nombre, cfg, fase, ent, por_caja)


def evaluar(nombre, cfg, fase, ent, por_caja):
    dev = torch.device("cuda")
    n_estados, pl = info(cfg)
    t0 = time.perf_counter()
    red = Red(CARPETA / "redes" / nombre / "red.npz", dev, pl)
    b, iou_cal, iou_cal0 = calibrar(red, cfg, dev, pl)
    vec = red.vectores([b], n_estados)[0]
    if cfg.get("vector"):
        vec, iou_cal = calibrar_vector(red, cfg, dev, pl, b)
    fila = probar(red, nombre, cfg, dev, pl, vec)
    red_gui(nombre, vec)
    (CARPETA / "redes" / nombre / "sesgo.json").write_text(json.dumps({"escalar": b, "vector": vec.tolist()}))
    fila.update(ent)
    fila.update({"nombre": nombre, "fase": fase, "ca": cfg["ca"], "mapa": cfg["mapa"] or "", "estados": n_estados,
                 "vecindad": cfg["vecindad"], "vecinos": len(VECINDADES[cfg["vecindad"]]), "capas": cfg["capas"],
                 "one_hot": int(cfg["one_hot"]), "paso_min": 30 * cfg["paso"], "epocas": cfg["epocas"],
                 "semilla": cfg["semilla"], "por_caja": por_caja, "t_evaluar_s": time.perf_counter() - t0,
                 "vector": int(bool(cfg.get("vector"))), "sesgo": b, "calibracion": iou_cal,
                 "calibracion_b0": iou_cal0})
    nuevo = not RESULTADOS.exists()
    with open(RESULTADOS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in fila.items() if k in CAMPOS})
    (CARPETA / "redes" / nombre / "config.json").write_text(json.dumps(cfg, indent=1))
    registrar(f"{nombre}: pérdida {ent['perdida']:.4f}, sesgo {b:g}, puntaje prueba {fila['puntaje']:.3f} (b=0: "
              f"{fila['puntaje_b0']:.3f}), IoU {fila['iou']:.3f}, área final red/real "
              f"{fila['area_red_fin']:.3f}/{fila['area_real_fin']:.3f}, "
              f"{ent['t_entrenar_s'] / 60:.1f} + {fila['t_evaluar_s'] / 60:.1f} min")
    del red
    torch.cuda.empty_cache()


def reevaluar():
    """Calibra y evalúa con el criterio actual (y las cajas de CALIBRACION) las redes de la fase 1."""
    viejas = list(csv.DictReader(open(CARPETA / "resultados.csv")))
    con_fila = {f["nombre"] for f in viejas}
    for nombre, cfg in FASES["fase1"]:                                 # entrenadas pero sin evaluar (p. ej. lineal)
        red = CARPETA / "redes" / nombre
        if nombre not in con_fila and (red / "red.npz").exists():
            (red / "config.json").write_text(json.dumps(cfg, indent=1))
            texto = (red / "entrenamiento.log").read_text()
            filas = list(csv.DictReader(open(red / "perdida.csv")))
            viejas.append({"nombre": nombre, "fase": "fase1", "perdida": filas[-1]["loss"],
                           "muestras": texto.split("Muestras: ")[1].split()[0].replace(",", ""),
                           "t_entrenar_s": "nan", "por_caja": datos(cfg)[1]})
    ya = hechas()
    registrar(f"== reevaluar: {len(viejas)} redes de la fase 1")
    for f in viejas:
        if f["nombre"] in ya:
            continue
        cfg = {**BASE, **json.loads((CARPETA / "redes" / f["nombre"] / "config.json").read_text())}
        ent = {"perdida": float(f["perdida"]), "muestras": int(f["muestras"]), "t_entrenar_s": float(f["t_entrenar_s"])}
        try:
            evaluar(f["nombre"], cfg, f["fase"], ent, int(f["por_caja"]))
        except Exception as e:
            registrar(f"{f['nombre']}: ERROR {e!r}")
    registrar("== reevaluar terminada")


if __name__ == "__main__":
    if sys.argv[1] == "reevaluar":
        reevaluar()
        sys.exit()
    fase = sys.argv[1]
    lista = FASES[fase]
    ya = hechas()
    registrar(f"== {fase}: {len(lista)} configuraciones, {sum(n in ya for n, _ in lista)} ya hechas")
    for nombre, cfg in lista:
        if nombre in ya:
            continue
        try:
            correr(nombre, cfg, fase)
        except Exception as e:                                         # una falla no detiene el barrido
            registrar(f"{nombre}: ERROR {e!r}")
    registrar(f"== {fase} terminada")
