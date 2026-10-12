"""Barrido de hormigas de Langton con 2 a 10 colores (dos hormigas con exclusión, 30 × 30, 10 000 pasos).

Para cada regla de CA/LangtonAnt/colores_30x30_2hormigas/<regla>/ (generadas con notebooks/langton.nb):
  1. verifica la evolución contra una simulación independiente y que la regla sea determinista con el rombo de radio 2;
  2. entrena 4:64:ReLU & 5n:Linear con one-hot (5n estados), tasa dinámica (plateau) y pérdida objetivo 1e-7;
  3. evalúa la exactitud a un paso y la evolución en lazo cerrado contra la regla exacta.
Escribe NCA/LangtonAnt/colores_30x30_2hormigas/resultados.csv.

Uso (desde Simulator/training):  ../GUI/venv/bin/python scripts/barrido_langton_colores.py [regla ...]
"""
import csv, json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

DATOS = Path("CA/LangtonAnt/colores_30x30_2hormigas")
SALIDA = Path("NCA/LangtonAnt/colores_30x30_2hormigas")
REGLAS = ["RL", "RLL", "RLLR", "RLRLL", "RRRLLL", "RLRLRLL", "RLRLRLRR", "RLRLRLLRL", "RLRLRLRRRL"]
ROMBO = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3) if abs(dx) + abs(dy) <= 2]
CAPAS, EPOCAS, LOTE, LR, OBJETIVO, SEMILLA = "64:relu 64:relu 64:relu 64:relu", 2000, 4096, 0.001, 1e-7, 42
PASOS_NUEVAS, N_NUEVAS = 3000, 5
MOV = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GIRO = {"R": 1, "L": -1, "N": 0, "U": 2}


def exacta(regla, L, ants, pasos):
    """Misma regla que langton.nb: síncrona, una hormiga bloqueada solo gira. ants 0-indexado."""
    n = len(regla); gs = [GIRO[c] for c in regla]
    g = np.zeros((L, L), np.int64); ants = [list(a) for a in ants]
    def frame():
        f = g.copy()
        for r, c, d in ants: f[r, c] = g[r, c] + n * (d + 1)
        return f
    evo = [frame()]; bloqueos = []
    for k in range(1, pasos + 1):
        for a in ants: a[2] = (a[2] + gs[g[a[0], a[1]]]) % 4
        t = [((a[0] + MOV[a[2]][0]) % L, (a[1] + MOV[a[2]][1]) % L) for a in ants]
        occ = [(a[0], a[1]) for a in ants]
        blk = []
        for i in range(len(ants)):
            ocup, mismo = t[i] in occ[:i] + occ[i + 1:], t[i] in t[:i] + t[i + 1:]
            blk.append(ocup or mismo)
            if ocup or mismo: bloqueos.append((k, "ocupado" if ocup else "mismo destino"))
        for i, a in enumerate(ants):
            if not blk[i]: g[a[0], a[1]] = (g[a[0], a[1]] + 1) % n; a[0], a[1] = t[i]
        evo.append(frame())
    return np.array(evo), bloqueos


def patrones_ambiguos(e):
    X, Y = e[:-1], e[1:].ravel()
    clave = np.zeros(X.shape, np.int64); base = int(e.max()) + 1
    for dx, dy in ROMBO: clave = clave * base + np.roll(X, (-dy, -dx), (1, 2))
    tabla = defaultdict(set)
    for k, v in zip(clave.ravel().tolist(), Y.tolist()): tabla[k].add(v)
    return len(tabla), sum(len(v) > 1 for v in tabla.values())


class Red:
    def __init__(self, ruta):
        w = np.load(ruta)
        self.w, self.nb, self.S = w, w["neighbourhood"], int(w["one_hot_states"])
        self.acts, self.capas = w["activations"], len(w["layer_in"])

    def paso(self, f):
        cols = [np.roll(np.roll(f, -int(dy), 0), -int(dx), 1).ravel() for dx, dy in self.nb]
        W0 = self.w["weights_0"]
        h = sum(W0[:, i * self.S + c].T for i, c in enumerate(cols)) + self.w["biases_0"]
        for k in range(self.capas):
            if k > 0: h = h @ self.w[f"weights_{k}"].T + self.w[f"biases_{k}"]
            if self.acts[k] == 0: h = np.maximum(h, 0)
            elif self.acts[k] == 1: h = 1 / (1 + np.exp(-h))
            elif self.acts[k] == 2: h = np.tanh(h)
        return h.argmax(1).reshape(f.shape)


def lazo(red, ref):
    f = ref[0]
    for k in range(1, len(ref)):
        f = red.paso(f)
        if not (f == ref[k]).all(): return k
    return None


def experimento(regla):
    n = len(regla); estados = 5 * n
    d = DATOS / regla; s = SALIDA / regla; s.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / "evo.json").read_text())
    e = np.load(d / "evo.npy")
    ants = [(a["row"] - 1, a["col"] - 1, a["direction"]) for a in meta["ants_start"]]
    ref, bl = exacta(regla, meta["size"], ants, meta["steps"])
    verificada = bool((ref == e).all())
    patrones, ambiguos = patrones_ambiguos(e)
    print(f"[{regla}] verificada={verificada} patrones={patrones} ambiguos={ambiguos} bloqueos={len(bl)}", flush=True)

    base = s / f"4-64-relu_{estados}-linear"
    cmd = [sys.executable, "scripts/main.py", "--data", str(d / "evo.npy"),
           "--neighbourhood", " ".join(f"{dx},{dy}" for dx, dy in ROMBO),
           "--layers", CAPAS, "--out-size", str(estados), "--out-act", "linear", "--one-hot", str(estados),
           "--epochs", str(EPOCAS), "--batch", str(LOTE), "--lr", str(LR), "--lr-schedule", "plateau",
           "--loss-target", str(OBJETIVO), "--seed", str(SEMILLA), "--print-every", "1",
           "--log-csv", f"{base}.csv", "--output", f"{base}.npz"]
    t0 = time.time()
    with open(s / "entrenar.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    segundos = time.time() - t0
    curva = list(csv.DictReader(open(f"{base}.csv")))
    # main.py repite la última época al converger: se cuenta por su número
    epocas, perdida, lr_final = int(curva[-1]["epoch"]) + 1, float(curva[-1]["loss"]), float(curva[-1]["lr"])

    red = Red(f"{base}.npz")
    pred = np.array([red.paso(f) for f in e[:-1]]); real = e[1:]
    hormiga = (real >= n) | (e[:-1] >= n)
    errores = int((pred != real).sum())
    pasos_bl = sorted({k for k, _ in bl})
    errores_bl = int(sum((pred[k - 1] != real[k - 1]).sum() for k in pasos_bl))
    lazo_ent = lazo(red, e)

    rng = np.random.default_rng(7); nuevas = []
    for _ in range(N_NUEVAS):
        a = []
        while len(a) < 2:
            x = (int(rng.integers(30)), int(rng.integers(30)), int(rng.integers(4)))
            if all(x[:2] != y[:2] for y in a): a.append(x)
        r, b = exacta(regla, 30, a, PASOS_NUEVAS)
        nuevas.append((a, lazo(red, r), len(b), sum(1 for _, t in b if t == "ocupado")))
    fila = dict(regla=regla, colores=n, estados=estados, verificada=verificada, patrones=patrones, ambiguos=ambiguos,
                bloqueos=len(bl), epocas=epocas, perdida_final=perdida, lr_final=lr_final, segundos=round(segundos, 1),
                exactitud=1 - errores / real.size, errores=errores,
                exactitud_hormiga=float((pred[hormiga] == real[hormiga]).mean()), errores_en_bloqueos=errores_bl,
                lazo_entrenamiento=lazo_ent if lazo_ent else "sin errores",
                nuevas_sin_errores=sum(1 for x in nuevas if x[1] is None),
                nuevas_primer_error=";".join(str(x[1]) if x[1] else "-" for x in nuevas),
                nuevas_bloqueos=";".join(str(x[2]) for x in nuevas),
                nuevas_bloqueos_ocupado=";".join(str(x[3]) for x in nuevas),
                nuevas_posiciones=";".join(str([(r + 1, c + 1, dd) for r, c, dd in x[0]]) for x in nuevas))
    print(fila, flush=True)
    return fila


def main():
    reglas = sys.argv[1:] or REGLAS
    ruta = SALIDA / "resultados.csv"
    filas = [experimento(r) for r in reglas]
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    print(f"Resultados → {ruta}")


if __name__ == "__main__":
    main()
