"""Barrido de vecindades para B3/S23 (sección 1.2 del experimento 1).

Para cada vecindad entrena las 4 configuraciones de activación × 6 tamaños de red (capas:neuronas)
con los mismos hiperparámetros, genera las evoluciones con el simulador y escribe la tabla LaTeX.

Uso (desde Simulator/training):
    python scripts/barrido_vecindad.py entrenar centro 3x3 5x5 7x7 8x8
    python scripts/barrido_vecindad.py evoluciones NO N centro 3x3 5x5 7x7 8x8
    python scripts/barrido_vecindad.py referencia
    python scripts/barrido_vecindad.py tablas
"""
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io

PY = "../GUI/venv/bin/python"
DATA = "CA/B3S23/evo.mat"
GEN_EVOS = str(Path(__file__).with_name("generar_evoluciones.py"))

# Hiperparámetros elegidos con la prueba corta (ver NCA/B3S23_NR/ajuste_hiperparametros/)
LR, LOTE, EPOCAS, LOSS_TARGET, SEED, HILOS = 0.001, 4096, 500, 0.001, 42, 6

# (activación oculta, neuronas de salida, activación de salida)
CONFIGS = [("relu", 2, "linear"), ("tanh", 1, "sigmoid"), ("tanh", 2, "linear"), ("relu", 1, "sigmoid")]
# capas:neuronas, misma notación que las tablas del documento
TAMANOS = [(1, 4), (2, 8), (4, 8), (4, 16), (8, 16), (16, 16)]

ACT_TXT = {"relu": "ReLU", "tanh": "TanH", "sigmoid": "Sigmoid", "linear": "Linear"}


def cuadrado(desde: int, hasta: int) -> list:
    """Desplazamientos (dx, dy) del cuadrado [desde, hasta]², dx = columna, dy = fila (dy < 0 es arriba)."""
    return [(dx, dy) for dy in range(desde, hasta + 1) for dx in range(desde, hasta + 1)]


def sin(celda: tuple) -> list:
    return [c for c in cuadrado(-1, 1) if c != celda]


EXPERIMENTOS = {
    "NO":     dict(vecindad=sin((-1, -1)), carpeta="NCA/B3S23_NR/1_01111111", etiqueta="Noroeste $(-1,-1)$"),
    "N":      dict(vecindad=sin((0, -1)), carpeta="NCA/B3S23_NR/1_10111111", etiqueta="Norte $(0,-1)$"),
    "centro": dict(vecindad=sin((0, 0)), carpeta="NCA/B3S23_NR/0_11111111_barrido", etiqueta="Centro $(0,0)$"),
    "3x3":    dict(vecindad=cuadrado(-1, 1), carpeta="NCA/Vecindad/3x3", etiqueta="$3\\times3$"),
    "5x5":    dict(vecindad=cuadrado(-2, 2), carpeta="NCA/Vecindad/5x5", etiqueta="$5\\times5$"),
    "7x7":    dict(vecindad=cuadrado(-3, 3), carpeta="NCA/Vecindad/7x7", etiqueta="$7\\times7$"),
    # El kernel admite hasta 64 vecinos (c_neighbourhood[128] en core/nca.cu): 8×8 es el máximo
    "8x8":    dict(vecindad=cuadrado(-4, 3), carpeta="NCA/Vecindad/8x8", etiqueta="$8\\times8$"),
}


def cargar_datos() -> np.ndarray:
    return scipy.io.loadmat(DATA)["evolution"].astype(np.int64)


def perdida_minima(datos: np.ndarray, vecindad: list) -> float:
    """Entropía condicional H(Y | patrón): la menor pérdida que cualquier red puede alcanzar con esa vecindad."""
    src, tgt = datos[:-1], datos[1:]
    codigo = np.zeros(src.shape, dtype=np.uint64)
    for i, (dx, dy) in enumerate(vecindad):
        codigo |= np.roll(np.roll(src, -dy, axis=1), -dx, axis=2).astype(np.uint64) << np.uint64(i)
    _, idx = np.unique(codigo.ravel(), return_inverse=True)
    n = np.bincount(idx).astype(float)
    p = np.bincount(idx, weights=tgt.ravel()) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nan_to_num(p * np.log(p)) - np.nan_to_num((1 - p) * np.log(1 - p))
    return float((n * h).sum() / n.sum())


def nombre(capas: int, neuronas: int, act: str, out_size: int, out_act: str) -> str:
    return f"{capas}:{neuronas}:{act}_{out_size}:{out_act}"


def entrenar(job: dict) -> dict:
    out = Path(job["salida"])
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".csv")
    capas, neuronas = job["capas"], job["neuronas"]
    cmd = [PY, "-W", "ignore", "scripts/main.py", "--data", DATA,
           "--neighbourhood", " ".join(f"{dx},{dy}" for dx, dy in EXPERIMENTOS[job["exp"]]["vecindad"]),
           "--layers", " ".join([f"{neuronas}:{job['act']}"] * capas),
           "--out-size", str(job["out_size"]), "--out-act", job["out_act"],
           "--epochs", str(EPOCAS), "--batch", str(LOTE), "--lr", str(LR),
           "--loss-target", str(LOSS_TARGET), "--seed", str(SEED), "--print-every", "1",
           "--output", str(out), "--log-csv", str(log)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    res = dict(job, segundos=time.perf_counter() - t0, error="")
    if r.returncode != 0 or not out.exists():
        res["error"] = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"código {r.returncode}"
        print(f"ERROR {out}: {res['error']}", flush=True)
        return res
    filas_log = list(csv.DictReader(open(log)))
    perdidas = [float(row["loss"]) for row in filas_log]
    # main.py repite la última época al converger: las épocas corridas son la última + 1
    res.update(final=perdidas[-1], minima=min(perdidas), epocas_corridas=int(filas_log[-1]["epoch"]) + 1)
    res["brecha"] = res["final"] - job["minimo"]
    print(f"{out}  final={res['final']:.6f}  brecha={res['brecha']:+.6f}  épocas={len(perdidas)}  "
          f"{res['segundos']:.0f}s", flush=True)
    return res


CAMPOS = ["exp", "carpeta", "capas", "neuronas", "act", "out_size", "out_act", "lr", "lote", "epocas",
          "epocas_corridas", "final", "minima", "minimo", "brecha", "segundos", "salida", "error"]


def cmd_entrenar(exps: list):
    datos = cargar_datos()
    jobs = []
    for exp in exps:
        e = EXPERIMENTOS[exp]
        minimo = perdida_minima(datos, e["vecindad"])
        print(f"{exp}: {len(e['vecindad'])} celdas, pérdida mínima alcanzable {minimo:.6f}", flush=True)
        for act, out_size, out_act in CONFIGS:
            for capas, neuronas in TAMANOS:
                jobs.append(dict(exp=exp, carpeta=e["carpeta"], capas=capas, neuronas=neuronas, act=act,
                                 out_size=out_size, out_act=out_act, lr=LR, lote=LOTE, epocas=EPOCAS,
                                 minimo=minimo,
                                 salida=f"{e['carpeta']}/{nombre(capas, neuronas, act, out_size, out_act)}.npz"))
    print(f"{len(jobs)} entrenamientos, {HILOS} en paralelo", flush=True)
    with ThreadPoolExecutor(HILOS) as ex:
        resultados = list(ex.map(entrenar, jobs))
    for exp in exps:
        ruta = Path(EXPERIMENTOS[exp]["carpeta"]) / "resultados.csv"
        with open(ruta, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
            w.writeheader()
            w.writerows(r for r in resultados if r["exp"] == exp)
        print(f"Resumen → {ruta}", flush=True)


def cmd_evoluciones(exps: list):
    for exp in exps:
        e = EXPERIMENTOS[exp]
        pesos = [str(Path(e["carpeta"]) / f"{nombre(c, n, a, o, oa)}.npz")
                 for a, o, oa in CONFIGS for c, n in TAMANOS]
        r = subprocess.run([PY, "-W", "ignore", GEN_EVOS, *pesos], capture_output=True, text=True)
        estado = "ok" if r.returncode == 0 else f"ERROR {r.stderr.strip().splitlines()[-1]}"
        print(f"evoluciones {exp}: {len(pesos)} modelos {estado}", flush=True)


def b3s23(estado: np.ndarray) -> np.ndarray:
    vecinos = sum(np.roll(np.roll(estado, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - estado
    return ((vecinos == 3) | ((estado == 1) & (vecinos == 2))).astype(np.int32)


def cmd_referencia():
    """Evoluciones de B3/S23 real desde las mismas condiciones iniciales aleatorias que las NCA."""
    sys.path.insert(0, str(Path(GEN_EVOS).parent))
    from generar_evoluciones import PASOS, condiciones_aleatorias, guardar
    for i, inicial in enumerate(condiciones_aleatorias()):
        cuadros = [inicial]
        for _ in range(PASOS):
            cuadros.append(b3s23(cuadros[-1]))
        guardar(Path("CA/B3S23/aleatorias") / f"test_{i}", np.stack(cuadros).astype(np.int32))
    print("referencia → CA/B3S23/aleatorias/test_*.npy|mat", flush=True)


def leer(exp: str) -> list:
    return list(csv.DictReader(open(Path(EXPERIMENTOS[exp]["carpeta"]) / "resultados.csv")))


def filas(exp: str, primera: str, con_epocas: bool) -> list:
    orden = {(a, o, oa): i for i, (a, o, oa) in enumerate(CONFIGS)}
    rs = sorted(leer(exp), key=lambda r: (orden[(r["act"], int(r["out_size"]), r["out_act"])],
                                          TAMANOS.index((int(r["capas"]), int(r["neuronas"])))))
    out = []
    for r in rs:
        oculta = f"{r['capas']}:{r['neuronas']}:{ACT_TXT[r['act']]}"
        salida = f"{r['out_size']}:{ACT_TXT[r['out_act']]}"
        perdida = f"{float(r['final']):.6f}" if not r.get("error") else "error"
        epocas = f" & {r['epocas_corridas']}" if con_epocas else ""
        out.append(f"\t\t{primera} & {oculta} & {salida}{epocas} & {perdida} \\\\")
    return out


def tabla(exps: list, columna: str, con_epocas: bool, titulo: str, nota: str, label: str) -> str:
    cuerpo = []
    for i, exp in enumerate(exps):
        if i:
            cuerpo.append("\t\t\\midrule")
        cuerpo += filas(exp, EXPERIMENTOS[exp]["etiqueta"], con_epocas)
    cols = "lllrr" if con_epocas else "lllr"
    enc = (f"\\textbf{{{columna}}} & \\textbf{{Capa Oculta}} & \\textbf{{Capa Salida}}"
           + (" & \\textbf{Épocas}" if con_epocas else "") + " & \\textbf{Pérdida} \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\t\\centering", f"\t\\begin{{tabular}}{{{cols}}}", "\t\t\\toprule", f"\t\t{enc}",
        "\t\t\\midrule", *cuerpo, "\t\t\\bottomrule", "\t\\end{tabular}",
        f"\t\\caption{{{titulo}: Pérdida objetivo {LOSS_TARGET}, límite de épocas: {EPOCAS}, "
        f"tamaño del lote: {LOTE}, tasa de aprendizaje: {LR}, semilla aleatoria: {SEED}.{nota}}}",
        f"\t\\label{{{label}}}", "\\end{table}", ""])


def minimo(exp: str) -> str:
    return f"{float(leer(exp)[0]['minimo']):.6f}"


def cmd_tablas():
    salidas = {
        "NCA/B3S23_NR/tabla_borde.tex": tabla(
            ["NO", "N"], "Vecino eliminado", False,
            "Comparativa de rendimiento con múltiples hiperparámetros de la regla B3/S23 sin un vecino del borde",
            f" Pérdida mínima alcanzable: {minimo('NO')} sin la esquina y {minimo('N')} sin el lado.",
            "tab:nca_gol_reduction_borde"),
        "NCA/B3S23_NR/tabla_centro.tex": tabla(
            ["centro"], "Vecino eliminado", False,
            "Comparativa de rendimiento con múltiples hiperparámetros de la regla B3/S23 sin la célula central",
            f" Pérdida mínima alcanzable: {minimo('centro')}.", "tab:nca_gol_reduction_centro"),
        "NCA/Vecindad/tabla.tex": tabla(
            ["3x3", "5x5", "7x7", "8x8"], "Vecindad", True,
            "Comparativa de rendimiento con múltiples hiperparámetros de la regla B3/S23 al aumentar la vecindad",
            " La vecindad de $8\\times8$ abarca los desplazamientos $-4$ a $3$ en cada eje.", "tab:nca_gol_vecindad"),
    }
    for ruta, texto in salidas.items():
        Path(ruta).write_text(texto)
        print(f"tabla → {ruta}", flush=True)


if __name__ == "__main__":
    accion, args = sys.argv[1], sys.argv[2:]
    {"entrenar": cmd_entrenar, "evoluciones": cmd_evoluciones}.get(accion, lambda a: None)(args)
    if accion == "referencia":
        cmd_referencia()
    if accion == "tablas":
        cmd_tablas()
