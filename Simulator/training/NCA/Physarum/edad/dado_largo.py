"""Experimento C del plan: red con ruido (dado oculto, dado/comun.py) cuya maestra es una red de paso largo con
edad de barrido/ (p. ej. e6_6h_circ6_da_edad), muestreada con un empujón de crecimiento.

1. Datos, por caja de entrenamiento: REALES imágenes reales (cualquier desfase de 30 min, con la edad relativa al
   paso de la maestra) con dados al azar → objetivo = paso_ideal; y PROPIOS pares de simulaciones libres del proceso
   ideal desde la primera imagen. → datos_dado/<clave>/<caja>.npy (N, 2, H, W) y vecindad.txt
2. Entrena la alumna con scripts/main.py (one-hot de visibles × 4 dados, frontera abierta, lr exp).
3. Evalúa como la GUI (argmax) desde la primera imagen con dados al azar, en las cajas de calibración y de prueba,
   con las medidas de medir.py. → redes_dado/<nombre>/evaluacion.json y libre_<caja>.npy

    python NCA/Physarum/edad/dado_largo.py <nombre> <maestra> <crecer> [reales=18] [propios=18] [semilla=42] [capas=16x128]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "dado"))
sys.path.insert(0, str(CARPETA.parent / "barrido"))
import comun as C                                                       # noqa: E402
import barrido as B                                                     # noqa: E402
from medir import frontera, interno, visible, PL                        # noqa: E402

REPS = 2


def preparar(maestra_nombre, crecer, dev):
    cfg = {**B.BASE, **json.loads((B.CARPETA / "redes" / maestra_nombre / "config.json").read_text())}
    C.NV, C.CRECER = B.info(cfg)[0], crecer
    C.NE = C.NV * C.DADOS
    return cfg, C.Red(B.CARPETA / "redes" / maestra_nombre / "red.npz", dev)


@torch.no_grad()
def datos(cfg, maestra, crecer, reales, propios, dev):
    clave = f"{maestra_nombre_de(cfg)}_c{crecer:g}_{reales}_{propios}"
    out = CARPETA / "datos_dado" / clave
    if (out / "listo").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    w = C.pesos_ruido(dev)
    g = torch.Generator(device=dev).manual_seed(1)
    rng = np.random.default_rng(1)
    k = cfg["paso"]
    for caja in B.cajas_entrenamiento(cfg):
        S30 = torch.from_numpy(B.cargar30(cfg, caja)).to(dev)
        S = S30[::k]
        ts = np.sort(rng.choice(len(S30) - 1, reales, replace=False))
        X = C.combinar(S30[ts], g, dev)
        Y = C.paso_ideal(maestra, X, w)
        corridas = -(-propios // (len(S) - 1))                         # las que hagan falta para PROPIOS pares
        L = C.libre_ideal(maestra, C.combinar(S[:1].repeat(corridas, 1, 1), g, dev), len(S), w)
        pares = [(c, t) for c in range(corridas) for t in range(len(S) - 1)]
        elegidos = [pares[i] for i in np.sort(rng.choice(len(pares), propios, replace=False))]
        Xp = torch.stack([L[c, t] for c, t in elegidos])
        Yp = torch.stack([L[c, t + 1] for c, t in elegidos])
        np.save(out / f"{caja}.npy", torch.stack([torch.cat([X, Xp]), torch.cat([Y, Yp])], 1).cpu().numpy())
    nb = C.vecindad_alumna(maestra)
    (out / "vecindad.txt").write_text(" ".join(f"{dx},{dy}" for dx, dy in nb))
    (out / "listo").touch()
    return out


def maestra_nombre_de(cfg):
    return cfg["_maestra"]


def entrenar(nombre, carpeta_datos, capas, semilla):
    n, ancho = map(int, capas.split("x"))
    out = CARPETA / "redes_dado" / nombre
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(C.TRAINING / "scripts" / "main.py"),
           "--data", *map(str, sorted(carpeta_datos.glob("physarum-*.npy"))), "--output", str(out / "red.npz"),
           "--neighbourhood", (carpeta_datos / "vecindad.txt").read_text(),
           "--layers", " ".join([f"{ancho}:relu"] * n), "--out-size", str(C.NE), "--out-act", "linear",
           "--one-hot", str(C.NE), "--epochs", "30", "--batch", "8192", "--lr", "1e-3", "--lr-schedule", "exp",
           "--lr-min", "1e-5", "--loss-target", "1e-9", "--print-every", "1", "--seed", str(semilla),
           "--open-boundary", "--log-csv", str(out / "perdida.csv")]
    with open(out / "entrenamiento.log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=C.TRAINING)
    if r.returncode:
        raise RuntimeError(f"falló main.py, ver {out / 'entrenamiento.log'}")
    return out


@torch.no_grad()
def evaluar(nombre, cfg, dev):
    carpeta = CARPETA / "redes_dado" / nombre
    red = C.Red(carpeta / "red.npz", dev)
    g = torch.Generator(device=dev).manual_seed(7)
    n_vis = B.info({**cfg, "edad": False})[0]
    res = {}
    for grupo, cajas in (("cal", B.CALIBRACION), ("prueba", B.PRUEBA)):
        filas = []
        for caja in cajas:
            real = torch.from_numpy(B.cargar(cfg, caja)).to(dev)
            E = C.combinar(real[:1].repeat(REPS, 1, 1), g, dev)
            ev = [E]
            for _ in range(1, len(real)):
                E = red.logits(E).argmax(-1).to(torch.int8)
                ev.append(E)
            L = torch.stack(ev, 1)
            if grupo == "prueba":
                np.save(carpeta / f"libre_{caja}.npy", L[0].cpu().numpy())
            sim, real = visible(L // C.DADOS, cfg), visible(real, cfg)
            d = real[0] != 0
            m = slice(len(real) // 2, len(real))
            f = {"C": B.puntaje_t(real, sim, PL)[:, 1:].mean().item(),
                 "area_fin": (((sim[:, -1] >= PL) & d).flatten(1).sum(1).float() / d.sum()).mean().item(),
                 "borde": torch.nanmean(frontera(sim[:, m])).item()}
            if n_vis - PL == 3:
                f["brillo"] = float(np.nanmean([torch.nanmean(interno(s[m])).item() for s in sim]))
            filas.append(f)
        res[grupo] = {"por_caja": filas, **{k: float(np.mean([f[k] for f in filas])) for k in filas[0]}}
    res["prueba"]["area_001"] = res["prueba"]["por_caja"][0]["area_fin"]
    (carpeta / "evaluacion.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    a = sys.argv[1:]
    nombre, maestra_nombre, crecer = a[0], a[1], float(a[2])
    reales, propios, semilla = (int(x) for x in (a[3:6] + ["18", "18", "42"][len(a[3:6]):]))
    capas = a[6] if len(a) > 6 else "16x128"
    dev = torch.device("cuda")
    cfg, maestra = preparar(maestra_nombre, crecer, dev)
    cfg["_maestra"] = maestra_nombre
    carpeta = datos(cfg, maestra, crecer, reales, propios, dev)
    del maestra
    torch.cuda.empty_cache()
    entrenar(nombre, carpeta, capas, semilla)
    r = evaluar(nombre, cfg, dev)
    p, c = r["prueba"], r["cal"]
    print(f"{nombre}: calibración C {c['C']:.3f} | prueba C {p['C']:.3f}, área final {p['area_fin']:.3f}, "
          f"área 001 {p['area_001']:.3f}, borde {p['borde']:.3f}" + (f", brillo {p['brillo']:.3f}" if "brillo" in p else ""),
          flush=True)


if __name__ == "__main__":
    main()
