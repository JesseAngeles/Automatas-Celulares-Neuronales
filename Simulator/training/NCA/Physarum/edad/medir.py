"""Compara redes de barrido/ en las 4 cajas de prueba con las medidas de tabla.py, usando sus simulaciones libres
(barrido/redes/<red>/libre_<caja>.npy, argmax con el sesgo calibrado).

- C: IoU × área menor / área mayor del plasmodio, promedio desde el primer paso (el puntaje de barrido.py)
- area_fin: fracción de la caja con plasmodio al final (y por caja)
- borde: fracción del plasmodio que toca algo que no es plasmodio, promedio de la segunda mitad
- brillo: fracción del plasmodio que cambia de nivel entre pasos, segunda mitad (solo con 3 niveles)

Los estados con edad se pasan a su nivel visible antes de medir.

    python NCA/Physarum/edad/medir.py salida.json red1 red2 ...   (grupos: red, red_s43, red_s44 se promedian)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "barrido"))
import barrido as B                                                     # noqa: E402

PL = 3


def frontera(X):
    p = X >= PL
    q = torch.nn.functional.pad(p, (1, 1, 1, 1))
    interior = q[..., :-2, 1:-1] & q[..., 2:, 1:-1] & q[..., 1:-1, :-2] & q[..., 1:-1, 2:]
    n = p.flatten(-2).sum(-1).float()
    return torch.where(n > 0, (p & ~interior).flatten(-2).sum(-1).float() / n.clamp(min=1),
                       torch.full_like(n, float("nan")))


def interno(X):
    a, b = X[:-1], X[1:]
    m = (a >= PL) & (b >= PL)
    n = m.flatten(-2).sum(-1).float()
    return torch.where(n > 0, (m & (a != b)).flatten(-2).sum(-1).float() / n.clamp(min=1),
                       torch.full_like(n, float("nan")))


def visible(S, cfg):
    if not cfg.get("edad"):
        return S
    n, pl = B.info({**cfg, "edad": False})
    return torch.where(S >= n, S - (n - pl), S)


@torch.no_grad()
def medir_red(nombre, dev):
    cfg = {**B.BASE, **json.loads((B.CARPETA / "redes" / nombre / "config.json").read_text())}
    n_vis, _ = B.info({**cfg, "edad": False})
    filas = []
    for caja in B.PRUEBA:
        real = visible(torch.from_numpy(B.cargar(cfg, caja)).to(dev), cfg)
        sim = visible(torch.from_numpy(np.load(B.CARPETA / "redes" / nombre / f"libre_{caja}.npy")).to(dev), cfg)
        d = real[0] != 0
        m = slice(len(real) // 2, len(real))
        f = {"C": B.puntaje_t(real, sim.unsqueeze(0), PL)[0, 1:].mean().item(),
             "area_fin": ((sim[-1] >= PL) & d).sum().item() / d.sum().item(),
             "area_real": ((real[-1] >= PL) & d).sum().item() / d.sum().item(),
             "borde": torch.nanmean(frontera(sim[m])).item(),
             "borde_real": torch.nanmean(frontera(real[m])).item()}
        if n_vis - PL == 3:
            f["brillo"] = torch.nanmean(interno(sim[m])).item()
            f["brillo_real"] = torch.nanmean(interno(real[m])).item()
        filas.append(f)
    return {"por_caja": filas, **{k: float(np.mean([f[k] for f in filas])) for k in filas[0]},
            "area_001": filas[0]["area_fin"], "area_001_real": filas[0]["area_real"]}


def main():
    salida = Path(sys.argv[1])
    dev = torch.device("cuda")
    res = {n: medir_red(n, dev) for n in sys.argv[2:] if (B.CARPETA / "redes" / n / "libre_" f"{B.PRUEBA[0]}.npy").exists()}
    grupos = {}
    for n in res:
        base = n.rsplit("_s4", 1)[0] if n.rsplit("_s4", 1)[-1] in ("3", "4") else n
        grupos.setdefault(base, []).append(n)
    resumen = {}
    for g, ns in grupos.items():
        resumen[g] = {k: (float(np.mean([res[n][k] for n in ns])), float(np.std([res[n][k] for n in ns])))
                      for k in ("C", "area_fin", "area_001", "borde", "brillo") if k in res[ns[0]]}
        resumen[g]["semillas"] = len(ns)
    salida.write_text(json.dumps({"redes": res, "grupos": resumen}, indent=1))
    r0 = next(iter(res.values()))
    print(f"real: área final {r0['area_real']:.3f}, área 001 {r0['area_001_real']:.3f}, borde {r0['borde_real']:.3f}"
          + (f", brillo {r0['brillo_real']:.3f}" if "brillo_real" in r0 else ""))
    print(f"{'red':32} {'sem':>3} {'C':>13} {'área final':>13} {'área 001':>13} {'borde':>13} {'brillo':>13}")
    for g, r in resumen.items():
        c = lambda k: f"{r[k][0]:.3f} ± {r[k][1]:.3f}" if k in r else "—"
        print(f"{g:32} {r['semillas']:>3} {c('C'):>13} {c('area_fin'):>13} {c('area_001'):>13} {c('borde'):>13} {c('brillo'):>13}")


if __name__ == "__main__":
    main()
