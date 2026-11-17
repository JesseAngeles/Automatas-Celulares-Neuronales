"""Prueba previa del experimento C: el proceso ideal de la red con ruido (muestrear de una maestra con el azar
correlacionado de dado/comun.py) con maestras a paso largo, sin y con edad. La alumna solo imita este proceso, así
que si el proceso no mejora, la alumna tampoco.

Simulación libre desde la primera imagen de las 4 cajas de prueba, REPS dados al azar por caja; mismas medidas que
medir.py. Los estados con edad se pasan a su nivel visible para medir.

    python NCA/Physarum/edad/ideal.py salida.json red1 [red2 ...]     (redes de barrido/redes; sin sesgo)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "dado"))
sys.path.insert(0, str(CARPETA.parent / "barrido"))
import comun as C                                                       # noqa: E402
import barrido as B                                                     # noqa: E402
from medir import frontera, visible, PL                                 # noqa: E402

REPS = 4


@torch.no_grad()
def probar(nombre, dev, crecer=0.0):
    cfg = {**B.BASE, **json.loads((B.CARPETA / "redes" / nombre / "config.json").read_text())}
    n, _ = B.info(cfg)
    C.NV, C.CRECER = n, crecer
    maestra = C.Red(B.CARPETA / "redes" / nombre / "red.npz", dev)
    w = C.pesos_ruido(dev)
    g = torch.Generator(device=dev).manual_seed(7)
    filas = []
    for caja in B.PRUEBA:
        real = torch.from_numpy(B.cargar(cfg, caja)).to(dev)
        E0 = C.combinar(real[:1].repeat(REPS, 1, 1), g, dev)
        sim = visible(C.libre_ideal(maestra, E0, len(real), w) // C.DADOS, cfg)          # (REPS, T, H, W)
        real = visible(real, cfg)
        d = real[0] != 0
        m = slice(len(real) // 2, len(real))
        filas.append({"C": B.puntaje_t(real, sim, PL)[:, 1:].mean().item(),
                      "area_fin": (((sim[:, -1] >= PL) & d).flatten(1).sum(1).float() / d.sum()).mean().item(),
                      "borde": torch.nanmean(frontera(sim[:, m])).item()})
    return {"por_caja": filas, **{k: float(np.mean([f[k] for f in filas])) for k in filas[0]},
            "area_001": filas[0]["area_fin"], "crecer": crecer}


def main():
    dev = torch.device("cuda")
    res = {}
    for arg in sys.argv[2:]:
        nombre, _, crecer = arg.partition("@")
        r = probar(nombre, dev, float(crecer or 0))
        res[arg] = r
        print(f"{arg:36} C {r['C']:.3f}  área final {r['area_fin']:.3f}  área 001 {r['area_001']:.3f}  "
              f"borde {r['borde']:.3f}", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
