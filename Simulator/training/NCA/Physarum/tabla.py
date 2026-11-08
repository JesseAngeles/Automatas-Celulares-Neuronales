"""Comparación real / NCA con sesgo / NCA con dado en las 4 cajas de prueba, todo muestreado cada 6 h para que sea
comparable (la de sesgo da un paso cada 6 h). Usa las corridas del núcleo de la GUI.

- coincidencia: puntaje (IoU × área menor / área mayor) por paso, promedio desde el primer paso
- final: fracción de la caja con plasmodio en la última imagen (la de sesgo, en su último paso)
- borde: fracción del plasmodio que toca algo que no es plasmodio, promedio de la segunda mitad
- brillo_6h: fracción del plasmodio (en t y t + 6 h) que cambia de nivel, segunda mitad
- brillo_30min: lo mismo cada 30 min (solo real y dado)

    python NCA/Physarum/tabla.py [nombre=b16x128]   → tabla.json junto a este script
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA / "dado"))
import comun as C                                                  # noqa: E402


def medir(real, sim):
    """real (T, H, W) y sim (T', H, W) visibles en la GPU, mismo muestreo."""
    T = min(len(real), len(sim))
    real, sim = real[:T], sim[:T]
    d = real[0] != 0
    m = slice(T // 2, T)
    return {"coincidencia": C.puntaje_t(real, sim.unsqueeze(0))[0, 1:].mean().item(),
            "final": ((sim[-1] >= C.PL) & d).sum().item() / d.sum().item(),
            "borde": torch.nanmean(C.frontera(sim[m])).item(),
            "brillo_6h": torch.nanmean(C.interno(sim[m])).item()}


def main(nombre):
    dev = torch.device("cuda")
    filas = {"real": [], "sesgo": [], "dado": []}
    for caja in C.PRUEBA:
        real = torch.from_numpy(C.cargar(caja)).to(dev)
        sesgo = torch.from_numpy(np.load(CARPETA / "barrido" / "redes" / "e6_6h_circ6_da" / f"gui_{caja}.npy")).to(dev)
        dado = torch.from_numpy(np.load(CARPETA / "dado" / "redes" / nombre / f"gui_{caja}.npy") // C.DADOS).to(dev)
        r6 = real[::12]
        filas["real"].append({**medir(r6, r6), "final": ((real[-1] >= C.PL) & (real[0] != 0)).sum().item() / (real[0] != 0).sum().item(),
                              "brillo_30min": torch.nanmean(C.interno(real[len(real) // 2:])).item()})
        filas["sesgo"].append(medir(r6, sesgo))
        filas["dado"].append({**medir(r6, dado[::12]),
                              "final": ((dado[-1] >= C.PL) & (real[0] != 0)).sum().item() / (real[0] != 0).sum().item(),
                              "brillo_30min": torch.nanmean(C.interno(dado[len(dado) // 2:])).item()})
    res = {k: {m: float(np.mean([f[m] for f in v])) for m in v[0]} for k, v in filas.items()}
    res["por_caja"] = filas
    (CARPETA / "tabla.json").write_text(json.dumps(res, indent=1))
    for k in ("real", "sesgo", "dado"):
        print(k, {m: round(x, 4) for m, x in res[k].items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "b16x128")
