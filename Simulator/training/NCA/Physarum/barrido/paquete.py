"""Paquete para la GUI de una red del barrido → gui/<nombre>/

    python paquete.py <nombre>

- <nombre>.npz: red_gui.npz (pesos con el sesgo calibrado en la última capa; se carga con Load NCA weights)
- physarum<estados>.csv: paleta (Load colors)
- iniciales/<caja>_inicio.npy (1, 160, 160): primera imagen de cada caja de prueba (Import)
- iniciales/<caja>_real.npy (T, 160, 160): la evolución real al paso de la red, para comparar
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import barrido as B
import video as V

CARPETA = Path(__file__).resolve().parent


def paquete(nombre):
    red = CARPETA / "redes" / nombre
    cfg = {**B.BASE, **json.loads((red / "config.json").read_text())}
    n, _ = B.info(cfg)
    out = CARPETA / "gui" / nombre
    (out / "iniciales").mkdir(parents=True, exist_ok=True)
    shutil.copy(red / "red_gui.npz", out / f"{nombre}.npz")
    (out / f"physarum{n}.csv").write_text("\n".join("#%02X%02X%02X" % tuple(c) for c in V.paleta(cfg)) + "\n")
    for caja in B.PRUEBA:
        S = B.cargar(cfg, caja).astype(np.int32)
        np.save(out / "iniciales" / f"{caja[9:18]}_inicio.npy", S[:1])
        np.save(out / "iniciales" / f"{caja[9:18]}_real.npy", S)
    print(out, sorted(p.name for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    paquete(sys.argv[1])
