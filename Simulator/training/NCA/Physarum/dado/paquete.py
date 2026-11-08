"""Paquete para la GUI de una alumna → gui/<nombre>/

    python paquete.py <nombre>

- <nombre>.npz: pesos tal cual de main.py (Load NCA weights); sin sesgo ni otros ajustes.
- physarum24.csv: paleta (Load colors), un color por estado, parecido entre los 4 dados de cada visible.
- iniciales/<caja>_inicio.npy (1, 160, 160): primera imagen de cada caja de prueba con dados al azar (Import).
  Sin dados al azar (todo dado 0) la regla del dado se queda en 0 y la red se congela.
- iniciales/<caja>_real.npy (T, 160, 160): la evolución real en estados visibles, para comparar.
"""
import shutil
import sys

import numpy as np

import comun as C


def paquete(nombre):
    out = C.CARPETA / "gui" / nombre
    (out / "iniciales").mkdir(parents=True, exist_ok=True)
    shutil.copy(C.CARPETA / "redes" / nombre / "red.npz", out / f"{nombre}.npz")
    (out / f"physarum{C.NE}.csv").write_text("".join(c + "\n" for c in C.PALETA24))
    rng = np.random.default_rng(0)
    for caja in C.PRUEBA:
        S = C.cargar(caja).astype(np.int32)
        E0 = S[:1] * C.DADOS + rng.integers(0, C.DADOS, S[:1].shape)
        np.save(out / "iniciales" / f"{caja[9:18]}_inicio.npy", E0.astype(np.int32))
        np.save(out / "iniciales" / f"{caja[9:18]}_real.npy", S)
    print(out, sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    paquete(sys.argv[1])
