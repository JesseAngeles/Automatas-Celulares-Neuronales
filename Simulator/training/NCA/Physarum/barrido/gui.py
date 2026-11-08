"""Corre red_gui.npz en el núcleo NCA real de la GUI (CA.NCA, core/nca.cu) y lo compara con la simulación de
barrido.py (PyTorch) paso a paso.

    python gui.py <nombre> [caja]

Carga los pesos igual que GUI/controllers/SimulationController.on_load_weights: frontera abierta (0), one-hot del
.npz, argmax de la salida.
"""
import json
import sys
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parents[3] / "GUI"))                    # CA.cpython-*.so
from CA import NCA                                                     # noqa: E402

import barrido as B                                                    # noqa: E402


def comparar(nombre, caja=B.PRUEBA[2]):
    red = CARPETA / "redes" / nombre
    cfg = json.loads((red / "config.json").read_text())
    S = B.cargar(cfg, caja)
    libre = np.load(red / f"libre_{caja}.npy")
    w = np.load(red / "red_gui.npz")
    n_capas = len(w["layer_in"])
    nb = w["neighbourhood"].astype(np.int32)
    H, W = S.shape[1:]
    nca = NCA(W, H, S[0].astype(np.float32).ravel(), nb, len(nb))
    nca.set_boundary(0)                                                # abierta: fuera = 0
    nca.set_one_hot(int(w["one_hot_states"]))
    nca.set_weights([w[f"weights_{i}"] for i in range(n_capas)], [w[f"biases_{i}"] for i in range(n_capas)],
                    w["layer_in"].tolist(), w["layer_out"].tolist(), w["activations"].tolist())
    dif = []
    gui = [S[0]]
    for t in range(1, len(S)):
        nca.step()
        E = np.asarray(nca.get_space(), np.float32).reshape(H, W).round().astype(np.int8)
        gui.append(E)
        dif.append(float((E != libre[t]).mean()))
    np.save(red / f"gui_{caja}.npy", np.stack(gui))
    print(f"{nombre} {caja}: {len(S) - 1} pasos en el núcleo de la GUI; células distintas a PyTorch: "
          f"media {np.mean(dif):.5f}, máx {np.max(dif):.5f}, último paso {dif[-1]:.5f}")
    return dif


if __name__ == "__main__":
    comparar(*sys.argv[1:])
