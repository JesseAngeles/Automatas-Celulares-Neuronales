"""Corre una alumna en el núcleo NCA real de la GUI (CA.NCA, core/nca.cu) desde iniciales/<caja>_inicio.npy del
paquete y la compara con PyTorch. Como la dinámica es caótica (dado), una diferencia de redondeo en un empate se
propaga: se reporta el primer paso con diferencias y las métricas de la corrida de la GUI.

    python gui.py <nombre>
"""
import sys

import numpy as np
import torch

import comun as C

sys.path.insert(0, str(C.TRAINING.parent / "GUI"))
from CA import NCA                                                         # noqa: E402


def correr(nombre):
    dev = torch.device("cuda")
    paq = C.CARPETA / "gui" / nombre
    w = np.load(paq / f"{nombre}.npz")
    red = C.Red(paq / f"{nombre}.npz", dev)
    n = len(w["layer_in"])
    nb = w["neighbourhood"].astype(np.int32)
    out = {}
    for caja in C.PRUEBA:
        corto = caja[9:18]
        E0 = np.load(paq / "iniciales" / f"{corto}_inicio.npy")[0]
        S = np.load(paq / "iniciales" / f"{corto}_real.npy")
        H, W = E0.shape
        nca = NCA(W, H, E0.astype(np.float32).ravel(), nb, len(nb))
        nca.set_boundary(0)
        nca.set_one_hot(int(w["one_hot_states"]))
        nca.set_weights([w[f"weights_{i}"] for i in range(n)], [w[f"biases_{i}"] for i in range(n)],
                        w["layer_in"].tolist(), w["layer_out"].tolist(), w["activations"].tolist())
        G = [E0.astype(np.int8)]
        P = torch.from_numpy(E0.astype(np.int8)).to(dev).unsqueeze(0)
        primero, dif = None, []
        for t in range(1, len(S)):
            nca.step()
            G.append(np.asarray(nca.get_space(), np.float32).reshape(H, W).round().astype(np.int8))
            P = red.logits(torch.from_numpy(G[-2]).to(dev).unsqueeze(0)).argmax(-1).to(torch.int8)   # mismo estado previo
            d = float((P[0].cpu().numpy() != G[-1]).mean())
            dif.append(d)
            if d > 0 and primero is None:
                primero = t
        G = np.stack(G)
        np.save(C.CARPETA / "redes" / nombre / f"gui_{caja}.npy", G)
        Gt = torch.from_numpy(G).to(dev)
        m = C.resumen(torch.from_numpy(S.astype(np.int8)).to(dev), (Gt // C.DADOS).to(torch.int8).unsqueeze(0))
        out[caja] = {"dif_un_paso_media": float(np.mean(dif)), "dif_un_paso_max": float(np.max(dif)),
                     **{k: float(np.mean(v)) for k, v in m.items()}}
        print(caja, {k: round(v, 5) for k, v in out[caja].items()}, flush=True)
    C.guardar_json(C.CARPETA / "redes" / nombre / "gui.json", out)


if __name__ == "__main__":
    correr(sys.argv[1])
