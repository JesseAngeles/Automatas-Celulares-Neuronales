"""Cierre del plan para una red con ruido de dado_largo.py: paquete para la GUI, verificación en el núcleo real
(core/nca.cu), video y figura.

    python NCA/Physarum/edad/cierre.py <nombre> <maestra>

→ gui/<nombre>/: <nombre>.npz (pesos de main.py), physarum<NE>.csv (paleta), iniciales/<caja>_inicio.npy (primera
  imagen con dados al azar) y <caja>_real.npy (evolución real visible, un paso = paso de la maestra)
→ redes_dado/<nombre>/gui_<caja>.npy y gui.json (diferencias contra PyTorch y medidas de la corrida de la GUI)
→ videos/<nombre>_<caja>.mp4 (real | NCA, 6 colores visibles) y <nombre>_finales.png
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "dado"))
sys.path.insert(0, str(CARPETA.parent / "barrido"))
import comun as C                                                       # noqa: E402
import barrido as B                                                     # noqa: E402
from medir import frontera, interno, visible, PL                        # noqa: E402

sys.path.insert(0, str(C.TRAINING.parent / "GUI"))
from CA import NCA                                                      # noqa: E402

RGB6 = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in C.PALETA6], np.uint8)


def paleta(cfg):
    """Un color por estado combinado (visible × dado), tomado de PALETA24. Con 4 estados el plasmodio usa los
    colores del medio; los estados nuevos (edad) usan los de su nivel corridos al naranja."""
    n_base = B.info({**cfg, "edad": False})[0]
    niveles = list(range(PL, n_base)) if n_base - PL == 3 else [PL + 1]
    grupo = lambda v: C.PALETA24[v * C.DADOS:(v + 1) * C.DADOS]
    out = [h for v in list(range(PL)) + niveles for h in grupo(v)]
    if cfg.get("edad"):
        for v in niveles:
            for h in grupo(v):
                r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
                out.append(f"#{min(255, r + 40):02X}{int(g * 0.55):02X}{int(b * 0.3):02X}")
    return out


def cfg_de(maestra):
    return {**B.BASE, **json.loads((B.CARPETA / "redes" / maestra / "config.json").read_text())}


def paquete(nombre, cfg):
    out = CARPETA / "gui" / nombre
    (out / "iniciales").mkdir(parents=True, exist_ok=True)
    shutil.copy(CARPETA / "redes_dado" / nombre / "red.npz", out / f"{nombre}.npz")
    pal = paleta(cfg)
    assert len(pal) == B.info(cfg)[0] * C.DADOS
    (out / f"physarum{len(pal)}.csv").write_text("".join(c + "\n" for c in pal))
    rng = np.random.default_rng(0)
    for caja in B.PRUEBA:
        S = B.cargar(cfg, caja).astype(np.int32)
        E0 = S[:1] * C.DADOS + rng.integers(0, C.DADOS, S[:1].shape)
        np.save(out / "iniciales" / f"{caja[9:18]}_inicio.npy", E0.astype(np.int32))
        np.save(out / "iniciales" / f"{caja[9:18]}_real.npy", S)
    return out


def gui(nombre, cfg, paq):
    dev = torch.device("cuda")
    w = np.load(paq / f"{nombre}.npz")
    red = C.Red(paq / f"{nombre}.npz", dev)
    n = len(w["layer_in"])
    nb = w["neighbourhood"].astype(np.int32)
    res = {}
    for caja in B.PRUEBA:
        corto = caja[9:18]
        E0 = np.load(paq / "iniciales" / f"{corto}_inicio.npy")[0]
        S = np.load(paq / "iniciales" / f"{corto}_real.npy")
        H, W = E0.shape
        nca = NCA(W, H, E0.astype(np.float32).ravel(), nb, len(nb))
        nca.set_boundary(0)
        nca.set_one_hot(int(w["one_hot_states"]))
        nca.set_weights([w[f"weights_{i}"] for i in range(n)], [w[f"biases_{i}"] for i in range(n)],
                        w["layer_in"].tolist(), w["layer_out"].tolist(), w["activations"].tolist())
        G, dif = [E0.astype(np.int8)], []
        for _ in range(1, len(S)):
            nca.step()
            G.append(np.asarray(nca.get_space(), np.float32).reshape(H, W).round().astype(np.int8))
            P = red.logits(torch.from_numpy(G[-2]).to(dev).unsqueeze(0)).argmax(-1).to(torch.int8)
            dif.append(float((P[0].cpu().numpy() != G[-1]).mean()))
        G = np.stack(G)
        np.save(CARPETA / "redes_dado" / nombre / f"gui_{caja}.npy", G)
        real = visible(torch.from_numpy(S.astype(np.int8)).to(dev), cfg)
        sim = visible(torch.from_numpy(G).to(dev) // C.DADOS, cfg)
        d = real[0] != 0
        m = slice(len(real) // 2, len(real))
        res[caja] = {"dif_un_paso_max": max(dif), "C": B.puntaje_t(real, sim.unsqueeze(0), PL)[0, 1:].mean().item(),
                     "area_fin": ((sim[-1] >= PL) & d).sum().item() / d.sum().item(),
                     "borde": torch.nanmean(frontera(sim[m])).item()}
        if B.info({**cfg, "edad": False})[0] - PL == 3:
            res[caja]["brillo"] = torch.nanmean(interno(sim[m])).item()
        print(caja, {k: round(v, 4) for k, v in res[caja].items()}, flush=True)
    res["promedio"] = {k: float(np.mean([r[k] for r in res.values()])) for k in next(iter(res.values()))}
    (CARPETA / "redes_dado" / nombre / "gui.json").write_text(json.dumps(res, indent=1))
    return res


def colores(X, cfg):
    """Estados visibles (con edad o no) → RGB con los 6 colores del reporte (4 estados: plasmodio = medio)."""
    n_vis = B.info({**cfg, "edad": False})[0]
    V = X.copy()
    if cfg.get("edad"):
        V = np.where(V >= n_vis, V - (n_vis - PL), V)
    if n_vis == 4:
        V = np.where(V == 3, 4, V)
    return RGB6[V]


def video(nombre, cfg, caja, fps=2, tile=480, sep=8):
    S = colores(B.cargar(cfg, caja), cfg)
    L = colores(np.load(CARPETA / "redes_dado" / nombre / f"gui_{caja}.npy") // C.DADOS, cfg)
    W, H = 2 * tile + sep, tile
    (CARPETA / "videos").mkdir(exist_ok=True)
    out = CARPETA / "videos" / f"{nombre}_{caja[9:18]}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libopenh264", "-b:v", "6M",
                           "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for t in range(len(S)):
        img = Image.new("RGB", (W, H), (0, 0, 0))
        for j, E in enumerate((S, L)):
            img.paste(Image.fromarray(E[t]).resize((tile, tile), Image.NEAREST), (j * (tile + sep), 0))
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    return out


def main():
    nombre, maestra = sys.argv[1:3]
    cfg = cfg_de(maestra)
    C.NV = B.info(cfg)[0]
    C.NE = C.NV * C.DADOS
    paq = paquete(nombre, cfg)
    r = gui(nombre, cfg, paq)
    print("promedio GUI", {k: round(v, 4) for k, v in r["promedio"].items()})
    for caja in B.PRUEBA:
        print(video(nombre, cfg, caja))


if __name__ == "__main__":
    main()
