"""Video real | NCA de una red con ruido y edad (corrida del núcleo de la GUI, redes_dado/<nombre>/gui_<caja>.npy),
con la evolución real cada 30 min y 20 fps, como dado/video.py.

La red da un paso cada 6 h (12 imágenes reales). Entre dos pasos de la red, los cuadros intermedios son una
TRANSICIÓN, no pasos de la red: cada célula que cambia toma su nuevo estado en un momento del intervalo, las que se
vuelven plasmodio en orden de distancia al plasmodio del paso anterior (el frente avanza) y las demás al azar.
En los cuadros múltiplos de 12 la imagen de la NCA es exactamente la del paso de la red.

    python NCA/Physarum/edad/video_suave.py <nombre> <maestra> <caja>   → videos/<nombre>_<caja>_suave.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "dado"))
sys.path.insert(0, str(CARPETA.parent / "barrido"))
import comun as C                                                       # noqa: E402
import barrido as B                                                     # noqa: E402

RGB6 = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in C.PALETA6], np.uint8)
PL = 3
FPS = 20


def a_visible(X, cfg):
    n = B.info({**cfg, "edad": False})[0]
    return np.where(X >= n, X - (n - PL), X) if cfg.get("edad") else X


def tiempos(A, Bk, rng):
    """Momento (0–1) en que cada célula pasa de A a Bk; 2 si no cambia."""
    cambia = A != Bk
    t = np.full(A.shape, 2.0)
    crece = cambia & (A < PL) & (Bk >= PL)
    if crece.any():
        dist = ndimage.distance_transform_edt(A < PL)                     # distancia al plasmodio de A
        d = dist[crece]
        t[crece] = 0.85 * (d - d.min()) / max(d.max() - d.min(), 1) + 0.1 * rng.random(d.size)
    otros = cambia & ~crece
    t[otros] = 0.95 * rng.random(otros.sum())
    return t


def main():
    nombre, maestra, caja = sys.argv[1:4]
    cfg = {**B.BASE, **json.loads((B.CARPETA / "redes" / maestra / "config.json").read_text())}
    k = cfg["paso"]
    real = np.load(B.CA / cfg["ca"] / f"{caja}.npy")                      # cada 30 min, 6 estados visibles
    G = a_visible(np.load(CARPETA / "redes_dado" / nombre / f"gui_{caja}.npy") // C.DADOS, cfg)
    T = min(len(real), (len(G) - 1) * k + 1)
    rng = np.random.default_rng(0)
    tile, sep = 480, 8
    W, H = 2 * tile + sep, tile
    out = CARPETA / "videos" / f"{nombre}_{caja[9:18]}_suave.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libopenh264", "-b:v", "6M",
                           "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    tt = None
    for i in range(T):
        s, j = divmod(i, k)
        if j == 0:
            N = G[s]
            if s + 1 < len(G):
                tt = tiempos(G[s], G[s + 1], rng)
        else:
            N = np.where(tt < j / k, G[s + 1], G[s])
        img = Image.new("RGB", (W, H), (0, 0, 0))
        for c, E in enumerate((real[i], N)):
            img.paste(Image.fromarray(RGB6[E]).resize((tile, tile), Image.NEAREST), (c * (tile + sep), 0))
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out, T, "cuadros")


if __name__ == "__main__":
    main()
