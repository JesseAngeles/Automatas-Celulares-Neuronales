"""Videos de una alumna con la corrida del núcleo de la GUI (redes/<nombre>/gui_<caja>.npy, de gui.py), 20 fps
(paso de 30 min).

    python video.py <nombre>
→ videos/<nombre>_<caja>.mp4      real | NCA, sin texto, la NCA con los 24 colores de la GUI
  videos/<nombre>_<caja>_6.mp4    igual, la NCA con los 6 colores del estado visible
  videos/<nombre>_prueba.mp4      las 4 cajas de prueba, real arriba y NCA (24 colores) abajo
"""
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import comun as C

RGB6 = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in C.PALETA6], np.uint8)
RGB24 = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in C.PALETA24], np.uint8)
VIDEOS = C.CARPETA / "videos"
FPS = 20


def ffmpeg(out, W, H):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                             f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libopenh264", "-b:v", "6M",
                             "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)


def series(nombre, caja, colores24=True):
    """(real RGB (T, H, W, 3), NCA RGB (T, H, W, 3))."""
    G = np.load(C.CARPETA / "redes" / nombre / f"gui_{caja}.npy")
    return RGB6[C.cargar(caja)], (RGB24[G] if colores24 else RGB6[G // C.DADOS])


def uno(nombre, caja=C.PRUEBA[2], colores24=True, tile=480, sep=8):
    S, L = series(nombre, caja, colores24)
    W, H = 2 * tile + sep, tile
    out = VIDEOS / f"{nombre}_{caja[9:18]}{'' if colores24 else '_6'}.mp4"
    ff = ffmpeg(out, W, H)
    for t in range(len(S)):
        img = Image.new("RGB", (W, H), (0, 0, 0))
        for j, E in enumerate((S, L)):
            img.paste(Image.fromarray(E[t]).resize((tile, tile), Image.NEAREST), (j * (tile + sep), 0))
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


def cuatro(nombre, tile=300):
    f1 = ImageFont.truetype("/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf", 20)
    f2 = ImageFont.truetype("/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf", 15)
    ss = [series(nombre, c) for c in C.PRUEBA]
    T = max(len(s) for s, _ in ss)
    sep, LAB, HEAD = 8, 22, 40
    W = 4 * (tile + sep) + sep
    H = HEAD + 2 * (tile + LAB)
    W, H = W + W % 2, H + H % 2
    out = VIDEOS / f"{nombre}_prueba.mp4"
    ff = ffmpeg(out, W, H)
    for t in range(T):
        img = Image.new("RGB", (W, H), (20, 20, 20))
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), f"{nombre}: real (arriba) vs NCA con dado (abajo)   {t * 0.5:5.1f} h", fill=(235,) * 3, font=f1)
        for j, (caja, (S, L)) in enumerate(zip(C.PRUEBA, ss)):
            k = min(t, len(S) - 1)
            for i, E in enumerate((S, L)):
                c = E[k] if t < len(S) else (E[k] * 0.45).astype(np.uint8)
                x, y = sep + j * (tile + sep), HEAD + i * (tile + LAB)
                img.paste(Image.fromarray(c).resize((tile, tile), Image.NEAREST), (x, y + LAB))
                dr.text((x, y + 2), f"{caja[9:18]} {'real' if i == 0 else 'NCA'}", fill=(200,) * 3, font=f2)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    VIDEOS.mkdir(exist_ok=True)
    uno(sys.argv[1])
    uno(sys.argv[1], colores24=False)
    cuatro(sys.argv[1])
