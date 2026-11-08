"""Videos de una red del barrido (simulación libre con argmax y el sesgo de red_gui.npz).

    python video.py uno <nombre> [caja]      → videos/<nombre>_<caja>.mp4: real (izq.) vs NCA (der.), sin texto
    python video.py cuatro <nombre>          → videos/<nombre>_prueba.mp4: las 4 cajas de prueba, real arriba
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import barrido as B

CARPETA = Path(__file__).resolve().parent
VIDEOS = CARPETA / "videos"
PALETA6 = ["#000000", "#2A2A30", "#E6CFA8", "#6E6A12", "#D2AE00", "#FFF46A"]


def paleta(cfg):
    """Colores por estado: borde, fondo, avena y niveles de plasmodio de oliva a amarillo claro."""
    n, pl = B.info(cfg)
    if (n, pl) == (6, 3):
        return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PALETA6], np.uint8)
    niveles = n - pl
    a, b = np.array([0x6E, 0x6A, 0x12]), np.array([0xFF, 0xF4, 0x6A])
    pl_col = [np.array([0xD2, 0xAE, 0x00])] if niveles == 1 else \
        [np.round(a + (b - a) * k / (niveles - 1)) for k in range(niveles)]
    base = [PALETA6[0], PALETA6[1]] + ([PALETA6[2]] if pl == 3 else [])
    rgb = [[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in base] + [list(map(int, c)) for c in pl_col]
    return np.array(rgb, np.uint8)


def ffmpeg(out, W, H, fps):
    VIDEOS.mkdir(exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libopenh264", "-b:v", "6M",
                             "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)


def fps_de(cfg, fps):
    """Por default 20 cuadros por segundo entre el paso (30 min → 20, 5 h → 2), mínimo 2."""
    return fps or max(2, round(20 / cfg["paso"]))


def uno(nombre, caja=B.PRUEBA[2], tile=480, sep=8, fps=None):
    cfg = {**B.BASE, **json.loads((CARPETA / "redes" / nombre / "config.json").read_text())}
    fps = fps_de(cfg, fps)
    rgb = paleta(cfg)
    S, L = B.cargar(cfg, caja), np.load(CARPETA / "redes" / nombre / f"libre_{caja}.npy")
    out = VIDEOS / f"{nombre}_{caja[9:18]}.mp4"
    W, H = 2 * tile + sep, tile
    ff = ffmpeg(out, W, H, fps)
    for t in range(len(S)):
        img = Image.new("RGB", (W, H), (0, 0, 0))
        for j, E in enumerate((S, L)):
            img.paste(Image.fromarray(rgb[E[t]]).resize((tile, tile), Image.NEAREST), (j * (tile + sep), 0))
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


def cuatro(nombre, tile=300, fps=None):
    cfg = {**B.BASE, **json.loads((CARPETA / "redes" / nombre / "config.json").read_text())}
    fps = fps_de(cfg, fps)
    rgb = paleta(cfg)
    f1 = ImageFont.truetype("/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf", 20)
    f2 = ImageFont.truetype("/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf", 15)
    series = [(B.cargar(cfg, c), np.load(CARPETA / "redes" / nombre / f"libre_{c}.npy")) for c in B.PRUEBA]
    T = max(len(s) for s, _ in series)
    sep, LAB, HEAD = 8, 22, 40
    W = 4 * (tile + sep) + sep
    H = HEAD + 2 * (tile + LAB)
    W, H = W + W % 2, H + H % 2
    out = VIDEOS / f"{nombre}_prueba.mp4"
    ff = ffmpeg(out, W, H, fps)
    for t in range(T):
        img = Image.new("RGB", (W, H), (20, 20, 20))
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), f"{nombre}: real (arriba) vs NCA (abajo)   {t * 0.5 * cfg['paso']:5.1f} h",
                fill=(235,) * 3, font=f1)
        for j, (caja, (S, L)) in enumerate(zip(B.PRUEBA, series)):
            k = min(t, len(S) - 1)
            for i, E in enumerate((S, L)):
                c = rgb[E[k]] if t < len(S) else (rgb[E[k]] * 0.45).astype(np.uint8)
                x, y = sep + j * (tile + sep), HEAD + i * (tile + LAB)
                img.paste(Image.fromarray(c).resize((tile, tile), Image.NEAREST), (x, y))
                dr.text((x + 4, y + tile + 2), f"{caja[9:18]} {'real' if i == 0 else 'NCA'}", fill=(210,) * 3,
                        font=f2)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    {"uno": uno, "cuatro": cuatro}[sys.argv[1]](*sys.argv[2:])
