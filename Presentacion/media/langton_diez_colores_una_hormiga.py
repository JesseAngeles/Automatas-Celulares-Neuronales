"""Video de la hormiga de Langton de 10 colores (RLRLRLRRRL) con una sola hormiga: original vs NCA.

Una hormiga en (49, 49) mirando a la izquierda, 100 × 100, frontera periódica, 13,000 pasos, red
NCA/LangtonAnt/colores_30x30_2hormigas/RLRLRLRRRL/4-64-relu_50-linear.npz (entrenada con dos hormigas).
La NCA difiere del original desde el paso 12,454 (contando desde 0; el reporte dice 12,455): la hormiga
se divide en dos. Cuadros cada 20 pasos hasta el 12,400 y cada 2 pasos después, para que se vea la división.
Usa exacta(), nca(), paleta() y video() de Entregables/Animation/LangtonAnt/generar.py.

Uso: Simulator/GUI/venv/bin/python Presentacion/media/langton_diez_colores_una_hormiga.py
Escribe media/videos/langton_diez_colores_una_hormiga.webm.
"""
import importlib.util
import subprocess
import tempfile
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
G = AQUI.parents[1] / "Entregables" / "Animation" / "LangtonAnt" / "generar.py"
spec = importlib.util.spec_from_file_location("generar", G)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

REGLA, LADO, PASOS = "RLRLRLRRRL", 100, 13000
RED = "NCA/LangtonAnt/colores_30x30_2hormigas/RLRLRLRRRL/4-64-relu_50-linear.npz"

orig, _ = g.exacta(REGLA, LADO, [(49, 49, 3)], PASOS)
red = g.nca(g.TRAIN / RED, orig[0], PASOS)
distintas = (orig != red).reshape(PASOS + 1, -1).sum(1)
print("primer paso con error:", int(np.nonzero(distintas)[0][0]))

idx = np.unique(np.concatenate([np.arange(0, 12400, 20), np.arange(12400, PASOS + 1, 2)]))
colores = g.paleta("langton_10colores_RLRLRLRRRL.csv")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    g.video(tmp / "o.mp4", orig[idx], idx, colores, f"Original · {REGLA}", 6)
    g.video(tmp / "n.mp4", red[idx], idx, colores, f"NCA · {REGLA}", 6)
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", tmp / "o.mp4", "-i", tmp / "n.mp4", "-filter_complex",
                    "[0:v]pad=iw+16:ih:0:0:white[a];[a][1:v]hstack=inputs=2,format=yuv420p",
                    "-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-row-mt", "1", "-deadline", "good",
                    "-cpu-used", "4", "-pix_fmt", "yuv420p", "-an",
                    AQUI / "videos" / "langton_diez_colores_una_hormiga.webm"], check=True)
print("listo")
