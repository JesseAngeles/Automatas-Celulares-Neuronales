"""Imágenes para la figura de los criterios de la selección visual: la configuración final (paso de 30 s, 9x9, 2:32,
semilla 42) en el simulador, con frontera periódica y un foco de fuego en el centro de un espacio frío sin quemar.

Usa el mismo núcleo CUDA que la GUI (Simulator/GUI/CA*.so), con los pesos que se cargan en "Load NCA weights":
    --entrenada flame_plot_2   redes/barrido/p3_a_p2/k6_9x9_2x32_onehot_s42.npz (la de la figura del reporte)
    --entrenada flame_plot_1   redes/barrido/p2_a_p3/k6_9x9_2x32_onehot_s42.npz

Espacio de --lado x --lado células en el estado 0 (25.3–59.0 °C) con un cuadrado de --foco x --foco células en el estado
--estado-foco en el centro. Guarda cada paso de --pasos en
Documentation/img/experimentacion/naturales/Flame3/criterios/entrenada_<conjunto>/paso_<n>.png (4 px por célula) y,
en la consola, el porcentaje de células con fuego (estados 5 a 11, 137.6 °C o más) y quemadas (estados 12 a 17).

Resultado (2026-10-18): la entrenada con flame_plot_2 quema todo el espacio hacia el paso 23 y sigue 100 % quemada
hasta el paso 60, sin volver a encender (igual con focos de 2 a 10 células y estados 7 a 11). La entrenada con
flame_plot_1 propaga más lento y, después de quemar el 95 %, un frente sigue avanzando sobre lo quemado.

Uso (desde Simulator/training):
    ../GUI/venv/bin/python NCA/FLAME3/burned_states/figura_criterios.py --entrenada flame_plot_2
"""
import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[2]
GUI = TRAINING.parent / "GUI"
IMG = TRAINING.parents[1] / "Documentation" / "img" / "experimentacion" / "naturales" / "Flame3" / "criterios"

try:                                                               # libcudart del paquete pip, como en GUI/main.py
    import nvidia.cu13
    for _dir in nvidia.cu13.__path__:
        _cudart = os.path.join(_dir, "lib", "libcudart.so.13")
        if os.path.exists(_cudart):
            ctypes.CDLL(_cudart, mode=ctypes.RTLD_GLOBAL)
            break
except (ImportError, OSError):
    pass
sys.path.insert(0, str(GUI))
from CA import NCA                                                 # noqa: E402

PALETA = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]
                   for h in (GUI / "colors" / "flame_log12_quemado6.csv").read_text().split()], dtype=np.uint8)
PERIODICA, ZOOM = 1, 4
RONDA = {"flame_plot_2": "p3_a_p2", "flame_plot_1": "p2_a_p3"}       # p2 = flame_plot_1, p3 = flame_plot_2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrenada", choices=RONDA, default="flame_plot_2")
    ap.add_argument("--lado", type=int, default=100)
    ap.add_argument("--foco", type=int, default=6)
    ap.add_argument("--estado-foco", type=int, default=10)
    ap.add_argument("--pasos", type=int, nargs="+", default=[0, 5, 10, 14, 18, 40])
    ap.add_argument("--destino", type=Path, default=None)
    args = ap.parse_args()

    datos = np.load(CARPETA / "redes" / "barrido" / RONDA[args.entrenada] / "k6_9x9_2x32_onehot_s42.npz")
    espacio = np.zeros((args.lado, args.lado), dtype=np.float32)
    a = (args.lado - args.foco) // 2
    espacio[a:a + args.foco, a:a + args.foco] = args.estado_foco

    nca = NCA(args.lado, args.lado, espacio.flatten(), datos["neighbourhood"].astype(np.int32),
              len(datos["neighbourhood"]))
    nca.set_boundary(PERIODICA)
    nca.set_one_hot(int(datos["one_hot_states"]))
    n = int(datos["layer_in"].shape[0])
    nca.set_weights([datos[f"weights_{i}"] for i in range(n)], [datos[f"biases_{i}"] for i in range(n)],
                    datos["layer_in"].tolist(), datos["layer_out"].tolist(), datos["activations"].tolist())

    destino = args.destino or IMG / f"entrenada_{args.entrenada}"
    destino.mkdir(parents=True, exist_ok=True)
    estado = espacio.astype(np.int32)
    for paso in range(max(args.pasos) + 1):
        if paso > 0:
            nca.step()
            crudo = nca.get_space()                                # argmax ya en el kernel, como en la GUI
            nca.set_space(crudo)
            estado = crudo.astype(np.int32)
        fuego = ((estado >= 5) & (estado < 12)).mean() * 100
        quemado = (estado >= 12).mean() * 100
        print(f"paso {paso:3d} ({paso * 30:4d} s): fuego {fuego:5.1f} %, quemado {quemado:5.1f} %")
        if paso in args.pasos:
            Image.fromarray(PALETA[estado].repeat(ZOOM, 0).repeat(ZOOM, 1)).save(destino / f"paso_{paso}.png")


if __name__ == "__main__":
    main()
