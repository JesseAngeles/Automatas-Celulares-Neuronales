"""2.3 — Destilar un ensamble en una sola red de 2 capas de 32 (la GUI corre una red).

La red aprende las probabilidades promedio del ensamble (entropía cruzada contra la distribución del ensamble) en
lugar del estado real. Entrenador en GPU, 5 semillas por ronda; comparar con ../0_control_gpu.
    dest_100   maestro: las 5 redes de sin_p2 (100×100)
    dest_200   maestro: las 5 redes de 200×200 con 9×9 de ../../opciones/7_resolucion200

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/2_3_destilar/destilar.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 2.3 (destilar el ensamble)")
    cr.correr_entrenador(CARPETA, "2_3_destilar", "dest_100", dev, lado=100,
                         maestros_fn=lambda ronda: [cr.red_referencia(ronda, s) for s in cr.SEMILLAS])
    cr.correr_entrenador(CARPETA, "2_3_destilar", "dest_200", dev, lado=200,
                         maestros_fn=lambda ronda: [cr.redes_200(ronda, s) for s in cr.SEMILLAS])
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
