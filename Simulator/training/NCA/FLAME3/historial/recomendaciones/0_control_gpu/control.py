"""Control del entrenador en GPU de ../comun_r.py: la referencia (9×9, paso 6, cruda, 2 capas de 32, 200 épocas,
exp 1e-3 → 1e-5) entrenada con entrenar_gpu sin ninguna opción, a 100×100 y a 200×200, 5 semillas por ronda.
Las variantes de 2.3 y del grupo 3 se comparan contra este control: así la diferencia no viene del entrenador
(entrenar_gpu baraja las muestras de otra forma que scripts/main.py).

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/0_control_gpu/control.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio control del entrenador en GPU")
    cr.correr_entrenador(CARPETA, "0_control_gpu", "gpu_100", dev, lado=100)
    cr.correr_entrenador(CARPETA, "0_control_gpu", "gpu_200", dev, lado=200)
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
