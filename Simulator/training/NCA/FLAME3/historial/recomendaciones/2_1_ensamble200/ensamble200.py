"""2.1 — Ensamble a 200×200.

    ens5          promedio de las 5 redes de 200×200 con 9×9 de ../../opciones/7_resolucion200 (main.py, sin reentrenar)
    ens5_tta8     lo mismo promediando además las 8 orientaciones de la entrada (rotaciones y reflejos)
    d4            reentrenadas a 200×200 con 9×9 y una orientación al azar por muestra (8 posibles), entrenador en GPU,
                  5 semillas por ronda (comparar con ../0_control_gpu, variante gpu_200)
    d4_ens5       promedio de las 5 redes d4
    d4_ens5_tta8  promedio de las 5 redes d4 y de las 8 orientaciones

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/2_1_ensamble200/ensamble200.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 2.1 (ensamble a 200×200)")
    originales = lambda ronda: [cr.redes_200(ronda, s) for s in cr.SEMILLAS]           # noqa: E731
    d4 = lambda ronda: [CARPETA / "redes" / ronda / f"d4_s{s}.npz" for s in cr.SEMILLAS]  # noqa: E731
    cr.evaluar_ensamble(CARPETA, "2_1_ensamble200", "ens5", originales, dev, lado=200)
    cr.evaluar_ensamble(CARPETA, "2_1_ensamble200", "ens5_tta8", originales, dev, lado=200, orientaciones=8)
    cr.correr_entrenador(CARPETA, "2_1_ensamble200", "d4", dev, lado=200, aug="d4")
    cr.evaluar_ensamble(CARPETA, "2_1_ensamble200", "d4_ens5", d4, dev, lado=200)
    cr.evaluar_ensamble(CARPETA, "2_1_ensamble200", "d4_ens5_tta8", d4, dev, lado=200, orientaciones=8)
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
