"""Opción 1 — ensamble: se promedian las probabilidades (softmax) de las 5 semillas antes de elegir el estado.

Redes: las mismas de la referencia (../0_base): ../../sin_p2 y ../../rotaciones, 9×9, paso 6, cruda, 2 capas de 32.
No se entrena nada. Decodificador argmax sobre el promedio. Una fila por variante y ronda.

Escribe resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/1_ensamble/ensamble.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun as cm                                                           # noqa: E402

VARIANTES = {"sin_p2": ("sin_p2", "log15"), "rotaciones": ("rotaciones", "log15_rot4")}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = cm.temperaturas()
    cm.log(CARPETA, "Inicio opción 1 (ensamble de 5 semillas)")
    for variante, (carpeta, prefijo) in VARIANTES.items():
        for entrena, prueba in cm.RONDAS:
            ronda = cm.ronda_nombre(entrena, prueba)
            clave = {"opcion": "1_ensamble", "variante": variante, "ronda": ronda, "semilla": "ensamble"}
            if cm.hecho(CARPETA, clave):
                continue
            redes = [cm.red_referencia(ronda, s, carpeta, prefijo) for s in cm.SEMILLAS]
            prob, nb = cm.predictor(redes, dev, cm.NT)
            fila = {**clave, "prueba": prueba, "paso_frames": cm.PASO, "decodificador": "argmax",
                    "red": f"{len(redes)} semillas de {carpeta}", **cm.evaluar(prob, nb, temps, prueba, dev)}
            cm.escribir(CARPETA, fila)
            cm.log(CARPETA, f"{variante} {ronda}: {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
