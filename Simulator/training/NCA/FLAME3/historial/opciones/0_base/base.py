"""Opción 0 — referencia: las redes de ../../sin_p2 y ../../rotaciones (5 semillas, 9×9, paso 6, cruda, 2 capas
de 32) evaluadas con la evaluación común de ../comun.py, para comparar todas las opciones con las mismas métricas.

Escribe resultados.csv (una fila por variante, ronda y semilla) y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/0_base/base.py
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
    cm.log(CARPETA, "Inicio opción 0 (referencia)")
    for variante, (carpeta, prefijo) in VARIANTES.items():
        for entrena, prueba in cm.RONDAS:
            ronda = cm.ronda_nombre(entrena, prueba)
            for s in cm.SEMILLAS:
                clave = {"opcion": "0_base", "variante": variante, "ronda": ronda, "semilla": s}
                if cm.hecho(CARPETA, clave):
                    continue
                red = cm.red_referencia(ronda, s, carpeta, prefijo)
                prob, nb = cm.predictor([red], dev, cm.NT)
                fila = {**clave, "prueba": prueba, "paso_frames": cm.PASO, "decodificador": "argmax",
                        "red": str(red.relative_to(cm.TRAINING)), **cm.evaluar(prob, nb, temps, prueba, dev)}
                cm.escribir(CARPETA, fila)
                cm.log(CARPETA, f"{variante} {ronda} s{s}: {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
