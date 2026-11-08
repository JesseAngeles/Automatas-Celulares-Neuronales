"""Opción 2 — decodificación estocástica: en vez de tomar el estado más probable (argmax) se usa
    muestreo   se sortea el siguiente estado con las probabilidades de la red (20 repeticiones; las métricas de un
               paso y de la simulación libre son el promedio; rollout_mae_c_campo_medio compara el promedio de las
               20 simulaciones contra el real)
    esperado   temperatura esperada Σ p · representante, discretizada de nuevo con los mismos cortes

Redes: las de la referencia (../0_base, ../../sin_p2: 9×9, paso 6, cruda, 2 capas de 32), 5 semillas. No se entrena.
Escribe resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/2_estocastico/estocastico.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun as cm                                                           # noqa: E402

DECODIFICADORES = {"muestreo": 20, "esperado": 1}     # decodificador → repeticiones


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = cm.temperaturas()
    cm.log(CARPETA, "Inicio opción 2 (decodificación estocástica)")
    for decodificador, rep in DECODIFICADORES.items():
        for entrena, prueba in cm.RONDAS:
            ronda = cm.ronda_nombre(entrena, prueba)
            for s in cm.SEMILLAS:
                clave = {"opcion": "2_estocastico", "variante": decodificador, "ronda": ronda, "semilla": s}
                if cm.hecho(CARPETA, clave):
                    continue
                red = cm.red_referencia(ronda, s)
                prob, nb = cm.predictor([red], dev, cm.NT)
                fila = {**clave, "prueba": prueba, "paso_frames": cm.PASO, "decodificador": decodificador,
                        "repeticiones": rep, "red": str(red.relative_to(cm.TRAINING)),
                        **cm.evaluar(prob, nb, temps, prueba, dev, decodificador=decodificador, repeticiones=rep,
                                     semilla=s)}
                cm.escribir(CARPETA, fila)
                cm.log(CARPETA, f"{decodificador} {ronda} s{s}: {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
