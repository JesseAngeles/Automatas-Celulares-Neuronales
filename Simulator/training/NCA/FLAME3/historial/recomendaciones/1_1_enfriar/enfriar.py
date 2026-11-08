"""1.1 — Línea base "solo enfriar": NCA sin vecinos.

    tabla      P(estado siguiente | estado actual) contada en el escenario de entrenamiento (todas las secuencias con
               desfase); se predice el estado más probable. Es el óptimo de una NCA que solo ve su propia celda.
    red1x1     red de 2 capas de 32 con vecindad 1×1, entrenada con el entrenador en GPU (5 semillas); debería dar
               lo mismo que la tabla.
    tabla_200  la tabla a 200×200 (para comparar con ../../opciones/7_resolucion200).
Paso 6, p1 → p3 y p3 → p1. Escribe resultados.csv, tablas.csv (la tabla de cada ronda) y registro.log.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/1_1_enfriar/enfriar.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 1.1 (solo enfriar)")
    for lado in (100, 200):
        temps = cr.temperaturas(lado)
        escala = lado // 100
        for entrena, prueba in cr.RONDAS:
            ronda = cr.ronda_nombre(entrena, prueba)
            secs = [s for k in entrena for s in cr.secs_offset(cr.estados(temps[k]))]
            variante = "tabla" if lado == 100 else "tabla_200"
            clave = {"experimento": "1_1_enfriar", "variante": variante, "ronda": ronda, "semilla": "-"}
            if not cr.hecho(CARPETA, clave):
                prob, nb, P = cr.tabla_transicion(secs, dev)
                with open(CARPETA / "tablas.csv", "a") as f:
                    for s in range(cr.NT):
                        f.write(f"{variante},{ronda},{s},{int(P[s].argmax())}," + ",".join(f"{x:.4f}" for x in P[s]) + "\n")
                fila = {**clave, "prueba": prueba, "lado": lado,
                        **cr.evaluar(prob, nb, temps, prueba, dev, escala=escala)}
                cr.escribir(CARPETA, fila)
                cr.log(CARPETA, f"{variante} {ronda}: {cr.resumen(fila)}")
            if lado == 200:
                continue
            for s in cr.SEMILLAS:
                clave = {"experimento": "1_1_enfriar", "variante": "red1x1", "ronda": ronda, "semilla": s}
                if cr.hecho(CARPETA, clave):
                    continue
                (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                base = CARPETA / "redes" / ronda / f"red1x1_k6_cruda_2x32_s{s}"
                info = cr.entrenar_gpu(secs, np.array([[0, 0]], dtype=np.int32), dict(cr.BASE, semilla=s), base, dev)
                prob, nb = cr.predictor([f"{base}.npz"], dev, cr.NT)
                fila = {**clave, "prueba": prueba, "lado": lado, **info, **cr.evaluar(prob, nb, temps, prueba, dev)}
                cr.escribir(CARPETA, fila)
                cr.log(CARPETA, f"red1x1 {ronda} s{s}: {cr.resumen(fila)}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
