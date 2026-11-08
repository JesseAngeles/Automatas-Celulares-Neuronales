"""4.1 (extra) — Mapa de combustible + sesgo de encendido calibrado (2.2), sin reentrenar.

Con el canal fijo la red casi no enciende (encendido ≤ 0.014), así que no se puede saber si el combustible ayuda a
decidir DÓNDE se enciende. Aquí se toma cada red de combustible.py, se calibra delta en el escenario de entrenamiento
(tantos encendidos predichos como reales, igual que ../2_2_sesgo) y se evalúa en el otro. Comparar con la variante
"calibrado" de ../2_2_sesgo (la referencia sin combustible con el mismo procedimiento).

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/4_1_combustible/combustible_sesgo.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402
from combustible import clases                                            # noqa: E402

VARIANTES = ["altura2", "altura3", "verdor2", "altura_verdor4"]


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 4.1 extra (combustible + sesgo calibrado)")
    temps = cr.temperaturas()
    C = clases()
    for v in VARIANTES:
        d, n = C[v], C[v]["n"]
        for entrena, prueba in cr.RONDAS:
            ronda = cr.ronda_nombre(entrena, prueba)
            secs = [s for k in entrena for s in cr.secs_offset(cr.estados(temps[k]) + cr.NT * d[k])]
            tsecs = [t for k in entrena for t in cr.secs_offset(temps[k])]
            for s in cr.SEMILLAS:
                clave = {"experimento": "4_1_combustible", "variante": f"{v}_sesgo", "ronda": ronda, "semilla": s}
                if cr.hecho(CARPETA, clave):
                    continue
                red = CARPETA / "redes" / ronda / f"{v}_k6_r4_cruda_2x32_exp_s{s}.npz"
                prob, nb = cr.predictor([red], dev, cr.NT * n)
                cal = cr.calibrar_sesgo(prob, nb, secs, tsecs)
                fila = {**clave, "prueba": prueba, "lado": 100, "clases": n, **cal,
                        **cr.evaluar(cr.con_sesgo(prob, cal["delta"]), nb, temps, prueba, dev,
                                     canal={k: d[k] for k in cr.ESC}, ncanal=n)}
                cr.escribir(CARPETA, fila)
                cr.log(CARPETA, f"{v}_sesgo {ronda} s{s}: delta {cal['delta']}; {cr.resumen(fila)}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
