"""2.2 — Sesgo de encendido: se suma una constante delta a la salida de los estados de fuego (8–14) antes del argmax.

    calibrado        delta elegido en el escenario de ENTRENAMIENTO para que se predigan tantos encendidos
                     (< 83 °C → ≥ 130 °C) como los reales; se evalúa en el otro escenario. Redes de sin_p2, 5 semillas.
                     Además se guarda el .npz con delta sumado al sesgo de la última capa (redes/) y se comprueba que
                     da las mismas predicciones que sumarlo por fuera: así lo usa la GUI sin cambios.
    d<x>             delta fijo (0.5, 1, 1.5, 2, 3), redes de sin_p2, 5 semillas: la curva entre "no enciende" y "se
                     desboca".
    ens5_calibrado   el ensamble de las 5 redes de sin_p2, delta calibrado igual.
    200_calibrado    redes de 200×200 con 9×9, delta calibrado a 200×200.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/2_2_sesgo/sesgo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402

FIJOS = [0.5, 1.0, 1.5, 2.0, 3.0]


def verificar_npz(red: Path, destino: Path, delta: float, temps, prueba, dev) -> int:
    """Celdas (de un frame de prueba) donde la red con el sesgo horneado difiere del sesgo sumado por fuera."""
    cr.guardar_con_sesgo(red, destino, delta)
    prob, nb = cr.predictor([red], dev, cr.NT)
    prob_h, _ = cr.predictor([destino], dev, cr.NT)
    S = cr.estados(temps[prueba][:2]).astype(np.float32)
    X, _ = cr.extract_patches(S, nb, True)
    a = cr.decodificar(cr.con_sesgo(prob, delta)(X), 1, "argmax")
    b = cr.decodificar(prob_h(X), 1, "argmax")
    return int((a != b).sum())


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 2.2 (sesgo de encendido)")
    e = "2_2_sesgo"
    for lado in (100, 200):
        temps = cr.temperaturas(lado)
        for entrena, prueba in cr.RONDAS:
            ronda = cr.ronda_nombre(entrena, prueba)
            secs = [s for k in entrena for s in cr.secs_offset(cr.estados(temps[k]))]
            tsecs = [t for k in entrena for t in cr.secs_offset(temps[k])]
            casos = []
            if lado == 100:
                casos += [("calibrado", s, [cr.red_referencia(ronda, s)]) for s in cr.SEMILLAS]
                casos += [("ens5_calibrado", "ensamble", [cr.red_referencia(ronda, s) for s in cr.SEMILLAS])]
                casos += [(f"d{d:g}", s, [cr.red_referencia(ronda, s)]) for d in FIJOS for s in cr.SEMILLAS]
            else:
                casos += [("200_calibrado", s, [cr.redes_200(ronda, s)]) for s in cr.SEMILLAS]
            for variante, semilla, redes in casos:
                clave = {"experimento": e, "variante": variante, "ronda": ronda, "semilla": semilla}
                if cr.hecho(CARPETA, clave):
                    continue
                prob, nb = cr.predictor(redes, dev, cr.NT)
                if "calibrado" in variante:
                    cal = cr.calibrar_sesgo(prob, nb, secs, tsecs)
                else:
                    cal = {"delta": float(variante[1:])}
                extra = {}
                if variante in ("calibrado", "200_calibrado"):
                    (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                    destino = CARPETA / "redes" / ronda / f"{Path(redes[0]).stem}_sesgo.npz"
                    extra["npz_diferencias"] = verificar_npz(redes[0], destino, cal["delta"], temps, prueba, dev)
                fila = {**clave, "prueba": prueba, "lado": lado, **cal, **extra,
                        **cr.evaluar(cr.con_sesgo(prob, cal["delta"]), nb, temps, prueba, dev, escala=lado // 100)}
                cr.escribir(CARPETA, fila)
                cr.log(CARPETA, f"{variante} {ronda} {semilla}: delta {cal['delta']}; {cr.resumen(fila)}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
