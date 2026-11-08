"""Opción 8 — pasos más largos: 9 y 12 frames (45 y 60 s) en lugar de 6 (30 s).

Vecindades 9×9 y 13×13 (en 60 s el fuego puede avanzar más). Resto igual que la referencia (../0_base): log15, cruda,
2 capas de 32, 200 épocas, exp 1e-3 → 1e-5, p1 → p3 y p3 → p1, semillas 42, 1, 2, 3, 4. Entrena con scripts/main.py.
La simulación libre dura lo más cercano a 120 frames (10 min): 13 pasos de 9 frames (117) o 10 pasos de 12 (120);
la columna rollout_frames dice cuántos.

Escribe datos/, redes/<ronda>/, resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/8_pasos/pasos.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun as cm                                                           # noqa: E402

PASOS = [9, 12]
RADIOS = [4, 6]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = cm.temperaturas()
    cm.log(CARPETA, "Inicio opción 8 (pasos de 9 y 12 frames)")
    S = {k: cm.estados(t) for k, t in temps.items()}
    for paso in PASOS:
        rutas = cm.secuencias(S, paso, CARPETA / "datos", "log15")
        for radio in RADIOS:
            vec = f"{2 * radio + 1}x{2 * radio + 1}"
            for entrena, prueba in cm.RONDAS:
                ronda = cm.ronda_nombre(entrena, prueba)
                for s in cm.SEMILLAS:
                    clave = {"opcion": "8_pasos", "variante": f"k{paso}_{vec}", "ronda": ronda, "semilla": s}
                    if cm.hecho(CARPETA, clave):
                        continue
                    (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                    base = CARPETA / "redes" / ronda / f"log15_k{paso}_r{radio}_cruda_2x32_exp_s{s}"
                    cm.log(CARPETA, f"paso {paso} {vec} {ronda} s{s}: entrenando")
                    info = cm.entrenar_main(dict(cm.BASE, radio=radio, semilla=s),
                                            [r for k in entrena for r in rutas[k]], base, cm.NT, CARPETA)
                    prob, nb = cm.predictor([f"{base}.npz"], dev, cm.NT)
                    fila = {**clave, "prueba": prueba, "paso_frames": paso, "decodificador": "argmax",
                            "vecindad": vec, **info, **cm.evaluar(prob, nb, temps, prueba, dev, paso=paso)}
                    cm.escribir(CARPETA, fila)
                    cm.log(CARPETA, f"    {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
