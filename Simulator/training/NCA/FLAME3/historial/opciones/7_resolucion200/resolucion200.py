"""Opción 7 — más resolución: 200×200 en lugar de 100×100.

Datos: Dataset/FLAME3/seleccion1/flame_plot_{1,3}/geo_thermal_tiff_celsius_aligned_200 (reducir de notebooks/FLAME3.ipynb con size=200,
gaussiano, a partir de los mismos recortes cuadrados alineados que la versión 100×100), en el orden de
CA/FLAME3/seleccion1/flame_plot_N/evo.json y recortados como CA/FLAME3/seleccion1/sin_ignicion (p1 desde 133, p3 desde 19). Misma
discretización log15 que la referencia. Las temperaturas se guardan en datos/pN_celsius_200.npy.

Vecindades: 9×9 (mismo número de celdas que la referencia, la mitad del área en metros) y 17×17 (la misma área
en metros que el 9×9 de 100×100). Resto igual que la referencia (../0_base): paso 6, cruda, 2 capas de 32, 200
épocas, exp 1e-3 → 1e-5, p1 → p3 y p3 → p1. Semillas: 42, 1, 2, 3, 4 con 9×9; 42, 1, 2 con 17×17 (más lento).
Las métricas en °C se calculan sobre la malla de 200×200; para comparar con 100×100 conviene usar la proporción
contra la copia.

9×9 se entrena con scripts/main.py. 17×17 se entrena con el entrenador de ../5_pesos (α = 0, que da exactamente los
mismos pesos que main.py): con 17×17 los datos ocupan ~7 GB y main.py los sube a la GPU porque compara contra la
VRAM total, pero la copia barajada de cada época ya no cabe (CUDA out of memory).

Escribe datos/, redes/<ronda>/, resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/7_resolucion200/resolucion200.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun as cm                                                           # noqa: E402

sys.path.insert(0, str(CARPETA.parent / "5_pesos"))
import pesos                                                                 # noqa: E402

CONFIGS = {4: [42, 1, 2, 3, 4], 8: [42, 1, 2]}        # radio → semillas


def temperaturas_200() -> dict:
    temps = {}
    (CARPETA / "datos").mkdir(parents=True, exist_ok=True)
    for k in cm.ESC:
        destino = CARPETA / "datos" / f"{k}_celsius_200.npy"
        if destino.exists():
            temps[k] = np.load(destino)
            continue
        archivos = json.load(open(cm.TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo.json"))["files"]
        carpeta = cm.TRAINING / f"Dataset/FLAME3/seleccion1/flame_plot_{k[1]}/geo_thermal_tiff_celsius_aligned_200"
        completo = np.stack([rasterio.open(carpeta / a).read(1) for a in archivos]).astype(np.float32)
        ref = np.load(cm.TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo_celsius.npy")
        dif = np.abs(completo.mean((1, 2)) - ref.mean((1, 2))).max()
        cm.log(CARPETA, f"{k}: {completo.shape}, diferencia máxima de la media por frame contra 100×100: {dif:.2f} °C")
        temps[k] = completo[cm.INICIO[k]:]
        np.save(destino, temps[k])
    return temps


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm.log(CARPETA, "Inicio opción 7 (200×200)")
    temps = temperaturas_200()
    rutas = cm.secuencias({k: cm.estados(t) for k, t in temps.items()}, cm.PASO, CARPETA / "datos", "log15_200")
    for radio, semillas in CONFIGS.items():
        for entrena, prueba in cm.RONDAS:
            ronda = cm.ronda_nombre(entrena, prueba)
            for s in semillas:
                vec = f"{2 * radio + 1}x{2 * radio + 1}"
                clave = {"opcion": "7_resolucion200", "variante": f"200_{vec}", "ronda": ronda, "semilla": s}
                if cm.hecho(CARPETA, clave):
                    continue
                (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                base = CARPETA / "redes" / ronda / f"200_k6_r{radio}_cruda_2x32_exp_s{s}"
                cm.log(CARPETA, f"{vec} {ronda} s{s}: entrenando")
                c = dict(cm.BASE, radio=radio, semilla=s)
                datos = [r for k in entrena for r in rutas[k]]
                if radio <= 4:
                    info = cm.entrenar_main(c, datos, base, cm.NT, CARPETA)
                else:
                    cm.log(CARPETA, "    entrenador de ../5_pesos con α = 0 (main.py se queda sin memoria en la GPU)")
                    nb = np.array([(dx, dy) for dy in range(-radio, radio + 1) for dx in range(-radio, radio + 1)],
                                  dtype=np.int32)          # el mismo orden que main.build_neighbourhood
                    p = pesos.entrenar(datos, 0, c, base, dev, nb)
                    info = {"epocas_hechas": p["epocas_hechas"], "loss_final": p["loss_final"],
                            "lr_final": c["lr_min"], "tiempo_s": p["tiempo_s"]}
                prob, nb = cm.predictor([f"{base}.npz"], dev, cm.NT)
                fila = {**clave, "prueba": prueba, "paso_frames": cm.PASO, "decodificador": "argmax", "lado": 200,
                        "vecindad": vec, **info, **cm.evaluar(prob, nb, temps, prueba, dev)}
                cm.escribir(CARPETA, fila)
                cm.log(CARPETA, f"    {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
