"""Opción 3 — mapa de terreno estático sacado de los frames anteriores a la ignición.

Antes de prender el fuego, la cámara térmica ya distingue zonas más calientes (suelo expuesto al sol) y más frías
(vegetación, sombra, humedad). Por celda se toma la mediana de °C en los frames previos a la ignición del escenario
completo (CA/FLAME3/seleccion1/flame_plot_N: p1 frames 0–122, p3 frames 0–13) y se divide en clases dentro de cada escenario:
    mediana2   2 clases: abajo / arriba de la mediana del escenario
    terciles3  3 clases por terciles
Estado de la NCA = T + 15 · clase (30 o 45 estados). En la simulación libre la clase se mantiene fija (es un dato
conocido del terreno); la red solo decide la temperatura.

Resto igual que la referencia (../0_base): 9×9, paso 6, cruda, 2 capas de 32, 200 épocas, exp 1e-3 → 1e-5,
p1 → p3 y p3 → p1, semillas 42, 1, 2, 3, 4. Entrena con scripts/main.py.
Escribe mapas.png, datos/, redes/<ronda>/, resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/3_terreno/terreno.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun as cm                                                           # noqa: E402

PRE = {"p1": 123, "p3": 14}            # frames previos a la ignición en el escenario completo
VARIANTES = {"mediana2": 2, "terciles3": 3}


def mapas() -> dict:
    """{variante: {escenario: clase (H, W)}} y guarda mapas.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    medianas = {k: np.median(np.load(cm.TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo_celsius.npy")[:PRE[k]], 0)
                for k in cm.ESC}
    salida = {}
    for v, n in VARIANTES.items():
        salida[v] = {}
        for k, m in medianas.items():
            cortes = np.percentile(m, np.linspace(0, 100, n + 1)[1:-1])
            salida[v][k] = np.digitize(m, cortes).astype(np.int32)
    fig, ax = plt.subplots(len(cm.ESC), 1 + len(VARIANTES), figsize=(4 * (1 + len(VARIANTES)), 4 * len(cm.ESC)))
    for i, k in enumerate(cm.ESC):
        im = ax[i, 0].imshow(medianas[k], cmap="viridis", vmax=np.percentile(medianas[k], 99))
        ax[i, 0].set_title(f"{k}: mediana °C antes de la ignición")
        plt.colorbar(im, ax=ax[i, 0])
        for j, v in enumerate(VARIANTES):
            ax[i, j + 1].imshow(salida[v][k], cmap="viridis")
            ax[i, j + 1].set_title(f"{k}: {v}")
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(CARPETA / "mapas.png", dpi=80)
    return salida


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = cm.temperaturas()
    cm.log(CARPETA, "Inicio opción 3 (mapa de terreno pre-ignición)")
    clases = mapas()
    for v, n in VARIANTES.items():
        S = {k: cm.estados(t) + cm.NT * clases[v][k] for k, t in temps.items()}
        rutas = cm.secuencias(S, cm.PASO, CARPETA / "datos", v)
        for entrena, prueba in cm.RONDAS:
            ronda = cm.ronda_nombre(entrena, prueba)
            for s in cm.SEMILLAS:
                clave = {"opcion": "3_terreno", "variante": v, "ronda": ronda, "semilla": s}
                if cm.hecho(CARPETA, clave):
                    continue
                (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                base = CARPETA / "redes" / ronda / f"{v}_k6_r4_cruda_2x32_exp_s{s}"
                c = dict(cm.BASE, semilla=s)
                cm.log(CARPETA, f"{v} {ronda} s{s}: entrenando")
                info = cm.entrenar_main(c, [r for k in entrena for r in rutas[k]], base, cm.NT * n, CARPETA)
                prob, nb = cm.predictor([f"{base}.npz"], dev, cm.NT * n)
                fila = {**clave, "prueba": prueba, "paso_frames": cm.PASO, "decodificador": "argmax",
                        "estados": cm.NT * n, **info,
                        **cm.evaluar(prob, nb, temps, prueba, dev, canal=clases[v], ncanal=n)}
                cm.escribir(CARPETA, fila)
                cm.log(CARPETA, f"    {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
