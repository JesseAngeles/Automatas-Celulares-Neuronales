"""5.2 — Estados iniciales para la GUI con la discretización con la que se entrenaron las redes.

CA/FLAME3/seleccion1/sin_ignicion usa los cortes log15 de estados_log (notebooks/FLAME3.ipynb) (22.3–604.6 °C, primer corte 27.7 °C), pero todas
las redes desde ../../sin_p2 se entrenaron con los cortes de sin_p2/datos/log15/discretizacion.json (24.7–604.6 °C,
primer corte 30.6 °C): con los mismos °C, el 35 % de las celdas de p1 caen en otro estado. Este script escribe los
escenarios sin ignición con los cortes de sin_p2, a 100×100 y 200×200, con el formato de CA/ (evo.npy, evo.mat con la
variable evolution, evo_celsius.npy, evo.json, evo.gif, evo_final.png) y la paleta GUI/colors/flame_log15_temperatura.csv:

    CA/FLAME3/seleccion1/sin_p2_log15/flame_plot_{1,3}/       100×100
    CA/FLAME3/seleccion1/sin_p2_log15_200/flame_plot_{1,3}/   200×200

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/5_2_final/estados_gui.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402

PALETA = "GUI/colors/flame_log15_temperatura.csv"
SALIDA = {100: cr.TRAINING / "CA/FLAME3/seleccion1/sin_p2_log15", 200: cr.TRAINING / "CA/FLAME3/seleccion1/sin_p2_log15_200"}


def main():
    cr.log(CARPETA, "Estados iniciales para la GUI con la discretización de sin_p2")
    disc = json.load(open(cr.FLAME / "sin_p2" / "datos" / "log15" / "discretizacion.json"))
    colores = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]
                        for h in (cr.TRAINING.parent / PALETA).read_text().split()], dtype=np.uint8)
    for lado, raiz in SALIDA.items():
        temps = cr.temperaturas(lado)
        for k, t in temps.items():
            S = cr.estados(t).astype(np.int32)
            out = raiz / f"flame_plot_{k[1]}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "evo.npy", S)
            np.save(out / "evo_celsius.npy", t.astype(np.float32))
            scipy.io.savemat(out / "evo.mat", {"evolution": S})
            zoom = 4 if lado == 100 else 2
            frames = [Image.fromarray(colores[s].repeat(zoom, 0).repeat(zoom, 1)) for s in S]
            frames[0].save(out / "evo.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
            frames[-1].save(out / "evo_final.png")
            original = json.load(open(cr.TRAINING / f"CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{k[1]}/evo.json"))
            cortes = disc["cortes_c"]
            bordes = [disc.get("min_c", 24.7)] + cortes + [disc.get("max_c", 604.6)]
            meta = {
                "simulation": "flame3", "size": lado, "shape": list(S.shape),
                "source": f"Dataset/FLAME3/seleccion1/flame_plot_{k[1]}/geo_thermal_tiff_celsius_aligned_{lado}",
                "files": original["files"], "timestamps": original["timestamps"],
                "first_frame": cr.INICIO[k], "time_step_seconds": 5.0, "k": cr.NT,
                "discretization": "log15 de NCA/FLAME3/historial/sin_p2/datos/log15/discretizacion.json",
                "thresholds_celsius": cortes, "representatives_celsius": disc["representantes_c"],
                "states": [f"{s}: {bordes[s]:.1f} – {bordes[s + 1]:.1f} °C" for s in range(cr.NT)],
                "state_count": np.bincount(S.ravel(), minlength=cr.NT).tolist(),
                "palette": PALETA,
                "pipeline": original["pipeline"][:-2] + (["downsample_tiff.py --size 200"] if lado == 200 else [])
                + ["trim_start.py", "recomendaciones/5_2_final/estados_gui.py"],
            }
            with open(out / "evo.json", "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            cr.log(CARPETA, f"{lado}×{lado} {k}: {S.shape} → {out.relative_to(cr.TRAINING)}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
