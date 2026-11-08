"""5.1 — Comprobación del cambio de memoria en scripts/main.py.

Cambio: main.py decide subir los datos a la GPU comparando con la VRAM LIBRE (antes: la total) y toma cada lote con
X[perm[s:s+b]] en lugar de copiar X[perm] completo en cada época. Los lotes son los mismos, así que los pesos deben
salir idénticos.

    referencia_s42   reentrena con main.py la red de ../../sin_p2/semillas/p1_a_p3/log15_k6_r4_cruda_2x32_exp_s42.npz
                     y compara todos los arreglos del .npz
    200_17x17_s42    entrena con main.py el 17×17 a 200×200 (p1 → p3, semilla 42), que antes se quedaba sin memoria,
                     y lo compara con la red que se entrenó con el entrenador de ../../opciones/5_pesos (α = 0)
Escribe resultados.csv y registro.log.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/5_1_memoria/memoria.py
"""
import sys
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402


def comparar(a: Path, b: Path) -> float:
    za, zb = np.load(a), np.load(b)
    assert sorted(za.files) == sorted(zb.files), (za.files, zb.files)
    return max(float(np.abs(za[k].astype(np.float64) - zb[k].astype(np.float64)).max()) for k in za.files)


def main():
    cr.log(CARPETA, "Inicio 5.1 (memoria de main.py)")
    (CARPETA / "redes").mkdir(exist_ok=True)
    casos = [
        ("referencia_s42", 4, [cr.FLAME / "sin_p2" / "datos" / "log15" / f"p1_k6_o{o}.npy" for o in range(6)],
         cr.red_referencia("p1_a_p3", 42)),
        ("200_17x17_s42", 8, [cr.OPC / "7_resolucion200" / "datos" / f"log15_200_p1_k6_o{o}.npy" for o in range(6)],
         cr.OPC / "7_resolucion200" / "redes" / "p1_a_p3" / "200_k6_r8_cruda_2x32_exp_s42.npz"),
    ]
    for nombre, radio, datos, original in casos:
        clave = {"experimento": "5_1_memoria", "caso": nombre}
        if cr.hecho(CARPETA, clave):
            continue
        base = CARPETA / "redes" / nombre
        info = cr.cm.entrenar_main(dict(cr.BASE, radio=radio, semilla=42), datos, base, cr.NT, CARPETA)
        dif = comparar(Path(f"{base}.npz"), original)
        fila = {**clave, "radio": radio, **info, "original": str(original.relative_to(cr.TRAINING)),
                "max_diferencia_pesos": dif, "identicos": dif == 0.0}
        cr.escribir(CARPETA, fila)
        cr.log(CARPETA, f"{nombre}: {info}; diferencia máxima contra el original {dif}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
