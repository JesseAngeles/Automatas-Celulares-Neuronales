"""4.1 (extra) — Mapa de combustible con la clase intercalada: estado = n·T + clase en lugar de T + 15·clase.

Con entrada cruda la red recibe el número de estado. Con T + 15·clase, una celda fría de la clase 1 (estado 15) vale
más que el fuego más intenso de la clase 0 (estado 14): el orden de temperatura se rompe entre clases. Con n·T + clase
el número sigue creciendo con la temperatura y la clase solo lo mueve dentro del mismo escalón. Sigue siendo un solo
estado (15·n estados), así que la GUI lo usa igual.

Mismas clases que combustible.py (altura2, altura3, verdor2), main.py con --out-size 15·n, 5 semillas. Para evaluar
con ../comun_r.py (que usa T + 15·clase) se traducen la entrada y la salida de la red (intercalado()).
    <v>_int          clase fija en la simulación
    <v>_int_sesgo    con el sesgo de encendido calibrado en el escenario de entrenamiento (como ../2_2_sesgo)

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/4_1_combustible/combustible_intercalado.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402
from combustible import clases                                            # noqa: E402

VARIANTES = ["altura2", "altura3", "verdor2"]


def intercalado(prob, n: int):
    """Red entrenada con n·T + c → función de probabilidad con entrada y salida en T + 15·c."""
    T = np.arange(cr.NT * n) % cr.NT
    c = np.arange(cr.NT * n) // cr.NT
    orden = torch.tensor(n * T + c)                 # salida en T + 15·c ← columna n·T + c de la red

    def p2(X):
        Xi = n * (X % cr.NT) + X // cr.NT
        p = prob(np.ascontiguousarray(Xi, dtype=np.float32))
        return p[:, orden.to(p.device)]
    return p2


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 4.1 extra (combustible con la clase intercalada)")
    temps = cr.temperaturas()
    C = clases()
    for v in VARIANTES:
        d, n = C[v], C[v]["n"]
        rutas = cr.secuencias({k: n * cr.estados(temps[k]) + d[k] for k in cr.ESC}, cr.PASO, CARPETA / "datos",
                              f"{v}_int")
        canal = {k: d[k] for k in cr.ESC}
        for entrena, prueba in cr.RONDAS:
            ronda = cr.ronda_nombre(entrena, prueba)
            secs = [s for k in entrena for s in cr.secs_offset(cr.estados(temps[k]) + cr.NT * d[k])]
            tsecs = [t for k in entrena for t in cr.secs_offset(temps[k])]
            for s in cr.SEMILLAS:
                base = CARPETA / "redes" / ronda / f"{v}_int_k6_r4_cruda_2x32_exp_s{s}"
                clave = {"experimento": "4_1_combustible", "variante": f"{v}_int", "ronda": ronda, "semilla": s}
                if not cr.hecho(CARPETA, clave):
                    cr.log(CARPETA, f"{v}_int {ronda} s{s}: entrenando con main.py")
                    info = cr.cm.entrenar_main(dict(cr.BASE, semilla=s), [r for k in entrena for r in rutas[k]],
                                               base, cr.NT * n, CARPETA)
                    prob, nb = cr.predictor([f"{base}.npz"], dev, cr.NT * n)
                    fila = {**clave, "prueba": prueba, "lado": 100, "clases": n, **info,
                            **cr.evaluar(intercalado(prob, n), nb, temps, prueba, dev, canal=canal, ncanal=n)}
                    cr.escribir(CARPETA, fila)
                    cr.log(CARPETA, f"    {cr.resumen(fila)}")
                clave = {**clave, "variante": f"{v}_int_sesgo"}
                if not cr.hecho(CARPETA, clave):
                    prob, nb = cr.predictor([f"{base}.npz"], dev, cr.NT * n)
                    p = intercalado(prob, n)
                    cal = cr.calibrar_sesgo(p, nb, secs, tsecs)
                    fila = {**clave, "prueba": prueba, "lado": 100, "clases": n, **cal,
                            **cr.evaluar(cr.con_sesgo(p, cal["delta"]), nb, temps, prueba, dev, canal=canal, ncanal=n)}
                    cr.escribir(CARPETA, fila)
                    cr.log(CARPETA, f"{v}_int_sesgo {ronda} s{s}: delta {cal['delta']}; {cr.resumen(fila)}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
