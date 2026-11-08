"""Pruebas para mejorar la alumna b16x128 sin quitar el dado (pedido 2026-11-04). Cada prueba cambia una cosa
de la configuración base y se evalúa con evaluar.py; se elige por el puntaje en las cajas de calibración
(entrenamiento) y se reporta en las de prueba. Una fila por red en barrido.csv.

    python barrido.py <fase> [<fase> …]
"""
import csv
import re
import sys
import time

import comun as C
import datos as Dt
import entrenar as E
import evaluar as Ev

BASE = {"capas": "16x128", "epocas": 30, "semilla": 42, "schedule": "exp", "reales": 20, "propios": 20,
        "crecer": 0.0}


def v(nombre, **c):
    return nombre, {**BASE, **c}


FASES = {
    "base": [v("b16x128")],                                    # ya entrenada: solo se reevalúa
    "semillas": [v("s1", semilla=1), v("s2", semilla=2)],
    "red": [v("e60", epocas=60), v("e100", epocas=100), v("r8x128", capas="8x128"), v("r24x128", capas="24x128"),
            v("r16x256", capas="16x256"), v("cos", schedule="cosine")],
    "datos": [v("c015", crecer=0.15), v("c025", crecer=0.25), v("p30r10", reales=10, propios=30),
              v("r30p10", reales=30, propios=10)],
    # combinaciones de lo mejor de las fases anteriores, con 3 semillas las candidatas
    "final": [v("cos_s1", schedule="cosine", semilla=1), v("cos_s2", schedule="cosine", semilla=2),
              v("c015cos", schedule="cosine", crecer=0.15), v("c015cos_s1", schedule="cosine", crecer=0.15, semilla=1),
              v("c015cos_s2", schedule="cosine", crecer=0.15, semilla=2), v("c025_s1", crecer=0.25, semilla=1),
              v("c025_s2", crecer=0.25, semilla=2), v("cos_e60", schedule="cosine", epocas=60)],
}
CSV = C.CARPETA / "barrido.csv"


def carpeta_datos(c):
    return f"datos_{c['reales']}_{c['propios']}_{c['crecer']:g}"


def correr(nombre, c):
    hechas = {f["nombre"] for f in csv.DictReader(open(CSV))} if CSV.exists() else set()
    if nombre in hechas:
        return
    d = carpeta_datos(c)
    if not (C.CARPETA / d).exists():
        Dt.generar(d, c["reales"], c["propios"], c["crecer"])
    t0 = time.perf_counter()
    if not (C.CARPETA / "redes" / nombre / "red.npz").exists():
        E.entrenar(nombre, c["capas"], c["epocas"], c["semilla"], d, c["schedule"])
    t = time.perf_counter() - t0
    log = (C.CARPETA / "redes" / nombre / "entrenamiento.log").read_text()
    perdida = float(re.findall(r"loss: ([\d.]+)", log)[-1])
    r = Ev.evaluar(nombre, c["crecer"])
    fila = {"nombre": nombre, **c, "perdida": perdida, "t_entrenar_s": round(t),
            **{f"paso_{k}": round(x, 4) for k, x in r["un_paso"].items()},
            **{f"cal_{k}": round(x, 4) for k, x in r["promedio_cal"].items()},
            **{f"prueba_{k}": round(x, 4) for k, x in r["promedio_libre"].items()}}
    nuevo = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


if __name__ == "__main__":
    for fase in sys.argv[1:]:
        for nombre, c in FASES[fase]:
            print(f"── {fase} {nombre} {c}", flush=True)
            correr(nombre, c)
