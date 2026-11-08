"""NCA para FLAME 3 sin ignición y sin p2, con más estados.

p2 se quita porque casi no tiene fuego (97 % de las celdas no cambia): castiga a cualquier red que prediga fuego.
Datos: CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{1,3}/evo_celsius.npy (p1 desde el frame 133, p3 desde el 19).
Validación cruzada con dos rondas: p1 → p3 y p3 → p1.

Discretizaciones logN (N estados con cortes iguales en log(°C), representante = centro geométrico) con la
escala de p1 y p3 sin ignición (24.7–604.6 °C). Configuraciones: entrada cruda, 2 capas de 32 ReLU, pasos de
1, 3 y 6 frames (5, 15, 30 s) con dos vecindades por paso; 200 épocas, exp 1e-3 → 1e-5, lote 8192, frontera abierta.

Partes (cada una con su CSV en esta carpeta; registro común en registro.log):
    pasos.csv      discretizaciones × pasos × vecindades × rondas (semilla 42)
    semillas.csv   la mejor de pasos.csv con semillas 42, 1, 2, 3, 4
    final.csv      la mejor entrenada con p1 y p3
La mejor es la de menor promedio (en las 2 rondas) de mae_c_prueba / copia_mae_c_prueba; el error está en °C,
así que es comparable entre discretizaciones.

Métricas: las de ../k6/barrido_k6.py (exact, exact_cambia, mae_c en °C, encendido, cuantizacion_mae_c,
rollout20_mae_c y copia_*), con sufijos _ent1 / _prueba. exact solo es comparable con la misma discretización.

No modifica el simulador: llama a scripts/main.py y reutiliza ../cruzado/barrido_cruzado.py y ../k6/barrido_k6.py.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/sin_p2/barrido_sin_p2.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[3]
sys.path.insert(0, str(CARPETA.parent / "k6"))
import barrido_k6 as k6                                                      # noqa: E402

bc = k6.bc
bc.CARPETA = k6.CARPETA = CARPETA            # registro.log en esta carpeta
log, entrenar = bc.log, bc.entrenar

ESC = ["p1", "p3"]
RONDAS = [(("p1",), "p3"), (("p3",), "p1")]
DISCRETIZACIONES = ["log15", "log20", "log30", "log50"]
PASOS = {1: (1, 3), 3: (3, 5), 6: (4, 6)}     # paso en frames → radios probados
BASE = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=200, lote=8192, frontera="abierta",
            entrada="cruda", capas=2, neuronas=32, semilla=42)
SEMILLAS = [42, 1, 2, 3, 4]


def nombre(disc: str, c: dict) -> str:
    return (f"{disc}_k{c['paso']}_r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}"
            f"_{c['schedule']}_s{c['semilla']}")


def preparar(disc_nombre: str, temps: dict) -> dict:
    """Cortes, representantes y secuencias con desfase por paso: rutas[paso][escenario] = [archivos]."""
    cortes, rep = k6.discretizacion(disc_nombre, temps)
    carpeta = CARPETA / "datos" / disc_nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    with open(carpeta / "discretizacion.json", "w") as f:
        json.dump({"nombre": disc_nombre, "cortes_c": cortes.tolist(), "representantes_c": rep.tolist()}, f, indent=2)
    rutas = {}
    for paso in PASOS:
        rutas[paso] = {}
        for k, t in temps.items():
            estados = np.digitize(t, cortes).astype(np.int32)
            rutas[paso][k] = []
            for o in range(paso):
                destino = carpeta / f"{k}_k{paso}_o{o}.npy"
                np.save(destino, estados[o::paso])
                rutas[paso][k].append(destino)
    return {"cortes": cortes, "rep": rep, "rutas": rutas}


def correr(parte: str, disc_nombre: str, disc: dict, c: dict, entrena: tuple, prueba, temps: dict, dev):
    destino = CARPETA / f"{parte}.csv"
    ronda = f"{'+'.join(entrena)}_a_{prueba}" if prueba else "todos"
    nom = nombre(disc_nombre, c)
    if destino.exists() and any(r["ronda"] == ronda and r["nombre"] == nom for r in csv.DictReader(open(destino))):
        log(f"{parte} {ronda} {nom}: ya está en {destino.name}, se omite")
        return
    carpeta = CARPETA / parte / ronda
    carpeta.mkdir(parents=True, exist_ok=True)
    base = carpeta / nom
    log(f"{parte} {ronda} {nom}: entrenando")
    t0 = time.time()
    fila = {"parte": parte, "discretizacion": disc_nombre, "estados": len(disc["rep"]), "ronda": ronda,
            "entrena": "+".join(entrena), "prueba": prueba or "", "nombre": nom, "paso_frames": c["paso"],
            "segundos": 5 * c["paso"], **{k: v for k, v in c.items() if k != "paso"},
            "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
    bc.ESTADOS = len(disc["rep"])      # --out-size de main.py: una neurona por estado
    fila.update(entrenar(c, [r for k in entrena for r in disc["rutas"][c["paso"]][k]], base))
    log(f"    entrenada: {fila['epocas_hechas']} épocas, loss {fila['loss_final']:.4f}, {fila['tiempo_s']} s; evaluando")
    escenarios = {f"ent{i + 1}": k for i, k in enumerate(entrena)}
    if prueba:
        escenarios["prueba"] = prueba
    k6.PASO = c["paso"]
    fila.update(k6.evaluar(c, base, disc, temps, escenarios, dev))
    nuevo = not destino.exists()
    with open(destino, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)
    e = "prueba" if prueba else "ent1"
    log(f"    {escenarios[e]}: exact {fila[f'exact_{e}']:.4f} (copia {fila[f'copia_exact_{e}']:.4f}), "
        f"mae {fila[f'mae_c_{e}']:.2f} °C (copia {fila[f'copia_mae_c_{e}']:.2f}, cuantización "
        f"{fila[f'cuantizacion_mae_c_{e}']:.2f}), encendido {fila[f'encendido_{e}']:.3f}, rollout20 "
        f"{fila[f'rollout20_mae_c_{e}']:.2f} °C (copia {fila[f'copia20_mae_c_{e}']:.2f})  [{time.time() - t0:.0f} s]")


def mejor() -> tuple:
    filas = list(csv.DictReader(open(CARPETA / "pasos.csv")))
    por_conf = {}
    for f in filas:
        por_conf.setdefault(f["nombre"], []).append(float(f["mae_c_prueba"]) / float(f["copia_mae_c_prueba"]))
    completos = {k: np.mean(v) for k, v in por_conf.items() if len(v) == len(RONDAS)}
    nom = min(completos, key=completos.get)
    f = next(f for f in filas if f["nombre"] == nom)
    c = dict(BASE, paso=int(f["paso_frames"]), radio=int(f["radio"]))
    log(f"Mejor configuración: {nom} (mae / copia promedio {completos[nom]:.4f})")
    return f["discretizacion"], c


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = {k: np.load(TRAINING / f"CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{k[1]}/evo_celsius.npy") for k in ESC}
    log(f"Inicio barrido sin p2 ({dev}): " + ", ".join(f"{k} {len(t)} frames" for k, t in temps.items())
        + f"; discretizaciones {DISCRETIZACIONES}, pasos {list(PASOS)}, {len(RONDAS)} rondas")
    discs = {}
    for d in DISCRETIZACIONES:
        discs[d] = preparar(d, temps)
        log(f"Discretización {d}: cortes °C " + ", ".join(f"{x:.1f}" for x in discs[d]["cortes"]))

    log("Parte 1: discretizaciones, pasos y vecindades (semilla 42)")
    for d in DISCRETIZACIONES:
        for paso, radios in PASOS.items():
            for radio in radios:
                for entrena, prueba in RONDAS:
                    correr("pasos", d, discs[d], dict(BASE, paso=paso, radio=radio), entrena, prueba, temps, dev)

    d, c = mejor()
    log("Parte 2: semillas")
    for s in SEMILLAS:
        for entrena, prueba in RONDAS:
            correr("semillas", d, discs[d], dict(c, semilla=s), entrena, prueba, temps, dev)

    log("Parte 3: red final con p1 y p3")
    correr("final", d, discs[d], c, ("p1", "p3"), None, temps, dev)
    log("Fin")


if __name__ == "__main__":
    main()
