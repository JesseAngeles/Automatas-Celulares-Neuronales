"""NCA para FLAME 3 (sin ignición, sin p2) con el entrenamiento aumentado por rotaciones de 90°, 180° y 270°.

Mismo esquema que ../sin_p2: p1 → p3 y p3 → p1, log15 con la escala de p1 y p3 sin ignición (24.7–604.6 °C).
Con aumento "rot4" cada escenario de entrenamiento entra cuatro veces (0°, 90°, 180°, 270°; las rotaciones se
aplican a toda la secuencia); el escenario de prueba nunca se rota. Con "sin" solo entra el original, para
comparar cada configuración con y sin rotaciones.

Configuraciones (200 épocas, exp 1e-3 → 1e-5, lote 8192, frontera abierta, semilla 42):
    aumento sin / rot4 × pasos de 1, 3, 6 frames (dos vecindades por paso) × entrada cruda / one-hot
    × 2 capas de 32 / 3 capas de 128.

Partes (cada una con su CSV en esta carpeta; registro común en registro.log):
    pasos.csv      todas las configuraciones en las dos rondas
    semillas.csv   la mejor con rot4 con semillas 42, 1, 2, 3, 4
    final.csv      esa misma entrenada con p1 y p3 (con rotaciones)
La mejor es la de menor promedio (en las 2 rondas) de mae_c_prueba / copia_mae_c_prueba.
Métricas: las de ../k6/barrido_k6.py con sufijos _ent1 (escenario de entrenamiento sin rotar) y _prueba.

No modifica el simulador: llama a scripts/main.py y reutiliza ../cruzado/barrido_cruzado.py y ../k6/barrido_k6.py.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/rotaciones/barrido_rotaciones.py
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
DISC = "log15"
AUMENTOS = {"sin": (0,), "rot4": (0, 1, 2, 3)}   # número de giros de 90° (np.rot90 sobre los ejes de la malla)
PASOS = {1: (1, 3), 3: (3, 5), 6: (4, 6)}         # paso en frames → radios probados
BASE = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=200, lote=8192, frontera="abierta", semilla=42)
SEMILLAS = [42, 1, 2, 3, 4]


def configuraciones() -> list:
    return [dict(BASE, aumento=a, paso=p, radio=r, entrada=e, capas=capas, neuronas=n)
            for a in AUMENTOS for p, radios in PASOS.items() for r in radios
            for e in ("cruda", "onehot") for capas, n in ((2, 32), (3, 128))]


def nombre(c: dict) -> str:
    return (f"{DISC}_{c['aumento']}_k{c['paso']}_r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}"
            f"_{c['schedule']}_s{c['semilla']}")


def preparar(temps: dict) -> dict:
    """Cortes, representantes y rutas[aumento][paso][escenario] = [archivos] (desfases × giros)."""
    cortes, rep = k6.discretizacion(DISC, temps)
    carpeta = CARPETA / "datos"
    carpeta.mkdir(parents=True, exist_ok=True)
    with open(carpeta / "discretizacion.json", "w") as f:
        json.dump({"nombre": DISC, "cortes_c": cortes.tolist(), "representantes_c": rep.tolist()}, f, indent=2)
    archivos = {}
    for k, t in temps.items():
        estados = np.digitize(t, cortes).astype(np.int32)
        for giro in (0, 1, 2, 3):
            girado = np.ascontiguousarray(np.rot90(estados, giro, axes=(1, 2)))
            for paso in PASOS:
                for o in range(paso):
                    destino = carpeta / f"{k}_rot{90 * giro}_k{paso}_o{o}.npy"
                    np.save(destino, girado[o::paso])
                    archivos[(k, giro, paso, o)] = destino
    rutas = {a: {p: {k: [archivos[(k, g, p, o)] for g in giros for o in range(p)] for k in temps}
                 for p in PASOS} for a, giros in AUMENTOS.items()}
    return {"cortes": cortes, "rep": rep, "rutas": rutas}


def correr(parte: str, disc: dict, c: dict, entrena: tuple, prueba, temps: dict, dev):
    destino = CARPETA / f"{parte}.csv"
    ronda = f"{'+'.join(entrena)}_a_{prueba}" if prueba else "todos"
    nom = nombre(c)
    if destino.exists() and any(r["ronda"] == ronda and r["nombre"] == nom for r in csv.DictReader(open(destino))):
        log(f"{parte} {ronda} {nom}: ya está en {destino.name}, se omite")
        return
    carpeta = CARPETA / parte / ronda
    carpeta.mkdir(parents=True, exist_ok=True)
    base = carpeta / nom
    log(f"{parte} {ronda} {nom}: entrenando")
    t0 = time.time()
    fila = {"parte": parte, "discretizacion": DISC, "ronda": ronda, "entrena": "+".join(entrena),
            "prueba": prueba or "", "nombre": nom, "aumento": c["aumento"], "paso_frames": c["paso"],
            "segundos": 5 * c["paso"], **{k: v for k, v in c.items() if k not in ("paso", "aumento")},
            "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
    bc.ESTADOS = len(disc["rep"])
    fila.update(entrenar(c, [r for k in entrena for r in disc["rutas"][c["aumento"]][c["paso"]][k]], base))
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
        f"mae {fila[f'mae_c_{e}']:.2f} °C (copia {fila[f'copia_mae_c_{e}']:.2f}), encendido "
        f"{fila[f'encendido_{e}']:.3f}, rollout20 {fila[f'rollout20_mae_c_{e}']:.2f} °C "
        f"(copia {fila[f'copia20_mae_c_{e}']:.2f})  [{time.time() - t0:.0f} s]")


def mejor(aumento: str) -> dict:
    filas = [f for f in csv.DictReader(open(CARPETA / "pasos.csv")) if f["aumento"] == aumento]
    por_conf = {}
    for f in filas:
        por_conf.setdefault(f["nombre"], []).append(float(f["mae_c_prueba"]) / float(f["copia_mae_c_prueba"]))
    completos = {k: np.mean(v) for k, v in por_conf.items() if len(v) == len(RONDAS)}
    nom = min(completos, key=completos.get)
    f = next(f for f in filas if f["nombre"] == nom)
    c = dict(BASE, aumento=aumento, paso=int(f["paso_frames"]), radio=int(f["radio"]), entrada=f["entrada"],
             capas=int(f["capas"]), neuronas=int(f["neuronas"]))
    log(f"Mejor con {aumento}: {nom} (mae / copia promedio {completos[nom]:.4f})")
    return c


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = {k: np.load(TRAINING / f"CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{k[1]}/evo_celsius.npy") for k in ESC}
    disc = preparar(temps)
    confs = configuraciones()
    log(f"Inicio barrido rotaciones ({dev}): {len(confs)} configuraciones × {len(RONDAS)} rondas; cortes °C "
        + ", ".join(f"{x:.1f}" for x in disc["cortes"]))

    log("Parte 1: configuraciones con y sin rotaciones (semilla 42)")
    for c in confs:
        for entrena, prueba in RONDAS:
            correr("pasos", disc, c, entrena, prueba, temps, dev)

    mejor("sin")
    c = mejor("rot4")
    log("Parte 2: semillas")
    for s in SEMILLAS:
        for entrena, prueba in RONDAS:
            correr("semillas", disc, dict(c, semilla=s), entrena, prueba, temps, dev)

    log("Parte 3: red final con p1 y p3")
    correr("final", disc, c, ("p1", "p3"), None, temps, dev)
    log("Fin")


if __name__ == "__main__":
    main()
