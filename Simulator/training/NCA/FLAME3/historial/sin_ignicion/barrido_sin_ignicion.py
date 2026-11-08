"""NCA para FLAME 3 sin la ignición: se repiten ../cruzado_pasos y ../k6 a partir de que el fuego ya existe.

La ignición la hace una persona trazando líneas de fuego, así que no es dinámica que la NCA deba aprender.
Datos: CA/FLAME3/seleccion1/sin_ignicion/flame_plot_N (quitar_ignicion en notebooks/FLAME3.ipynb), desde el frame
    p1 = 133 (la línea del borde izquierdo se traza en 123–132)
    p2 = 171 (sin fuego hasta 167; la línea de la esquina aparece en 167–170)
    p3 = 19  (la línea superior se traza en 14–18)
Misma discretización log15 que antes (escala de los escenarios completos, 22.3–604.6 °C).

Partes (cada una con su CSV en esta carpeta; registro común en registro.log):
    pasos.csv       validación cruzada ({p1,p2}→p3, {p1,p3}→p2, {p2,p3}→p1) con pasos de 1, 3 y 6 frames
                    (5, 15, 30 s), entrada cruda / one-hot, 2 capas de 32 / 3 capas de 128, dos vecindades por paso.
    referencia.csv  las redes de ../k6/semillas (entrenadas con ignición, 9×9 cruda 2 capas de 32, paso 6,
                    semilla 42) evaluadas en los datos sin ignición.
    semillas.csv    la mejor configuración de pasos.csv con semillas 42, 1, 2, 3, 4.
    final.csv       la mejor configuración entrenada con los tres escenarios.
La mejor configuración es la de menor promedio (en las 3 rondas) de mae_c_prueba / copia_mae_c_prueba.

Métricas: las de ../k6/barrido_k6.py (exact, exact_cambia, mae_c en °C, encendido, rollout20_mae_c y copia_*),
con sufijos _ent1 / _ent2 / _prueba. Un paso = paso_frames frames; rollout20 = 20 pasos libres.

No modifica el simulador: llama a scripts/main.py y reutiliza ../cruzado/barrido_cruzado.py y ../k6/barrido_k6.py.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/sin_ignicion/barrido_sin_ignicion.py
"""
import csv
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
bc.CARPETA = k6.CARPETA = CARPETA            # registro.log y rutas de esta carpeta
log, entrenar = bc.log, bc.entrenar

ESC = k6.ESC
RONDAS = k6.RONDAS
INICIO = {"p1": 133, "p2": 171, "p3": 19}
PASOS = {1: (1, 3), 3: (3, 5), 6: (4, 6)}     # paso en frames → radios probados
BASE = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=200, lote=8192, frontera="abierta", semilla=42)
SEMILLAS = [42, 1, 2, 3, 4]


def configuraciones() -> list:
    confs = []
    for paso, radios in PASOS.items():
        for entrada in ("cruda", "onehot"):
            for radio in radios:
                for capas, neuronas in ((2, 32), (3, 128)):
                    confs.append(dict(BASE, paso=paso, entrada=entrada, radio=radio, capas=capas, neuronas=neuronas))
    return confs


def nombre(c: dict) -> str:
    return f"k{c['paso']}_r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}_{c['schedule']}_s{c['semilla']}"


def preparar(temps: dict, disc: dict) -> dict:
    """Secuencias con desfase para cada paso: {paso: {escenario: [rutas]}}."""
    rutas = {}
    for paso in PASOS:
        carpeta = CARPETA / "datos" / f"paso{paso}"
        carpeta.mkdir(parents=True, exist_ok=True)
        rutas[paso] = {}
        for k, t in temps.items():
            estados = np.digitize(t, disc["cortes"]).astype(np.int32)
            rutas[paso][k] = []
            for o in range(paso):
                destino = carpeta / f"{k}_o{o}.npy"
                np.save(destino, estados[o::paso])
                rutas[paso][k].append(destino)
    return rutas


def hecho(destino: Path, ronda: str, nom: str) -> bool:
    return destino.exists() and any(r["ronda"] == ronda and r["nombre"] == nom for r in csv.DictReader(open(destino)))


def escribir(destino: Path, fila: dict):
    nuevo = not destino.exists()
    with open(destino, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def resumen(fila: dict, escenarios: dict, t0: float) -> str:
    e = "prueba" if "prueba" in escenarios else "ent1"
    return (f"    {escenarios[e]}: exact {fila[f'exact_{e}']:.4f} (copia {fila[f'copia_exact_{e}']:.4f}), "
            f"mae {fila[f'mae_c_{e}']:.2f} °C (copia {fila[f'copia_mae_c_{e}']:.2f}), encendido "
            f"{fila[f'encendido_{e}']:.3f}, rollout20 {fila[f'rollout20_mae_c_{e}']:.2f} °C "
            f"(copia {fila[f'copia20_mae_c_{e}']:.2f})  [{time.time() - t0:.0f} s]")


def correr(parte: str, c: dict, entrena: tuple, prueba, rutas: dict, disc: dict, temps: dict, dev):
    destino = CARPETA / f"{parte}.csv"
    ronda = f"{'+'.join(entrena)}_a_{prueba}" if prueba else "todos"
    nom = nombre(c)
    if hecho(destino, ronda, nom):
        log(f"{parte} {ronda} {nom}: ya está en {destino.name}, se omite")
        return
    carpeta = CARPETA / parte / ronda
    carpeta.mkdir(parents=True, exist_ok=True)
    base = carpeta / nom
    log(f"{parte} {ronda} {nom}: entrenando")
    t0 = time.time()
    fila = {"parte": parte, "ronda": ronda, "entrena": "+".join(entrena), "prueba": prueba or "", "nombre": nom,
            "paso_frames": c["paso"], "segundos": 5 * c["paso"], **{k: v for k, v in c.items() if k != "paso"},
            "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
    fila.update(entrenar(c, [r for k in entrena for r in rutas[c["paso"]][k]], base))
    log(f"    entrenada: {fila['epocas_hechas']} épocas, loss {fila['loss_final']:.4f}, {fila['tiempo_s']} s; evaluando")
    escenarios = {f"ent{i + 1}": k for i, k in enumerate(entrena)}
    if prueba:
        escenarios["prueba"] = prueba
    k6.PASO = c["paso"]
    fila.update(k6.evaluar(c, base, disc, temps, escenarios, dev))
    escribir(destino, fila)
    log(resumen(fila, escenarios, t0))


def referencia(disc: dict, temps: dict, dev):
    """Redes de ../k6 (entrenadas con ignición) evaluadas en los datos sin ignición."""
    destino = CARPETA / "referencia.csv"
    c = dict(BASE, paso=6, entrada="cruda", radio=4, capas=2, neuronas=32)
    for entrena, prueba in RONDAS:
        ronda = f"{'+'.join(entrena)}_a_{prueba}"
        nom = "k6_con_ignicion_" + nombre(c)
        if hecho(destino, ronda, nom):
            continue
        t0 = time.time()
        base = CARPETA.parent / "k6" / "semillas" / ronda / "log15_r4_cruda_2x32_exp_s42"
        log(f"referencia {ronda}: {base.relative_to(TRAINING)}.npz")
        escenarios = {"ent1": entrena[0], "ent2": entrena[1], "prueba": prueba}
        k6.PASO = 6
        fila = {"parte": "referencia", "ronda": ronda, "entrena": "+".join(entrena), "prueba": prueba, "nombre": nom,
                "paso_frames": 6, "segundos": 30, **{k: v for k, v in c.items() if k != "paso"}, "vecindad": "9x9",
                **k6.evaluar(c, base, disc, temps, escenarios, dev)}
        escribir(destino, fila)
        log(resumen(fila, escenarios, t0))


def mejor() -> dict:
    filas = list(csv.DictReader(open(CARPETA / "pasos.csv")))
    por_conf = {}
    for f in filas:
        por_conf.setdefault(f["nombre"], []).append(float(f["mae_c_prueba"]) / float(f["copia_mae_c_prueba"]))
    completos = {k: np.mean(v) for k, v in por_conf.items() if len(v) == len(RONDAS)}
    nom = min(completos, key=completos.get)
    f = next(f for f in filas if f["nombre"] == nom)
    c = dict(BASE, paso=int(f["paso_frames"]), entrada=f["entrada"], radio=int(f["radio"]),
             capas=int(f["capas"]), neuronas=int(f["neuronas"]))
    log(f"Mejor configuración: {nom} (mae / copia promedio {completos[nom]:.4f})")
    return c


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completos = k6.celsius()
    disc = dict(zip(("cortes", "rep"), k6.discretizacion("log15", completos)))
    temps = {k: np.load(TRAINING / f"CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{k[1]}/evo_celsius.npy") for k in ESC}
    for k, t in temps.items():
        assert np.array_equal(t, completos[k][INICIO[k]:]), f"{k}: no coincide con el original desde {INICIO[k]}"
        original = np.load(TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo.npy")[INICIO[k]:]
        assert np.array_equal(np.digitize(t, disc["cortes"]), original), f"{k}: log15 no reproduce evo.npy"
    rutas = preparar(temps, disc)
    confs = configuraciones()
    log(f"Inicio barrido sin ignición ({dev}): frames " + ", ".join(f"{k} {len(t)} (desde {INICIO[k]})"
                                                                   for k, t in temps.items())
        + f"; {len(confs)} configuraciones × {len(RONDAS)} rondas, referencia, {len(SEMILLAS)} semillas, red final")

    log("Parte 1: pasos y configuraciones (semilla 42)")
    for c in confs:
        for entrena, prueba in RONDAS:
            correr("pasos", c, entrena, prueba, rutas, disc, temps, dev)

    log("Parte 2: redes de k6 (con ignición) evaluadas sin ignición")
    referencia(disc, temps, dev)

    c = mejor()
    log("Parte 3: semillas")
    for s in SEMILLAS:
        for entrena, prueba in RONDAS:
            correr("semillas", dict(c, semilla=s), entrena, prueba, rutas, disc, temps, dev)

    log("Parte 4: red final con los tres escenarios")
    correr("final", c, ("p1", "p2", "p3"), None, rutas, disc, temps, dev)
    log("Fin")


if __name__ == "__main__":
    main()
