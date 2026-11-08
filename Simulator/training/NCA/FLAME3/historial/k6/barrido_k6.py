"""NCA para FLAME 3 con pasos de 6 frames (30 s): semillas, discretizaciones y red final.

Configuración base (la mejor de ../cruzado_pasos): vecindad 9×9, entrada cruda, 2 capas de 32 ReLU,
salida de una neurona por estado (cross-entropy), exp 1e-3 → 1e-5, 200 épocas, lote 8192, frontera abierta.

Partes (cada una con su CSV en esta carpeta; registro común en registro.log):
    semillas.csv  log15 con semillas 42, 1, 2, 3, 4 en las 3 rondas de validación cruzada
                  ({p1,p2}→p3, {p1,p3}→p2, {p2,p3}→p1).
    clases.csv    otras discretizaciones de la temperatura (semilla 42, 3 rondas):
                  logN   N estados con cortes iguales en log(°C); representante = centro geométrico del intervalo
                  kmN    N estados por K-Means 1D sobre °C de los tres escenarios; representante = centroide
                  Todas usan la misma escala para los tres escenarios (mínimo y máximo de los tres).
    final.csv     red base entrenada con los tres escenarios (log15, semilla 42), evaluada en los tres.

Datos: CA/FLAME3/seleccion1/flame_plot_N/evo_celsius.npy (°C 100×100) → estados con cada discretización →
secuencias con desfase o = 0…5 (frames o, o+6, …) en datos/<disc>/pN_o<o>.npy. log15 reproduce
exactamente CA/FLAME3/seleccion1/flame_plot_N/evo.npy.

Métricas (un paso = 6 frames, sobre todas las secuencias del escenario):
    exact, exact_cambia                      en estados (solo comparables con la misma discretización)
    mae_c                                    error absoluto medio en °C: representante del estado predicho vs °C real
    encendido                                celdas con °C real < 83 en t y ≥ 130 en t+1 que la red predice ≥ 130 °C
    rollout20_mae_c                          °C tras 20 pasos libres desde el primer frame (desfase 0)
    copia_*                                  lo mismo para "el siguiente paso es igual" (estado actual → representante)
    cuantizacion_mae_c                       error de solo discretizar: representante del estado real vs °C real
Sufijos: _ent1 / _ent2 (escenarios de entrenamiento, en el orden de "entrena"), _prueba (el no visto).

No modifica el simulador: llama a scripts/main.py y reutiliza funciones de scripts/barrido_flame.py y
../cruzado/barrido_cruzado.py.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/k6/barrido_k6.py
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
sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(CARPETA.parent / "cruzado"))
from main import extract_patches                                            # noqa: E402
from barrido_flame import cargar_red                                        # noqa: E402
import barrido_cruzado as bc                                                 # noqa: E402

bc.CARPETA = CARPETA
log, entrenar = bc.log, bc.entrenar

PASO = 6
PASOS_ROLLOUT = 20
FRIO_C, FUEGO_C = 83.0, 130.0
ESC = ["p1", "p2", "p3"]
RONDAS = bc.RONDAS
BASE = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=200, lote=8192, frontera="abierta",
            entrada="cruda", radio=4, capas=2, neuronas=32, semilla=42)
DISCRETIZACIONES = ["log5", "log10", "log15", "log20", "log30", "km5", "km10", "km15", "km20"]
SEMILLAS = [42, 1, 2, 3, 4]


# ── Discretizaciones ─────────────────────────────────────────────────────────

def celsius() -> dict:
    return {k: np.load(TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo_celsius.npy") for k in ESC}


def discretizacion(nombre: str, temps: dict) -> tuple:
    """(cortes, representantes) en °C, comunes a los tres escenarios."""
    lo = min(float(t.min()) for t in temps.values())
    hi = max(float(t.max()) for t in temps.values())
    n = int(nombre.lstrip("logkm"))
    if nombre.startswith("log"):
        bordes = np.geomspace(lo, hi, n + 1)
        return bordes[1:-1], np.sqrt(bordes[:-1] * bordes[1:])
    from sklearn.cluster import KMeans
    todo = np.concatenate([t.ravel() for t in temps.values()])
    muestra = np.random.default_rng(0).choice(todo, size=400_000, replace=False).reshape(-1, 1)
    centros = np.sort(KMeans(n_clusters=n, n_init=10, random_state=0).fit(muestra).cluster_centers_.ravel())
    return (centros[:-1] + centros[1:]) / 2, centros


def preparar(nombre: str, temps: dict) -> dict:
    """Guarda las secuencias de la discretización y regresa {escenario: [rutas]} y metadatos."""
    carpeta = CARPETA / "datos" / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    cortes, rep = discretizacion(nombre, temps)
    with open(carpeta / "discretizacion.json", "w") as f:
        json.dump({"nombre": nombre, "cortes_c": cortes.tolist(), "representantes_c": rep.tolist(),
                   "paso_frames": PASO}, f, indent=2)
    rutas = {}
    for k, t in temps.items():
        estados = np.digitize(t, cortes).astype(np.int32)
        rutas[k] = []
        for o in range(PASO):
            destino = carpeta / f"{k}_o{o}.npy"
            np.save(destino, estados[o::PASO])
            rutas[k].append(destino)
    return {"rutas": rutas, "cortes": cortes, "rep": rep}


# ── Evaluación ───────────────────────────────────────────────────────────────

def predecir(red, X: np.ndarray, dev, lote: int = 1 << 16) -> np.ndarray:
    salidas = []
    with torch.no_grad():
        for s in range(0, len(X), lote):
            salidas.append(red(torch.from_numpy(X[s:s + lote]).to(dev)).argmax(1).cpu())
    return torch.cat(salidas).numpy()


def evaluar(c: dict, base: Path, disc: dict, temps: dict, escenarios: dict, dev) -> dict:
    """escenarios: {etiqueta: clave del escenario}, p. ej. {"ent1": "p1", "ent2": "p2", "prueba": "p3"}."""
    red, nb = cargar_red(f"{base}.npz", dev)
    abierta = c["frontera"] == "abierta"
    centro = int(np.where((nb == 0).all(1))[0][0])
    cortes, rep = disc["cortes"], disc["rep"]
    fila = {}
    for etiqueta, clave in escenarios.items():
        pred, actual, real, real_t = [], [], [], []
        for o in range(PASO):
            t = temps[clave][o::PASO]
            est = np.digitize(t, cortes)
            X, _ = extract_patches(est.astype(np.float32), nb, abierta)
            pred.append(predecir(red, X, dev))
            actual.append(X[:, centro].astype(np.int64))
            real.append(t[1:].ravel())
            real_t.append(t[:-1].ravel())
        pred, actual = np.concatenate(pred), np.concatenate(actual)
        real, real_t = np.concatenate(real), np.concatenate(real_t)
        sig = np.digitize(real, cortes)
        cambia = sig != actual
        enciende = (real_t < FRIO_C) & (real >= FUEGO_C)
        fila.update({
            f"exact_{etiqueta}": round(float((pred == sig).mean()), 5),
            f"exact_cambia_{etiqueta}": round(float((pred[cambia] == sig[cambia]).mean()), 5),
            f"mae_c_{etiqueta}": round(float(np.abs(rep[pred] - real).mean()), 4),
            f"encendido_{etiqueta}": round(float((rep[pred][enciende] >= FUEGO_C).mean()), 5),
            f"copia_exact_{etiqueta}": round(float((actual == sig).mean()), 5),
            f"copia_mae_c_{etiqueta}": round(float(np.abs(rep[actual] - real).mean()), 4),
            f"cuantizacion_mae_c_{etiqueta}": round(float(np.abs(rep[sig] - real).mean()), 4),
        })
        # 20 pasos libres desde el primer frame de la secuencia con desfase 0
        t0 = temps[clave][0::PASO]
        estado = np.digitize(t0[0], cortes).astype(np.float32)
        for _ in range(PASOS_ROLLOUT):
            X, _ = extract_patches(np.stack([estado, estado]), nb, abierta)
            estado = predecir(red, X, dev).reshape(estado.shape).astype(np.float32)
        fila[f"rollout20_mae_c_{etiqueta}"] = round(float(np.abs(rep[estado.astype(int)] - t0[PASOS_ROLLOUT]).mean()), 3)
        fila[f"copia20_mae_c_{etiqueta}"] = round(float(np.abs(rep[np.digitize(t0[0], cortes)] - t0[PASOS_ROLLOUT]).mean()), 3)
    del red
    torch.cuda.empty_cache()
    return fila


# ── Corridas ─────────────────────────────────────────────────────────────────

def correr(parte: str, disc_nombre: str, disc: dict, c: dict, entrena: tuple, prueba, temps: dict, dev):
    destino = CARPETA / f"{parte}.csv"
    ronda = f"{'+'.join(entrena)}_a_{prueba}" if prueba else "todos"
    nom = f"{disc_nombre}_r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}_{c['schedule']}_s{c['semilla']}"
    if destino.exists() and any(r["ronda"] == ronda and r["nombre"] == nom for r in csv.DictReader(open(destino))):
        log(f"{parte} {ronda} {nom}: ya está en {destino.name}, se omite")
        return
    carpeta = CARPETA / parte / ronda
    carpeta.mkdir(parents=True, exist_ok=True)
    base = carpeta / nom
    log(f"{parte} {ronda} {nom}: entrenando")
    t0 = time.time()
    fila = {"parte": parte, "discretizacion": disc_nombre, "estados": len(disc["rep"]), "ronda": ronda,
            "entrena": "+".join(entrena), "prueba": prueba or "", "nombre": nom, "paso_frames": PASO,
            **c, "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
    bc.ESTADOS = len(disc["rep"])      # --out-size de main.py: una neurona por estado
    fila.update(entrenar(c, [r for k in entrena for r in disc["rutas"][k]], base))
    log(f"    entrenada: {fila['epocas_hechas']} épocas, loss {fila['loss_final']:.4f}, {fila['tiempo_s']} s; evaluando")
    escenarios = {f"ent{i + 1}": k for i, k in enumerate(entrena)}
    if prueba:
        escenarios["prueba"] = prueba
    fila.update(evaluar(c, base, disc, temps, escenarios, dev))
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


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = celsius()
    log(f"Inicio barrido k6 ({dev}): semillas {SEMILLAS}, discretizaciones {DISCRETIZACIONES}, red final")
    discs = {}
    for nombre in DISCRETIZACIONES:
        discs[nombre] = preparar(nombre, temps)
        log(f"Discretización {nombre}: cortes °C " + ", ".join(f"{x:.1f}" for x in discs[nombre]["cortes"]))

    log("Parte 1: semillas (log15)")
    for s in SEMILLAS:
        for entrena, prueba in RONDAS:
            correr("semillas", "log15", discs["log15"], dict(BASE, semilla=s), entrena, prueba, temps, dev)

    log("Parte 2: discretizaciones (semilla 42)")
    for nombre in DISCRETIZACIONES:
        for entrena, prueba in RONDAS:
            correr("clases", nombre, discs[nombre], BASE, entrena, prueba, temps, dev)

    log("Parte 3: red final con los tres escenarios (log15, semilla 42)")
    correr("final", "log15", discs["log15"], BASE, ("p1", "p2", "p3"), None, temps, dev)
    log("Fin")


if __name__ == "__main__":
    main()
