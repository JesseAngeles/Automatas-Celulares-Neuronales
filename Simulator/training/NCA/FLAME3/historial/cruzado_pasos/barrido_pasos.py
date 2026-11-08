"""Validación cruzada de NCA para FLAME 3 con pasos de tiempo más largos.

Igual que ../cruzado/barrido_cruzado.py (entrenar con dos escenarios, evaluar en el tercero), pero cada
paso de la NCA son k frames (k = 3 → 15 s, k = 6 → 30 s). Para no perder transiciones, cada
escenario se divide en k secuencias con desfase o = 0 … k−1 (frames o, o+k, o+2k, …), guardadas en
datos/pN_k<k>_o<o>.npy, y se le pasan todas a scripts/main.py con varios --data.

No modifica el simulador: llama a scripts/main.py y reutiliza funciones de scripts/barrido_flame.py
y ../cruzado/barrido_cruzado.py. Escribe en esta carpeta:
    datos/                      secuencias submuestreadas
    resultados.csv              una fila por (paso, ronda, configuración)
    registro.log                qué se está haciendo, con hora
    k<k>/<ronda>/<nombre>.npz   pesos (+ _loss.csv)

Métricas de un paso (k frames) sobre todas las secuencias del escenario: loss, exact, exact_cambia,
encendido, mae con sufijo _ent1 / _ent2 / _prueba; rollout20_mae_prueba = error en número de estado
tras 20 pasos libres desde el primer frame (20·k frames); copia_* = "el siguiente paso es igual".

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/cruzado_pasos/barrido_pasos.py
"""
import csv
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
from barrido_flame import cargar_red, metricas                              # noqa: E402
import barrido_cruzado as bc                                                 # noqa: E402

bc.CARPETA = CARPETA            # registro.log de esta carpeta
log, entrenar = bc.log, bc.entrenar

ESC = {f"p{i}": TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{i}/evo.npy" for i in (1, 2, 3)}
RONDAS = bc.RONDAS
PASOS_ROLLOUT = 20
RADIOS = {3: (3, 5), 6: (4, 6)}
FIJO = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=200, lote=8192, frontera="abierta", semilla=42)


def configuraciones(k: int) -> list:
    return [dict(FIJO, paso=k, entrada=entrada, radio=radio, capas=capas, neuronas=neuronas)
            for entrada in ("cruda", "onehot")
            for radio in RADIOS[k]
            for capas, neuronas in ((2, 32), (3, 128))]


def nombre(c: dict) -> str:
    return f"k{c['paso']}_r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}_{c['schedule']}_s{c['semilla']}"


def predecir(red, X: np.ndarray, dev, lote: int = 1 << 16) -> torch.Tensor:
    """Como barrido_flame.predecir pero en lotes chicos: con one-hot y vecindades de 11×11 o más,
    un lote de 2^18 ocupa varios GB de GPU y deja sin memoria al siguiente entrenamiento."""
    salidas = []
    with torch.no_grad():
        for s in range(0, len(X), lote):
            salidas.append(red(torch.from_numpy(X[s:s + lote]).to(dev)))
    return torch.cat(salidas)


def secuencias(k: int) -> dict:
    """{escenario: [rutas de las k secuencias submuestreadas]}"""
    (CARPETA / "datos").mkdir(exist_ok=True)
    rutas = {}
    for clave, ruta in ESC.items():
        evo = np.load(ruta)
        rutas[clave] = []
        for o in range(k):
            destino = CARPETA / "datos" / f"{clave}_k{k}_o{o}.npy"
            if not destino.exists():
                np.save(destino, evo[o::k])
            rutas[clave].append(destino)
    return rutas


def evaluar(c: dict, base: Path, seqs: dict, prueba: str, dev) -> dict:
    red, nb = cargar_red(f"{base}.npz", dev)
    abierta = c["frontera"] == "abierta"
    centro = int(np.where((nb == 0).all(1))[0][0])
    entrenamiento = [k for k in seqs if k != prueba]
    fila = {}
    for clave, evos in seqs.items():
        partes = [extract_patches(e.astype(np.float32), nb, abierta) for e in evos]
        X = np.concatenate([p[0] for p in partes])
        Y = np.concatenate([p[1] for p in partes]).astype(np.int64)
        m = metricas(predecir(red, X, dev), X[:, centro].astype(np.int64), Y, dev)
        etiqueta = "prueba" if clave == prueba else f"ent{entrenamiento.index(clave) + 1}"
        for nom, v in m.items():
            fila[f"{nom}_{etiqueta}"] = round(v, 5)
        del X, Y, partes
    evo = seqs[prueba][0]
    estado = evo[0].astype(np.float32)
    for _ in range(PASOS_ROLLOUT):
        X, _ = extract_patches(np.stack([estado, estado]), nb, abierta)
        estado = predecir(red, X, dev).argmax(1).cpu().numpy().reshape(estado.shape).astype(np.float32)
    fila["rollout20_mae_prueba"] = round(float(np.abs(estado - evo[PASOS_ROLLOUT]).mean()), 4)
    del red
    torch.cuda.empty_cache()   # libera la GPU para el siguiente entrenamiento (main.py es otro proceso)
    return fila


def copia(evos: list) -> dict:
    x = np.concatenate([e[:-1].ravel() for e in evos]).astype(np.int64)
    y = np.concatenate([e[1:].ravel() for e in evos]).astype(np.int64)
    e0 = evos[0]
    return {"copia_exact_prueba": round(float((x == y).mean()), 5),
            "copia_mae_prueba": round(float(np.abs(x - y).mean()), 5),
            "copia20_mae_prueba": round(float(np.abs(e0[0] - e0[PASOS_ROLLOUT]).astype(float).mean()), 4)}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    destino = CARPETA / "resultados.csv"
    hechos = ({(r["paso"], r["ronda"], r["nombre"]) for r in csv.DictReader(open(destino))}
              if destino.exists() else set())
    total = sum(len(configuraciones(k)) for k in RADIOS) * len(RONDAS)
    log(f"Inicio: pasos {list(RADIOS)} × {len(RONDAS)} rondas = {total} entrenamientos ({dev})")

    i = 0
    for k in RADIOS:
        rutas = secuencias(k)
        seqs = {clave: [np.load(r) for r in rs] for clave, rs in rutas.items()}
        log(f"Paso k = {k} ({5 * k} s): secuencias de "
            + ", ".join(f"{c} {[len(s) for s in seqs[c]]} frames" for c in seqs))
        for entrenamiento, prueba in RONDAS:
            ronda = f"{'+'.join(entrenamiento)}_a_{prueba}"
            (CARPETA / f"k{k}" / ronda).mkdir(parents=True, exist_ok=True)
            log(f"Ronda {ronda} (k = {k}): entrena con {', '.join(entrenamiento)}, evalúa en {prueba}")
            for c in configuraciones(k):
                i += 1
                nom = nombre(c)
                if (str(k), ronda, nom) in hechos:
                    log(f"[{i}/{total}] {nom} {ronda}: ya está en resultados.csv, se omite")
                    continue
                log(f"[{i}/{total}] {nom} {ronda}: entrenando")
                base = CARPETA / f"k{k}" / ronda / nom
                t0 = time.time()
                fila = {"paso": k, "segundos": 5 * k, "ronda": ronda, "entrena": "+".join(entrenamiento),
                        "prueba": prueba, "nombre": nom, **{x: v for x, v in c.items() if x != "paso"},
                        "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
                datos = [r for clave in entrenamiento for r in rutas[clave]]
                fila.update(entrenar(c, datos, base))
                log(f"    entrenada: {fila['epocas_hechas']} épocas, loss {fila['loss_final']:.4f}, "
                    f"{fila['tiempo_s']} s; evaluando")
                fila.update(evaluar(c, base, {x: seqs[x] for x in entrenamiento + (prueba,)}, prueba, dev))
                fila.update(copia(seqs[prueba]))
                nuevo = not destino.exists()
                with open(destino, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(fila))
                    if nuevo:
                        w.writeheader()
                    w.writerow(fila)
                log(f"    {prueba} (no vista): loss {fila['loss_prueba']:.4f}, exact {fila['exact_prueba']:.4f} "
                    f"(copia {fila['copia_exact_prueba']:.4f}), exact_cambia {fila['exact_cambia_prueba']:.3f}, "
                    f"encendido {fila['encendido_prueba']:.3f}, rollout20 {fila['rollout20_mae_prueba']:.3f} "
                    f"(copia {fila['copia20_mae_prueba']:.3f})  [{time.time() - t0:.0f} s]")
    log("Fin")


if __name__ == "__main__":
    main()
