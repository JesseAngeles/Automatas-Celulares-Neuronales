"""NCA para FLAME 3 con memoria explícita: bandera de "ya se quemó" en el estado, y los tres primeros estados
de log15 fusionados en uno.

Temperatura: log15 de ../sin_p2 (escala de p1 y p3 sin ignición, 24.7–604.6 °C) con los estados 0, 1 y 2 unidos
(24.7–46.8 °C, representante = centro geométrico) → 13 estados de temperatura T = 0…12.
Bandera Q = 1 si la celda llegó a ≥ 135.9 °C (inicio del estado "fuego") en algún frame anterior o en el actual.
Se calcula con la historia completa (CA/FLAME3/seleccion1/flame_plot_N, con ignición) y después se recorta igual que
CA/FLAME3/seleccion1/sin_ignicion (p1 desde el frame 133, p3 desde el 19), así que lo que se quemó durante la ignición ya
empieza marcado.

Variantes:
    t13    estado = T                 (13 estados; referencia para separar el efecto de la bandera)
    t13q   estado = T + 13 · Q        (26 estados; los de Q = 0 con T ≥ 6 no existen)

Validación cruzada p1 → p3 y p3 → p1, sin rotaciones. Configuraciones (2 capas de 32 ReLU, 200 épocas,
exp 1e-3 → 1e-5, lote 8192, frontera abierta): variantes × pasos de 3 y 6 frames (dos vecindades por paso)
× entrada cruda / one-hot.

Partes (cada una con su CSV en esta carpeta; registro común en registro.log):
    pasos.csv      todas las configuraciones (semilla 42)
    semillas.csv   la mejor de t13q con semillas 42, 1, 2, 3, 4, y la misma configuración en t13
    final.csv      la mejor de t13q entrenada con p1 y p3
La mejor es la de menor promedio (en las 2 rondas) de mae_c_prueba / copia_mae_c_prueba.

Métricas (un paso = paso_frames frames, todas las secuencias con desfase; sufijos _ent1 y _prueba):
    exact              estado completo (T y Q)
    exact_temp         solo T
    exact_cambia       estado completo, en las celdas cuyo estado cambia
    exact_bandera      solo Q (solo t13q)
    mae_c              °C: representante de T predicho vs °C real
    encendido          de las celdas < 83 °C en t que llegan a ≥ 130 °C en t+1, fracción que la red predice ≥ 130 °C
    precision_encendido  de las celdas < 83 °C en t que la red predice ≥ 130 °C, fracción que de verdad llega
    cuantizacion_mae_c error de solo discretizar
    rollout20_mae_c    °C tras 20 pasos libres desde el primer frame (desfase 0)
    copia_*            lo mismo para "el siguiente paso es igual"

No modifica el simulador: llama a scripts/main.py y reutiliza ../cruzado/barrido_cruzado.py y ../k6/barrido_k6.py.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/bandera/barrido_bandera.py
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
extract_patches, cargar_red, predecir = k6.extract_patches, k6.cargar_red, k6.predecir

ESC = ["p1", "p3"]
INICIO = {"p1": 133, "p3": 19}
RONDAS = [(("p1",), "p3"), (("p3",), "p1")]
FUSION = 3                                    # estados de log15 que se unen en el primero
NT = 15 - FUSION + 1                          # estados de temperatura
VARIANTES = {f"t{NT}": 1, f"t{NT}q": 2}       # variante → valores de la bandera
PASOS = {3: (3, 5), 6: (4, 6)}
BASE = dict(lr=1e-3, schedule="exp", lr_min=1e-5, epocas=200, lote=8192, frontera="abierta",
            capas=2, neuronas=32, semilla=42)
SEMILLAS = [42, 1, 2, 3, 4]
FRIO_C, FUEGO_C = k6.FRIO_C, k6.FUEGO_C      # 83 y 130 °C, como en k6
PASOS_ROLLOUT = 20


def nombre(c: dict) -> str:
    return (f"{c['variante']}_k{c['paso']}_r{c['radio']}_{c['entrada']}_{c['capas']}x{c['neuronas']}"
            f"_{c['schedule']}_s{c['semilla']}")


# ── Datos ────────────────────────────────────────────────────────────────────

def preparar() -> dict:
    temps, quemado = {}, {}
    for k in ESC:
        completo = np.load(TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo_celsius.npy")
        recortado = np.load(TRAINING / f"CA/FLAME3/seleccion1/sin_ignicion/flame_plot_{k[1]}/evo_celsius.npy")
        assert np.array_equal(completo[INICIO[k]:], recortado), f"{k}: sin_ignicion no coincide con el original"
        temps[k] = recortado
        quemado[k] = completo  # se reemplaza abajo, cuando se conoce el umbral
    cortes15, rep15 = k6.discretizacion("log15", temps)
    lo = min(float(t.min()) for t in temps.values())
    cortes = cortes15[FUSION - 1:]                             # sin los cortes entre los estados que se unen
    rep = np.concatenate([[np.sqrt(lo * cortes15[FUSION - 1])], rep15[FUSION:]])
    umbral = float(cortes15[7])                                # inicio del estado 8 de log15 (135.9 °C)
    for k in ESC:
        quemado[k] = np.maximum.accumulate(quemado[k] >= umbral, axis=0)[INICIO[k]:].astype(np.int32)
    carpeta = CARPETA / "datos"
    carpeta.mkdir(parents=True, exist_ok=True)
    with open(carpeta / "discretizacion.json", "w") as f:
        json.dump({"temperatura": f"log15 de sin_p2 con los estados 0…{FUSION - 1} unidos", "cortes_c": cortes.tolist(),
                   "representantes_c": rep.tolist(), "umbral_quemado_c": umbral,
                   "estado": f"T + {NT} · Q (t{NT}q) o T (t{NT})", "inicio": INICIO}, f, indent=2, ensure_ascii=False)
    estados, rutas = {}, {}
    for v, nq in VARIANTES.items():
        estados[v] = {k: (np.digitize(temps[k], cortes) + NT * (quemado[k] if nq == 2 else 0)).astype(np.int32)
                      for k in ESC}
        rutas[v] = {}
        for paso in PASOS:
            rutas[v][paso] = {}
            for k in ESC:
                rutas[v][paso][k] = []
                for o in range(paso):
                    destino = carpeta / f"{v}_{k}_k{paso}_o{o}.npy"
                    np.save(destino, estados[v][k][o::paso])
                    rutas[v][paso][k].append(destino)
    for k in ESC:
        log(f"{k}: {len(temps[k])} frames, quemado al inicio {quemado[k][0].mean():.1%}, al final "
            f"{quemado[k][-1].mean():.1%}; estados t{NT}q presentes {np.unique(estados[f't{NT}q'][k]).size} de {2 * NT}")
    return {"cortes": cortes, "rep": rep, "umbral": umbral, "temps": temps, "estados": estados, "rutas": rutas}


# ── Evaluación ───────────────────────────────────────────────────────────────

def evaluar(c: dict, base: Path, d: dict, escenarios: dict, dev) -> dict:
    red, nb = cargar_red(f"{base}.npz", dev)
    abierta = c["frontera"] == "abierta"
    centro = int(np.where((nb == 0).all(1))[0][0])
    rep, paso, v = d["rep"], c["paso"], c["variante"]
    fila = {}
    for etiqueta, clave in escenarios.items():
        pred, actual, sig, real, real_t = [], [], [], [], []
        for o in range(paso):
            est, t = d["estados"][v][clave][o::paso], d["temps"][clave][o::paso]
            X, _ = extract_patches(est.astype(np.float32), nb, abierta)
            pred.append(predecir(red, X, dev))
            actual.append(X[:, centro].astype(np.int64))
            sig.append(est[1:].ravel())
            real.append(t[1:].ravel())
            real_t.append(t[:-1].ravel())
        pred, actual, sig = np.concatenate(pred), np.concatenate(actual), np.concatenate(sig)
        real, real_t = np.concatenate(real), np.concatenate(real_t)
        cambia = sig != actual
        frio = real_t < FRIO_C
        enciende = frio & (real >= FUEGO_C)
        pred_fuego = frio & (rep[pred % NT] >= FUEGO_C)
        fila.update({
            f"exact_{etiqueta}": round(float((pred == sig).mean()), 5),
            f"exact_temp_{etiqueta}": round(float((pred % NT == sig % NT).mean()), 5),
            f"exact_cambia_{etiqueta}": round(float((pred[cambia] == sig[cambia]).mean()), 5),
            f"exact_bandera_{etiqueta}": round(float((pred // NT == sig // NT).mean()), 5) if v.endswith("q") else "",
            f"mae_c_{etiqueta}": round(float(np.abs(rep[pred % NT] - real).mean()), 4),
            f"encendido_{etiqueta}": round(float((rep[pred % NT][enciende] >= FUEGO_C).mean()), 5),
            f"predichos_encendido_{etiqueta}": int(pred_fuego.sum()),
            f"precision_encendido_{etiqueta}": round(float((real[pred_fuego] >= FUEGO_C).mean()), 5) if pred_fuego.any() else "",
            f"copia_exact_{etiqueta}": round(float((actual == sig).mean()), 5),
            f"copia_mae_c_{etiqueta}": round(float(np.abs(rep[actual % NT] - real).mean()), 4),
            f"cuantizacion_mae_c_{etiqueta}": round(float(np.abs(rep[sig % NT] - real).mean()), 4),
        })
        est, t = d["estados"][v][clave][0::paso], d["temps"][clave][0::paso]
        estado = est[0].astype(np.float32)
        for _ in range(PASOS_ROLLOUT):
            X, _ = extract_patches(np.stack([estado, estado]), nb, abierta)
            estado = predecir(red, X, dev).reshape(estado.shape).astype(np.float32)
        fila[f"rollout20_mae_c_{etiqueta}"] = round(float(np.abs(rep[estado.astype(int) % NT] - t[PASOS_ROLLOUT]).mean()), 3)
        fila[f"copia20_mae_c_{etiqueta}"] = round(float(np.abs(rep[est[0] % NT] - t[PASOS_ROLLOUT]).mean()), 3)
    del red
    torch.cuda.empty_cache()
    return fila


# ── Corridas ─────────────────────────────────────────────────────────────────

def correr(parte: str, c: dict, entrena: tuple, prueba, d: dict, dev):
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
    estados = NT * VARIANTES[c["variante"]]
    fila = {"parte": parte, "variante": c["variante"], "estados": estados, "ronda": ronda,
            "entrena": "+".join(entrena), "prueba": prueba or "", "nombre": nom, "paso_frames": c["paso"],
            "segundos": 5 * c["paso"], **{k: v for k, v in c.items() if k not in ("paso", "variante")},
            "vecindad": f"{2 * c['radio'] + 1}x{2 * c['radio'] + 1}"}
    bc.ESTADOS = estados                       # --out-size y --one-hot de main.py
    fila.update(entrenar(c, [r for k in entrena for r in d["rutas"][c["variante"]][c["paso"]][k]], base))
    log(f"    entrenada: {fila['epocas_hechas']} épocas, loss {fila['loss_final']:.4f}, {fila['tiempo_s']} s; evaluando")
    escenarios = {f"ent{i + 1}": k for i, k in enumerate(entrena)}
    if prueba:
        escenarios["prueba"] = prueba
    fila.update(evaluar(c, base, d, escenarios, dev))
    nuevo = not destino.exists()
    with open(destino, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)
    e = "prueba" if prueba else "ent1"
    log(f"    {escenarios[e]}: exact {fila[f'exact_{e}']:.4f} (copia {fila[f'copia_exact_{e}']:.4f}), "
        f"mae {fila[f'mae_c_{e}']:.2f} °C (copia {fila[f'copia_mae_c_{e}']:.2f}), encendido "
        f"{fila[f'encendido_{e}']:.3f} (precisión {fila[f'precision_encendido_{e}']}), rollout20 "
        f"{fila[f'rollout20_mae_c_{e}']:.2f} °C (copia {fila[f'copia20_mae_c_{e}']:.2f})  [{time.time() - t0:.0f} s]")


def mejor(variante: str) -> dict:
    filas = [f for f in csv.DictReader(open(CARPETA / "pasos.csv")) if f["variante"] == variante]
    por_conf = {}
    for f in filas:
        por_conf.setdefault(f["nombre"], []).append(float(f["mae_c_prueba"]) / float(f["copia_mae_c_prueba"]))
    completos = {k: np.mean(v) for k, v in por_conf.items() if len(v) == len(RONDAS)}
    nom = min(completos, key=completos.get)
    f = next(f for f in filas if f["nombre"] == nom)
    c = dict(BASE, variante=variante, paso=int(f["paso_frames"]), radio=int(f["radio"]), entrada=f["entrada"])
    log(f"Mejor con {variante}: {nom} (mae / copia promedio {completos[nom]:.4f})")
    return c


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Inicio barrido bandera ({dev})")
    d = preparar()
    log("Cortes °C: " + ", ".join(f"{x:.1f}" for x in d["cortes"]) + f"; umbral de quemado {d['umbral']:.1f} °C")

    log("Parte 1: variantes, pasos, vecindades y entrada (semilla 42)")
    for v in VARIANTES:
        for paso, radios in PASOS.items():
            for radio in radios:
                for entrada in ("cruda", "onehot"):
                    for entrena, prueba in RONDAS:
                        correr("pasos", dict(BASE, variante=v, paso=paso, radio=radio, entrada=entrada),
                               entrena, prueba, d, dev)

    mejor(f"t{NT}")
    c = mejor(f"t{NT}q")
    log("Parte 2: semillas (la mejor con bandera y la misma sin bandera)")
    for s in SEMILLAS:
        for v in VARIANTES:
            for entrena, prueba in RONDAS:
                correr("semillas", dict(c, variante=v, semilla=s), entrena, prueba, d, dev)

    log("Parte 3: red final con p1 y p3")
    correr("final", c, ("p1", "p3"), None, d, dev)
    log("Fin")


if __name__ == "__main__":
    main()
