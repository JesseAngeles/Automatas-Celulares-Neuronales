"""Evalúa una alumna contra el proceso ideal (maestra + dado + localidad, con su empujón CRECER).

- un paso (cajas de prueba): desde imágenes reales (dados al azar) y desde la simulación ideal: acierto del
  visible, del dado y del visible donde cambia.
- libre con argmax como la GUI desde la primera imagen con dados al azar: métricas del visible (puntaje, área,
  frontera, movimiento interno, brotes) y salud del dado (errores de la regla, frecuencias), en las cajas de
  calibración (dish0/dish1, entrenamiento; para elegir) y en las de prueba (dish5, no vistas; para reportar).
→ redes/<nombre>/evaluacion.json, libre_<caja>.npy (prueba, 1.ª repetición); ideal_<crecer>.json una vez.

    python evaluar.py <nombre> [crecer=0]
"""
import json
import sys

import numpy as np
import torch

import comun as C

REPETICIONES = 2


@torch.no_grad()
def libre_alumna(red, E0, T):
    E = E0
    ev = [E]
    for _ in range(1, T):
        E = red.logits(E).argmax(-1).to(torch.int8)
        ev.append(E)
    return torch.stack(ev, 1)


def metricas(S, L):
    m = C.resumen(S, (L // C.DADOS).to(torch.int8))
    return {k: float(np.mean(v)) for k, v in m.items()}


def promedio(d):
    claves = [k for k in next(iter(d.values())) if k != "dados_fin"]
    return {k: float(np.mean([v[k] for v in d.values()])) for k in claves}


@torch.no_grad()
def libre(red, cajas, g, dev, reps, guardar=None):
    out = {}
    for caja in cajas:
        S = torch.from_numpy(C.cargar(caja)).to(dev)
        L = libre_alumna(red, C.combinar(S[:1].repeat(reps, 1, 1), g, dev), len(S))
        D = L % C.DADOS
        frec = torch.bincount(D[:, -1].flatten().long(), minlength=4).float()
        out[caja] = {**metricas(S, L), "error_regla_dado": (D[:, 1:] != C.paso_dado(D[:, :-1])).float().mean().item(),
                     "dados_fin": (frec / frec.sum()).cpu().numpy().round(3).tolist()}
        if guardar:
            np.save(guardar / f"libre_{caja}.npy", L[0].cpu().numpy())
    return out


@torch.no_grad()
def evaluar(nombre, crecer=0.0):
    C.CRECER = crecer
    dev = torch.device("cuda")
    maestra = C.Red(C.MAESTRA, dev)
    w = C.pesos_ruido(dev)
    carpeta = C.CARPETA / "redes" / nombre
    red = C.Red(carpeta / "red.npz", dev)
    g = torch.Generator(device=dev).manual_seed(7)
    ideal_json = C.CARPETA / f"ideal_{crecer:g}.json"
    ideal = {}
    tot = np.zeros(5)
    for caja in C.PRUEBA:                                                   # un paso
        S = torch.from_numpy(C.cargar(caja)).to(dev)
        I = C.libre_ideal(maestra, C.combinar(S[:1].repeat(REPETICIONES, 1, 1), g, dev), len(S), w)
        if not ideal_json.exists():
            ideal[caja] = metricas(S, I)
        d = S[0] != 0
        for X in (C.combinar(S[:-1], g, dev), I[0, :-1]):
            for s in range(0, len(X), 16):
                x = X[s:s + 16]
                y, p = C.paso_ideal(maestra, x, w), red.logits(x).argmax(-1).to(torch.int8)
                vy, vp, vx = y // C.DADOS, p // C.DADOS, x // C.DADOS
                cambia = (vy != vx) & d
                tot += [d.sum().item() * len(x), ((vy == vp) & d).sum().item(), ((y % 4 == p % 4) & d).sum().item(),
                        cambia.sum().item(), ((vy == vp) & cambia).sum().item()]
    if ideal:
        ideal["promedio"] = promedio(ideal)
        C.guardar_json(ideal_json, ideal)
    res = {"crecer": crecer,
           "un_paso": {"visible": tot[1] / tot[0], "dado": tot[2] / tot[0], "visible_donde_cambia": tot[4] / tot[3]},
           "cal": libre(red, C.CALIBRACION, g, dev, 1),
           "libre": libre(red, C.PRUEBA, g, dev, REPETICIONES, carpeta)}
    res["promedio_cal"] = promedio(res["cal"])
    res["promedio_libre"] = promedio(res["libre"])
    res["ideal"] = json.loads(ideal_json.read_text())["promedio"]
    C.guardar_json(carpeta / "evaluacion.json", res)
    print(nombre, "un paso", {k: round(v, 4) for k, v in res["un_paso"].items()})
    print("  cal   ", {k: round(v, 4) for k, v in res["promedio_cal"].items()})
    print("  prueba", {k: round(v, 4) for k, v in res["promedio_libre"].items()})
    print("  ideal ", {k: round(v, 4) for k, v in res["ideal"].items()})
    return res


if __name__ == "__main__":
    evaluar(sys.argv[1], *(float(x) for x in sys.argv[2:3]))
