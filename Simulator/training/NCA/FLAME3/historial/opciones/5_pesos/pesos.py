"""Opción 5 — pérdida pesada: las celdas cuyo estado cambia entre t y t+1 pesan más en la entropía cruzada.

Peso de cada muestra w = 1 + α · [estado(t+1) ≠ estado(t)]; pérdida del lote = Σ w · CE / Σ w.
    α = 0    control: el mismo entrenamiento que scripts/main.py, con este entrenador
    α = 3, 10
Entrenador propio (no modifica scripts/main.py): el mismo modelo (main.build_model), Adam, exp 1e-3 → 1e-5 en 200
épocas, lote 8192, semillas como main.py, frontera abierta, pesos guardados con main.save_weights (se cargan igual
en la GUI). Resto igual que la referencia (../0_base): 9×9, paso 6, cruda, 2 capas de 32, p1 → p3 y p3 → p1,
semillas 42, 1, 2, 3, 4.

Escribe datos/, redes/<ronda>/<nombre>.npz (+ _loss.csv), resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/5_pesos/pesos.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun as cm                                                           # noqa: E402

ALFAS = [0, 3, 10]


def vecindad() -> np.ndarray:
    """La misma vecindad 9×9 (y en el mismo orden) que las redes de la referencia."""
    return np.load(cm.red_referencia("p1_a_p3", 42))["neighbourhood"]


def entrenar(rutas: list, alfa: float, c: dict, base: Path, dev, nb: np.ndarray = None) -> dict:
    """Con alfa = 0 da los mismos pesos que scripts/main.py (verificado con las 10 redes de la referencia)."""
    np.random.seed(c["semilla"])
    torch.manual_seed(c["semilla"])
    torch.cuda.manual_seed_all(c["semilla"])
    nb = vecindad() if nb is None else nb
    centro = int(np.where((nb == 0).all(1))[0][0])
    partes = [cm.extract_patches(np.load(r).astype(np.float32), nb, True) for r in rutas]
    X = torch.from_numpy(np.concatenate([p[0] for p in partes])).to(dev)
    Y = torch.from_numpy(np.concatenate([p[1] for p in partes])).long().to(dev)
    W = 1 + alfa * (Y != X[:, centro].long()).float()
    capas = [(c["neuronas"], "relu")] * c["capas"]
    modelo = cm.build_model(len(nb), capas, cm.NT, "linear").to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=c["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(c["lr_min"] / c["lr"]) ** (1 / (c["epocas"] - 1)))
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    N, t0 = len(X), time.perf_counter()
    with open(f"{base}_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "lr"])
        for epoca in range(c["epocas"]):
            perm = torch.randperm(N, device=dev)
            total, lotes = 0.0, 0
            for s in range(0, N, c["lote"]):
                i = perm[s:s + c["lote"]]
                perdida = (ce(modelo(X[i]), Y[i]) * W[i]).sum() / W[i].sum()
                opt.zero_grad()
                perdida.backward()
                opt.step()
                total += perdida.item()
                lotes += 1
            w.writerow([epoca, total / lotes, opt.param_groups[0]["lr"]])
            sched.step()
    cm.save_weights(f"{base}.npz", modelo, nb, capas, "linear", 0)
    return {"epocas_hechas": c["epocas"], "loss_final": round(total / lotes, 6),
            "fraccion_cambia": round(float((Y != X[:, centro].long()).float().mean()), 4),
            "tiempo_s": round(time.perf_counter() - t0, 1)}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = cm.temperaturas()
    cm.log(CARPETA, "Inicio opción 5 (pérdida pesada en las celdas que cambian)")
    rutas = cm.secuencias({k: cm.estados(t) for k, t in temps.items()}, cm.PASO, CARPETA / "datos", "log15")
    for alfa in ALFAS:
        for entrena, prueba in cm.RONDAS:
            ronda = cm.ronda_nombre(entrena, prueba)
            for s in cm.SEMILLAS:
                clave = {"opcion": "5_pesos", "variante": f"alfa{alfa}", "ronda": ronda, "semilla": s}
                if cm.hecho(CARPETA, clave):
                    continue
                (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                base = CARPETA / "redes" / ronda / f"alfa{alfa}_k6_r4_cruda_2x32_exp_s{s}"
                cm.log(CARPETA, f"alfa {alfa} {ronda} s{s}: entrenando")
                info = entrenar([r for k in entrena for r in rutas[k]], alfa, dict(cm.BASE, semilla=s), base, dev)
                prob, nb = cm.predictor([f"{base}.npz"], dev, cm.NT)
                fila = {**clave, "prueba": prueba, "paso_frames": cm.PASO, "decodificador": "argmax", "alfa": alfa,
                        **info, **cm.evaluar(prob, nb, temps, prueba, dev)}
                cm.escribir(CARPETA, fila)
                cm.log(CARPETA, f"    loss {info['loss_final']:.4f}, {info['tiempo_s']} s; {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
