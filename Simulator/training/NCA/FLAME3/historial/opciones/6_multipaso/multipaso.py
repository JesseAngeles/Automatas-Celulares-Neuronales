"""Opción 6 — entrenamiento con simulación de varios pasos (truco "pushforward").

Se parte de cada red de la referencia (../../sin_p2/semillas, 9×9, paso 6, cruda, 2 capas de 32) y se ajusta así:
desde un frame real t la red simula K − 1 pasos con sus propias predicciones (argmax, sin gradiente) y se le pide
acertar el frame real t + K a partir de ese estado simulado. Así aprende a corregir sus propios errores, que es lo
que se acumula en la simulación libre. La pérdida suma ese término y el de un paso (frame real t + K − 1 → t + K).
    K = 1   control: el mismo ajuste, solo con pérdida de un paso (separa el efecto de entrenar más épocas)
    K = 2, 4
Ajuste: 60 épocas sobre todos los inicios t de las secuencias con desfase del escenario de entrenamiento, lotes de 8
mallas completas (80 000 celdas), Adam 3e-4 → 1e-5 exponencial, frontera abierta. Pesos guardados con
main.save_weights (se cargan igual en la GUI). p1 → p3 y p3 → p1, semillas 42, 1, 2, 3, 4.

Escribe redes/<ronda>/<nombre>.npz (+ _loss.csv), resultados.csv y registro.log en esta carpeta.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/6_multipaso/multipaso.py
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

KS = [1, 2, 4]
EPOCAS, MALLAS_POR_LOTE, LR, LR_MIN = 60, 8, 3e-4, 1e-5
CAPAS = [(32, "relu"), (32, "relu")]


def cargar_modelo(ruta: Path, dev):
    w = np.load(ruta)
    nb = w["neighbourhood"]
    modelo = cm.build_model(len(nb), CAPAS, int(w["layer_out"][-1]), "linear").to(dev)
    lineales = [m for m in modelo if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for i, capa in enumerate(lineales):
            capa.weight.copy_(torch.from_numpy(w[f"weights_{i}"]))
            capa.bias.copy_(torch.from_numpy(w[f"biases_{i}"]))
    return modelo, nb


def parches(estado: torch.Tensor, nb: np.ndarray) -> torch.Tensor:
    """(B, H, W) → (B·H·W, vecinos), igual que main.extract_patches con frontera abierta (relleno con ceros)."""
    B, H, W = estado.shape
    r = int(np.abs(nb).max())
    p = torch.nn.functional.pad(estado, (r, r, r, r))
    return torch.stack([p[:, r + dy:r + dy + H, r + dx:r + dx + W] for dx, dy in nb], -1).reshape(-1, len(nb))


def ajustar(ruta: Path, seqs: list, K: int, semilla: int, base: Path, dev) -> dict:
    torch.manual_seed(semilla)
    modelo, nb = cargar_modelo(ruta, dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(LR_MIN / LR) ** (1 / (EPOCAS - 1)))
    ce = torch.nn.CrossEntropyLoss()
    inicios = [(i, t) for i, s in enumerate(seqs) for t in range(len(s) - K)]
    t0 = time.perf_counter()
    with open(f"{base}_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "lr"])
        for epoca in range(EPOCAS):
            orden = torch.randperm(len(inicios)).tolist()
            total, lotes = 0.0, 0
            for s in range(0, len(orden), MALLAS_POR_LOTE):
                lote = [inicios[j] for j in orden[s:s + MALLAS_POR_LOTE]]
                x = torch.stack([seqs[i][t] for i, t in lote]).float()
                objetivo = torch.stack([seqs[i][t + K] for i, t in lote]).reshape(-1)
                previo = torch.stack([seqs[i][t + K - 1] for i, t in lote]).float()
                with torch.no_grad():
                    for _ in range(K - 1):
                        x = modelo(parches(x, nb)).argmax(1).reshape(x.shape).float()
                perdida = ce(modelo(parches(x, nb)), objetivo)
                if K > 1:
                    perdida = perdida + ce(modelo(parches(previo, nb)), objetivo)
                opt.zero_grad()
                perdida.backward()
                opt.step()
                total += perdida.item()
                lotes += 1
            w.writerow([epoca, total / lotes, opt.param_groups[0]["lr"]])
            sched.step()
    cm.save_weights(f"{base}.npz", modelo, nb, CAPAS, "linear", 0)
    return {"epocas_ajuste": EPOCAS, "loss_final": round(total / lotes, 6), "inicios": len(inicios),
            "tiempo_s": round(time.perf_counter() - t0, 1)}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = cm.temperaturas()
    cm.log(CARPETA, "Inicio opción 6 (ajuste con simulación de varios pasos)")
    for K in KS:
        for entrena, prueba in cm.RONDAS:
            ronda = cm.ronda_nombre(entrena, prueba)
            seqs = [torch.from_numpy(cm.estados(temps[k])[o::cm.PASO]).long().to(dev)
                    for k in entrena for o in range(cm.PASO)]
            for s in cm.SEMILLAS:
                clave = {"opcion": "6_multipaso", "variante": f"K{K}", "ronda": ronda, "semilla": s}
                if cm.hecho(CARPETA, clave):
                    continue
                (CARPETA / "redes" / ronda).mkdir(parents=True, exist_ok=True)
                base = CARPETA / "redes" / ronda / f"K{K}_k6_r4_cruda_2x32_s{s}"
                cm.log(CARPETA, f"K {K} {ronda} s{s}: ajustando {cm.red_referencia(ronda, s).name}")
                info = ajustar(cm.red_referencia(ronda, s), seqs, K, s, base, dev)
                prob, nb = cm.predictor([f"{base}.npz"], dev, cm.NT)
                fila = {**clave, "prueba": prueba, "paso_frames": cm.PASO, "decodificador": "argmax", "K": K,
                        **info, **cm.evaluar(prob, nb, temps, prueba, dev)}
                cm.escribir(CARPETA, fila)
                cm.log(CARPETA, f"    loss {info['loss_final']:.4f}, {info['tiempo_s']} s; {cm.resumen(fila)}")
    cm.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
