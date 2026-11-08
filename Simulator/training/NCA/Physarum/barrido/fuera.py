"""Generalización a un experimento no visto: la mejor configuración (e4_5h_9x9_da) entrenada sin un experimento
completo (18 cajas de los otros 3), sesgo calibrado en dish0 y dish1 de esos 3, prueba en las 6 cajas del
experimento excluido. Todo en fuera/ (datos, redes, resultados.csv); referencias congelarse / llenar la caja.

    python fuera.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

import barrido as B

EXPERIMENTOS = ("001", "002", "008", "009")
CFG = {**B.BASE, "mapa": "4", "paso": 10, "vecindad": "9x9", "desfases": True, "aumento": True}
SALIDA = Path(__file__).resolve().parent / "fuera"


def referencias(cfg, cajas, dev):
    _, pl = B.info(cfg)
    fr, fi = [], []
    for c in cajas:
        S = torch.from_numpy(B.cargar(cfg, c)).to(dev)
        d = S[0] != 0
        fr.append(B.puntaje_t(S, S[:1].repeat(len(S), 1, 1).unsqueeze(0), pl)[0, 1:].mean().item())
        lleno = torch.where(d, torch.tensor(pl, dtype=torch.int8, device=dev), torch.zeros_like(S[0]))
        fi.append(B.puntaje_t(S, lleno.unsqueeze(0).repeat(len(S), 1, 1).unsqueeze(0), pl)[0, 1:].mean().item())
    return float(np.mean(fr)), float(np.mean(fi))


def main():
    dev = torch.device("cuda")
    todas = sorted(p.stem for p in (B.CA / CFG["ca"]).glob("physarum-*.npy"))
    B.RESULTADOS = SALIDA / "resultados.csv"
    SALIDA.mkdir(exist_ok=True)
    resumen = []
    for exp in EXPERIMENTOS:
        nombre = f"sin_{exp}"
        B.CARPETA = SALIDA / nombre                        # datos/ y redes/ propios: la caché de datos no distingue
                                                           # qué cajas son de entrenamiento
        B.PRUEBA = [c for c in todas if c.startswith(f"physarum-{exp}-")]
        B.CALIBRACION = [f"physarum-{e}-dish{d}-r0c{d}" for e in EXPERIMENTOS if e != exp for d in (0, 1)]
        if not (B.CARPETA / "redes" / nombre / "red_gui.npz").exists():
            B.correr(nombre, CFG, f"fuera_{exp}")
        f = next(r for r in csv.DictReader(open(B.RESULTADOS)) if r["nombre"] == nombre)
        congelar, llenar = referencias(CFG, B.PRUEBA, dev)
        resumen.append({"excluido": exp, "puntaje": float(f["puntaje"]), "congelar": congelar, "llenar": llenar,
                        "iou": float(f["iou"]), "area_red_fin": float(f["area_red_fin"]),
                        "area_real_fin": float(f["area_real_fin"]), "sesgo": float(f["sesgo"])})
        print(resumen[-1], flush=True)
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
