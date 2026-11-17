"""Generalización a un experimento no visto de la red con ruido y edad (como barrido/fuera.py con las redes con
sesgo): por cada experimento excluido entrena la maestra e6_6h_circ6_da_edad y la alumna (empujón 0.35, 18 + 18
pares, semilla 43) solo con las 18 cajas de los otros 3, y prueba en las 6 cajas del excluido.
Todo en fuera_dado/ (cada exclusión con sus datos y redes); referencias congelarse / llenar la caja.

    python NCA/Physarum/edad/fuera_dado.py
"""
import json
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "dado"))
sys.path.insert(0, str(CARPETA.parent / "barrido"))
import barrido as B                                                     # noqa: E402
import dado_largo as D                                                  # noqa: E402
from fuera import EXPERIMENTOS, referencias                             # noqa: E402

CFG = {**B.BASE, "paso": 12, "vecindad": "circ6", "desfases": True, "aumento": True, "edad": True}
SALIDA = CARPETA / "fuera_dado"


def main():
    dev = torch.device("cuda")
    todas = sorted(p.stem for p in (B.CA / CFG["ca"]).glob("physarum-*.npy"))
    SALIDA.mkdir(exist_ok=True)
    resumen = []
    for exp in EXPERIMENTOS:
        maestra = f"maestra_sin_{exp}"
        B.CARPETA = SALIDA / f"sin_{exp}"                                 # datos/ y redes/ propios
        B.RESULTADOS = SALIDA / "maestras.csv"
        B.PRUEBA = [c for c in todas if c.startswith(f"physarum-{exp}-")]
        B.CALIBRACION = [f"physarum-{e}-dish{d}-r0c{d}" for e in EXPERIMENTOS if e != exp for d in (0, 1)]
        if not (B.CARPETA / "redes" / maestra / "red.npz").exists():
            B.correr(maestra, CFG, f"fuera_dado_{exp}")
        D.CARPETA = B.CARPETA                                             # datos_dado/ y redes_dado/ propios
        cfg, red_maestra = D.preparar(maestra, 0.35, dev)
        cfg["_maestra"] = maestra
        datos = D.datos(cfg, red_maestra, 0.35, 18, 18, dev)
        del red_maestra
        torch.cuda.empty_cache()
        alumna = f"alumna_sin_{exp}"
        if not (D.CARPETA / "redes_dado" / alumna / "red.npz").exists():
            D.entrenar(alumna, datos, "16x128", 43)
        r = D.evaluar(alumna, cfg, dev)["prueba"]
        congelar, llenar = referencias(CFG, B.PRUEBA, dev)
        resumen.append({"excluido": exp, "C": r["C"], "congelar": congelar, "llenar": llenar,
                        "area_fin": r["area_fin"], "borde": r["borde"], "brillo": r.get("brillo"),
                        "area_real": float(sum(((torch.from_numpy(B.cargar(CFG, c))[-1] >= 3)
                                                & (torch.from_numpy(B.cargar(CFG, c))[0] != 0)).sum().item()
                                               / (torch.from_numpy(B.cargar(CFG, c))[0] != 0).sum().item()
                                               for c in B.PRUEBA) / len(B.PRUEBA))})
        print(resumen[-1], flush=True)
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
