"""1.2 — Métricas de propagación (tiempo de llegada, área quemada nueva, FSS) para todas las redes ya guardadas de
../../opciones (sin reentrenar nada). Las métricas están descritas en ../comun_r.py.

Una fila por opción, variante, ronda y semilla (o "ensamble"); la copia va en las columnas copia_*.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/1_2_propagacion/propagacion.py
"""
import sys
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402

sys.path.insert(0, str(cr.OPC / "3_terreno"))
import terreno                                                            # noqa: E402

O = cr.OPC


def casos():
    """(opción, variante, ronda, semilla, redes, kwargs de propagacion, lado)."""
    for entrena, prueba in cr.RONDAS:
        ronda = cr.ronda_nombre(entrena, prueba)
        for s in cr.SEMILLAS:
            yield "0_base", "sin_p2", ronda, s, [cr.red_referencia(ronda, s)], {}, 100
            yield "0_base", "rotaciones", ronda, s, [cr.red_referencia(ronda, s, "rotaciones", "log15_rot4")], {}, 100
        yield "1_ensamble", "sin_p2", ronda, "ensamble", [cr.red_referencia(ronda, s) for s in cr.SEMILLAS], {}, 100
        yield ("1_ensamble", "rotaciones", ronda, "ensamble",
               [cr.red_referencia(ronda, s, "rotaciones", "log15_rot4") for s in cr.SEMILLAS], {}, 100)
        for s in cr.SEMILLAS:
            yield ("2_estocastico", "muestreo", ronda, s, [cr.red_referencia(ronda, s)],
                   {"decodificador": "muestreo", "repeticiones": 20, "semilla": s}, 100)
            yield "2_estocastico", "esperado", ronda, s, [cr.red_referencia(ronda, s)], {"decodificador": "esperado"}, 100
        for v, n in terreno.VARIANTES.items():
            for s in cr.SEMILLAS:
                yield ("3_terreno", v, ronda, s, [O / "3_terreno" / "redes" / ronda / f"{v}_k6_r4_cruda_2x32_exp_s{s}.npz"],
                       {"canal": "terreno:" + v, "ncanal": n}, 100)
        for a in (0, 3, 10):
            for s in cr.SEMILLAS:
                yield ("5_pesos", f"alfa{a}", ronda, s, [O / "5_pesos" / "redes" / ronda / f"alfa{a}_k6_r4_cruda_2x32_exp_s{s}.npz"],
                       {}, 100)
        for K in (1, 2, 4):
            for s in cr.SEMILLAS:
                yield ("6_multipaso", f"K{K}", ronda, s, [O / "6_multipaso" / "redes" / ronda / f"K{K}_k6_r4_cruda_2x32_s{s}.npz"],
                       {}, 100)
        for s in cr.SEMILLAS:
            yield "7_resolucion200", "200_9x9", ronda, s, [cr.redes_200(ronda, s)], {}, 200
        for s in (42, 1, 2):
            yield ("7_resolucion200", "200_17x17", ronda, s,
                   [O / "7_resolucion200" / "redes" / ronda / f"200_k6_r8_cruda_2x32_exp_s{s}.npz"], {}, 200)
        for paso in (9, 12):
            for radio, vec in ((4, "9x9"), (6, "13x13")):
                for s in cr.SEMILLAS:
                    yield ("8_pasos", f"k{paso}_{vec}", ronda, s,
                           [O / "8_pasos" / "redes" / ronda / f"log15_k{paso}_r{radio}_cruda_2x32_exp_s{s}.npz"],
                           {"paso": paso}, 100)


def main():
    dev = torch.device("cuda")
    cr.log(CARPETA, "Inicio 1.2 (métricas de propagación de las redes guardadas)")
    temps = {100: cr.temperaturas(100), 200: cr.temperaturas(200)}
    mapas = terreno.mapas()
    for opcion, variante, ronda, semilla, redes, kw, lado in casos():
        clave = {"experimento": "1_2_propagacion", "opcion": opcion, "variante": variante, "ronda": ronda,
                 "semilla": semilla}
        if cr.hecho(CARPETA, clave):
            continue
        prueba = ronda.split("_a_")[1]
        kw = dict(kw)
        if "canal" in kw:
            kw["canal"] = mapas[kw["canal"].split(":")[1]]
        n_estados = cr.NT * kw.get("ncanal", 1)
        prob, nb = cr.predictor(redes, dev, n_estados)
        fila = {**clave, "prueba": prueba, "lado": lado, "paso_frames": kw.get("paso", cr.PASO),
                **cr.propagacion(prob, nb, temps[lado], prueba, dev, escala=lado // 100, **kw)}
        cr.escribir(CARPETA, fila)
        cr.log(CARPETA, f"{opcion} {variante} {ronda} {semilla}: llegada {fila['llegada_mae_min']} min "
                        f"(copia {fila['copia_llegada_mae_min']}), área {fila['area_quemada_rel']}, "
                        f"IoU nuevo {fila['iou_quemado_nuevo']}, FSS quemado 1/3/9 {fila['fss_quemado_1']}/"
                        f"{fila['fss_quemado_3']}/{fila['fss_quemado_9']}")
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
