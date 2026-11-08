"""Resumen del barrido, redes finales y GIF desde el frame 26 de flame_plot_2 (12 estados).

1. resumen.csv: promedio por configuración (paso, red, vecindad, decodificador) de las dos rondas y las semillas.
   Proporciones contra la copia (< 1 = mejor que copiar): mae_rel, rollout_rel, llegada_rel; y FSS, IoU, área.
2. Redes finales, entrenadas con p1 + p2 + p3 (semilla 42), para:
       mejor_un_paso       menor mae_rel con argmax
       mejor_propagacion   menor llegada_rel (con el decodificador con el que la logró)
       referencia          paso 6, 9×9, 2 capas de 32 (la mejor configuración de los experimentos anteriores)
       tu_config           paso 1, vecindad dispersa de 41 celdas, 4 capas de 256 (la de Personal/test_01)
   Para cada una se calibra el sesgo de encendido con los tres conjuntos y se guarda también <red>_sesgo.npz, que la
   GUI usa sin cambios. Si existe NCA/FLAME3/Personal/test_01.npz también se incluye (test_01).
3. GIF (gif/): simulación libre desde el frame 26 de flame_plot_2 (estados12) hasta el final, mostrada cada 30 s:
       final_<nombre>.gif           Real | NCA | NCA + sesgo, con las redes finales (vieron flame_plot_2)
       validacion_<config>.gif      lo mismo con la red de la ronda {p1, p3} → p2, semilla 42 (no vio flame_plot_2),
                                    a escala 1
       mosaico_k<paso>_<dec>.gif    Real y las 9 configuraciones de ese paso, ronda {p1, p3} → p2, semilla 42,
                                    a escala 1
   indice_gifs.csv: qué es cada GIF y sus métricas de la simulación desde el frame 26.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/flame3_12estados/finales.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
import comun12 as c                                                       # noqa: E402

FIN = CARPETA / "finales"
GIF = CARPETA / "gif"
ZOOM = 2                      # los mosaicos van con zoom 1 (son 10 paneles)
CADA = 6                      # un cuadro del GIF cada 6 frames (30 s)
RONDA_GIF = "p1+p3_a_p2"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def resumen() -> list:
    g = defaultdict(list)
    for r in csv.DictReader(open(CARPETA / "resultados.csv")):
        g[(int(r["paso"]), r["red"], r["vecindad"], r["decodificador"])].append(r)
    filas = []
    for (paso, red, vec, dec), rs in g.items():
        m = lambda k: float(np.nanmean([num(r.get(k)) for r in rs]))         # noqa: E731
        p2 = [r for r in rs if r["prueba"] == "p2"]
        mg = lambda k: float(np.nanmean([num(r.get(k)) for r in p2])) if p2 else np.nan   # noqa: E731
        filas.append({"paso": paso, "red": red, "vecindad": vec, "decodificador": dec, "n": len(rs),
                      "mae_rel": round(m("mae_c") / m("copia_mae_c"), 4),
                      "rollout_rel": round(m("rollout_mae_c") / m("copia_rollout_mae_c"), 4),
                      "llegada_rel": round(m("llegada_mae_min") / m("copia_llegada_mae_min"), 4),
                      "mae_c": round(m("mae_c"), 3), "rollout_mae_c": round(m("rollout_mae_c"), 3),
                      "llegada_mae_min": round(m("llegada_mae_min"), 3), "encendido": round(m("encendido"), 4),
                      "precision_encendido": round(m("precision_encendido"), 4),
                      "area_quemada_rel": round(m("area_quemada_rel"), 4),
                      "iou_quemado_nuevo": round(m("iou_quemado_nuevo"), 4),
                      "fss_quemado_9": round(m("fss_quemado_9"), 4), "fss_fuego_3": round(m("fss_fuego_3"), 4),
                      "gif_rollout_mae_c": round(mg("gif_rollout_mae_c"), 3),
                      "gif_llegada_mae_min": round(mg("gif_llegada_mae_min"), 3),
                      "gif_iou_quemado_nuevo": round(mg("gif_iou_quemado_nuevo"), 4),
                      "gif_fss_quemado_9": round(mg("gif_fss_quemado_9"), 4),
                      "tiempo_s": round(m("tiempo_s"), 1)})
    filas.sort(key=lambda f: (f["paso"], f["red"], f["vecindad"], f["decodificador"]))
    with open(CARPETA / "resumen.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return filas


# ── GIF ──────────────────────────────────────────────────────────────────────

def fuente(tam):
    for ttf in ("/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"):
        if Path(ttf).exists():
            return ImageFont.truetype(ttf, tam)
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:
        return ImageFont.load_default()


PAL = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in c.PALETA.read_text().split()], dtype=np.uint8)
F_TIT, F_CAB = fuente(13), fuente(14)
# Paleta fija del GIF: los 12 estados más blanco, negro y gris; sin tramado, así el GIF comprime bien
_fija = np.vstack([PAL, [[255, 255, 255], [0, 0, 0], [220, 220, 220]]]).astype(np.uint8)
PAL_GIF = Image.new("P", (1, 1))
PAL_GIF.putpalette(_fija.ravel().tolist() + [0] * (768 - _fija.size))


def cuadro(paneles: list, columnas: int, cabecera: str, zoom: int = ZOOM) -> Image.Image:
    """paneles: [(título, estados (H, W) o None)]."""
    lado = 100 * zoom
    alto_tit, alto_cab, sep = 18, 22, 6
    filas = (len(paneles) + columnas - 1) // columnas
    W = columnas * lado + (columnas + 1) * sep
    H = alto_cab + filas * (lado + alto_tit + sep) + sep
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.fontmode = "1"                                         # texto sin suavizado (solo blanco y negro)
    d.text((sep, 3), cabecera, fill="black", font=F_CAB)
    for i, (titulo, est) in enumerate(paneles):
        x = sep + (i % columnas) * (lado + sep)
        y = alto_cab + (i // columnas) * (lado + alto_tit + sep)
        d.text((x, y), titulo, fill="black", font=F_TIT)
        if est is None:
            d.rectangle([x, y + alto_tit, x + lado - 1, y + alto_tit + lado - 1], fill=(220, 220, 220))
            d.text((x + 8, y + alto_tit + lado // 2 - 8), "sin red", fill="black", font=F_TIT)
        else:
            im.paste(Image.fromarray(PAL[est].repeat(zoom, 0).repeat(zoom, 1)), (x, y + alto_tit))
    return im.quantize(palette=PAL_GIF, dither=Image.Dither.NONE)


def guardar_gif(destino: Path, series: list, columnas: int, titulo: str, zoom: int = ZOOM):
    """series: [(título del panel, [estados por cuadro] o None)]."""
    n = max(len(s) for _, s in series if s is not None)
    cuadros = []
    for j in range(n):
        cab = f"{titulo}   t = +{j * CADA * 5} s (frame {c.INICIO_GIF + j * CADA})"
        cuadros.append(cuadro([(t, s[j] if s is not None else None) for t, s in series], columnas, cab, zoom))
    cuadros[0].save(destino, save_all=True, append_images=cuadros[1:], duration=450, loop=0, optimize=True)


def simulacion(prob, nb, S0, paso, n_cuadros):
    """Estados en los tiempos 26 + 6j (j = 0…n−1) simulando con el paso de la red."""
    cada = CADA // paso
    Ts = c.trayectoria(prob, nb, S0, (n_cuadros - 1) * cada)
    return [S0] + [Ts[j * cada - 1] for j in range(1, n_cuadros)]


def main():
    dev = torch.device("cuda")
    S, T = c.estados_y_temps()
    filas = resumen()
    c.log(CARPETA, f"Resumen: {len(filas)} configuraciones con decodificador")
    for f in filas:
        c.log(CARPETA, f"  k{f['paso']} {f['vecindad']} {f['red']} {f['decodificador']} n={f['n']}: mae {f['mae_rel']} "
                       f"rollout {f['rollout_rel']} llegada {f['llegada_rel']} FSS9 {f['fss_quemado_9']} "
                       f"IoU nuevo {f['iou_quemado_nuevo']}")
    arg = [f for f in filas if f["decodificador"] == "argmax"]
    elegidas = {}
    if arg:
        b = min(arg, key=lambda f: f["mae_rel"])
        elegidas["mejor_un_paso"] = (b["paso"], b["vecindad"], b["red"])
        b = min(filas, key=lambda f: f["llegada_rel"] if np.isfinite(f["llegada_rel"]) else 9)
        elegidas["mejor_propagacion"] = (b["paso"], b["vecindad"], b["red"])
    elegidas["referencia"] = (6, "9x9", "2x32")
    elegidas["tu_config"] = (1, "dispersa41", "4x256")
    vistas, finales = set(), {}
    for nombre, cfg in elegidas.items():
        if cfg in vistas:
            nombre_prev = [n for n, v in finales.items() if v == cfg][0]
            c.log(CARPETA, f"{nombre} = {nombre_prev} ({cfg})")
            continue
        vistas.add(cfg)
        finales[nombre] = cfg
    c.log(CARPETA, f"Redes finales: {finales}")

    FIN.mkdir(exist_ok=True)
    GIF.mkdir(exist_ok=True)
    n_cuadros = (len(S["p2"]) - 1 - c.INICIO_GIF) // CADA + 1
    real = [S["p2"][c.INICIO_GIF + j * CADA] for j in range(n_cuadros)]
    t_real = T["p2"][c.INICIO_GIF::CADA][:n_cuadros]
    indice = []

    def registrar(archivo, tipo, nombre, cfg, prob, nb, delta, paso):
        for dec, p in (("argmax", prob), ("sesgo", c.con_sesgo(prob, delta))):
            sim = simulacion(p, nb, S["p2"][c.INICIO_GIF], paso, n_cuadros)
            m = c.propagacion(sim[1:], t_real, CADA)
            indice.append({"gif": archivo, "tipo": tipo, "nombre": nombre, "paso": cfg[0], "vecindad": cfg[1],
                           "red": cfg[2], "decodificador": dec, "delta": delta,
                           **{f"desde26_{k}": v for k, v in m.items()}})

    # finales
    todos = list(c.ESC)
    for nombre, (paso, vec, red) in finales.items():
        base = FIN / f"{nombre}_k{paso}_{vec}_{red}"
        if not Path(f"{base}.npz").exists():
            datos = [CARPETA / "datos" / f"{k}_k{paso}_o{o}.npy" for k in todos for o in range(paso)]
            c.log(CARPETA, f"final {nombre}: entrenando k{paso} {vec} {red} con p1 + p2 + p3")
            info = c.entrenar(datos, base, vec, red, 42)
            c.log(CARPETA, f"    {info}")
        prob, nb = c.predictor(f"{base}.npz", dev)
        cal = c.calibrar_sesgo(prob, nb, [q for k in todos for q in c.secuencias(S[k], paso)],
                               [q for k in todos for q in c.secuencias(T[k], paso)])
        c.guardar_con_sesgo(Path(f"{base}.npz"), Path(f"{base}_sesgo.npz"), cal["delta"])
        archivo = f"final_{nombre}_k{paso}_{vec}_{red}.gif"
        series = [("Real", real)]
        for t, p in (("NCA", prob), (f"NCA + sesgo ({cal['delta']:.2f})", c.con_sesgo(prob, cal["delta"]))):
            series.append((t, simulacion(p, nb, S["p2"][c.INICIO_GIF], paso, n_cuadros)))
        guardar_gif(GIF / archivo, series, 3, f"Final {nombre}: k{paso} {vec} {red}")
        registrar(archivo, "final", nombre, (paso, vec, red), prob, nb, cal["delta"], paso)
        c.log(CARPETA, f"GIF {archivo} (delta {cal['delta']})")

    test01 = c.TRAINING / "NCA" / "FLAME3" / "Personal" / "test_01.npz"
    if test01.exists():
        prob, nb = c.predictor(test01, dev)
        cal = c.calibrar_sesgo(prob, nb, [S[k] for k in todos], [T[k] for k in todos])
        archivo = "final_test_01_k1_dispersa41_4x256.gif"
        series = [("Real", real)]
        for t, p in (("NCA", prob), (f"NCA + sesgo ({cal['delta']:.2f})", c.con_sesgo(prob, cal["delta"]))):
            series.append((t, simulacion(p, nb, S["p2"][c.INICIO_GIF], 1, n_cuadros)))
        guardar_gif(GIF / archivo, series, 3, "Tu test_01 (k1 dispersa41 4x256, 2001 épocas)")
        registrar(archivo, "final", "test_01", (1, "dispersa41", "4x256"), prob, nb, cal["delta"], 1)
        c.log(CARPETA, f"GIF {archivo} (delta {cal['delta']})")

    # validación cruzada: la red de {p1, p3} → p2, semilla 42
    deltas = {(int(r["paso"]), r["vecindad"], r["red"]): float(r["delta"])
              for r in csv.DictReader(open(CARPETA / "resultados.csv"))
              if r["ronda"] == RONDA_GIF and r["semilla"] == "42" and r["decodificador"] == "sesgo"}
    for paso in (6, 3, 1):
        mosaico = {"argmax": [("Real", real)], "sesgo": [("Real", real)]}
        for red in c.REDES:
            for vec in c.VECINDADES:
                ruta = CARPETA / "redes" / RONDA_GIF / f"k{paso}_{vec}_{red}_s42.npz"
                etiqueta = f"{red} {vec}"
                if not ruta.exists() or (paso, vec, red) not in deltas:
                    mosaico["argmax"].append((etiqueta, None))
                    mosaico["sesgo"].append((etiqueta, None))
                    continue
                prob, nb = c.predictor(ruta, dev)
                d = deltas[(paso, vec, red)]
                sim_a = simulacion(prob, nb, S["p2"][c.INICIO_GIF], paso, n_cuadros)
                sim_s = simulacion(c.con_sesgo(prob, d), nb, S["p2"][c.INICIO_GIF], paso, n_cuadros)
                mosaico["argmax"].append((etiqueta, sim_a))
                mosaico["sesgo"].append((etiqueta, sim_s))
                archivo = f"validacion_k{paso}_{vec}_{red}.gif"
                guardar_gif(GIF / archivo, [("Real", real), ("NCA", sim_a), (f"+ sesgo ({d:.2f})", sim_s)], 3,
                            f"Validación k{paso} {vec} {red}", zoom=1)
                registrar(archivo, "validacion", "", (paso, vec, red), prob, nb, d, paso)
        for dec, series in mosaico.items():
            if all(s is None for _, s in series[1:]):
                continue
            archivo = f"mosaico_k{paso}_{dec}.gif"
            guardar_gif(GIF / archivo, series, 5, f"Validación (no vio p2), paso {paso * 5} s, {dec}", zoom=1)
            c.log(CARPETA, f"GIF {archivo}")
    with open(GIF / "indice_gifs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(indice[0]))
        w.writeheader()
        w.writerows(indice)
    tam = sum(p.stat().st_size for p in GIF.glob("*.gif"))
    c.log(CARPETA, f"GIF listos: {len(list(GIF.glob('*.gif')))} archivos, {tam / 1e6:.1f} MB")
    c.log(CARPETA, "Fin finales")


if __name__ == "__main__":
    main()
