"""Entrenamientos de prueba de NCA sobre Physarum (CA/Physarum/estados_cada30min*).

Entrena con scripts/main.py (20 cajas, POR_CAJA transiciones por caja) y evalúa en las 4 cajas dish5 que no vio:
un paso (todas las transiciones) y simulación libre desde la primera imagen hasta el final, con argmax (como la GUI)
y muestreando de la salida softmax. Cada variante escribe en <variante>/: red.npz, resultados.json, area.csv y
real_vs_red.mp4 (real / argmax / muestreo).

    python prueba.py <variante> [datos | entrenar | evaluar | video | todo]
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[2]
sys.path.insert(0, str(TRAINING / "scripts"))
from barrido_flame import cargar_red                                       # noqa: E402

PY = sys.executable
MAIN = TRAINING / "scripts" / "main.py"

ESTADOS6 = ["borde", "fondo", "avena", "plasmodio tenue", "plasmodio medio", "plasmodio intenso"]
PALETA6 = ["#000000", "#2A2A30", "#E6CFA8", "#6E6A12", "#D2AE00", "#FFF46A"]
PLASMODIO = 3
PRUEBA = [f"physarum-{e}-dish5-r2c1" for e in ("001", "002", "008", "009")]
CAPAS = [64] * 16
EPOCAS = 30
SEMILLA = 42

# Una variante = datos (carpeta de CA/Physarum), cuántos estados (4 = plasmodio de un solo nivel) y transiciones por
# caja (≈ 8.7 GB de muestras float32 con 53 vecinos: 20 a 320 × 320, 80 a 160 × 160).
VARIANTES = {
    "320_6estados": {"ca": "estados_cada30min", "estados": 6, "por_caja": 20},
    "160_6estados": {"ca": "estados_cada30min_160", "estados": 6, "por_caja": 80},
    "160_binario": {"ca": "estados_cada30min_160", "estados": 4, "por_caja": 80},
}
MAPA4 = np.array([0, 1, 2, 3, 3, 3], np.int8)                             # plasmodio tenue/medio/intenso → 3


def configurar(nombre):
    global VARIANTE, CA, SALIDA, DATOS, RED, ESTADOS, PALETA, POR_CAJA, MAPA
    VARIANTE = nombre
    v = VARIANTES[nombre]
    CA = TRAINING / "CA" / "Physarum" / v["ca"]
    SALIDA = CARPETA / nombre
    DATOS, RED = SALIDA / "datos", SALIDA / "red.npz"
    POR_CAJA = v["por_caja"]
    if v["estados"] == 4:
        ESTADOS, PALETA, MAPA = ESTADOS6[:3] + ["plasmodio"], PALETA6[:3] + [PALETA6[4]], MAPA4
    else:
        ESTADOS, PALETA, MAPA = ESTADOS6, PALETA6, None


def cargar(nombre):
    S = np.load(CA / f"{nombre}.npy")
    return MAPA[S] if MAPA is not None else S


def anillos(radios=(1, 2, 4, 6), puntos=(8, 12, 16, 20)) -> list:
    """Centro y anillos de radio r con n puntos (los anillos impares girados medio paso), como en SEVIR."""
    nb = [(0, 0)]
    for i, (r, n) in enumerate(zip(radios, puntos)):
        giro = np.pi / n if i % 2 else 0.0
        for k in range(n):
            a = giro + 2 * np.pi * k / n
            p = (int(round(r * np.cos(a))), int(round(r * np.sin(a))))
            if p not in nb:
                nb.append(p)
    return nb


VECINDAD = anillos()


def cajas_entrenamiento():
    return [p.stem for p in sorted(CA.glob("physarum-*.npy")) if p.stem not in PRUEBA]


def datos():
    """Por caja de entrenamiento, POR_CAJA transiciones repartidas en el tiempo → datos/<caja>.npy (N, 2, H, W)."""
    DATOS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    for nombre in cajas_entrenamiento():
        S = cargar(nombre)
        bordes = np.linspace(0, len(S) - 1, POR_CAJA + 1).astype(int)            # un t por tramo de tiempo
        ts = [rng.integers(a, max(a + 1, b)) for a, b in zip(bordes[:-1], bordes[1:])]
        np.save(DATOS / f"{nombre}.npy", np.stack([S[[t, t + 1]] for t in ts]))
    print(f"{len(cajas_entrenamiento())} cajas × {POR_CAJA} transiciones → {DATOS}")


def entrenar():
    cmd = [PY, "-u", str(MAIN), "--data", *map(str, sorted(DATOS.glob("*.npy"))), "--output", str(RED),
           "--neighbourhood", " ".join(f"{dx},{dy}" for dx, dy in VECINDAD),
           "--layers", " ".join(f"{c}:relu" for c in CAPAS), "--out-size", str(len(ESTADOS)), "--out-act", "linear",
           "--one-hot", str(len(ESTADOS)), "--epochs", str(EPOCAS), "--batch", "8192", "--lr", "1e-3",
           "--lr-schedule", "exp", "--lr-min", "1e-5", "--loss-target", "1e-9", "--print-every", "1",
           "--seed", str(SEMILLA), "--open-boundary", "--log-csv", str(SALIDA / "perdida.csv")]
    t0 = time.perf_counter()
    with open(SALIDA / "entrenamiento.log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=TRAINING)
    if r.returncode != 0:
        raise RuntimeError(f"falló main.py, ver {SALIDA / 'entrenamiento.log'}")
    print(f"entrenada en {time.perf_counter() - t0:.0f} s → {RED}")


def red_de_pasos(dev):
    """(paso_argmax, probas) de la red guardada; frontera abierta = 0 (borde), como la GUI."""
    red, nb = cargar_red(str(RED), dev)
    nb = [tuple(map(int, p)) for p in nb]
    r = max(max(abs(dx), abs(dy)) for dx, dy in nb)

    @torch.no_grad()
    def logits(E):
        H, W = E.shape
        Pd = torch.nn.functional.pad(E, (r, r, r, r))
        X = torch.stack([Pd[r + dy:r + dy + H, r + dx:r + dx + W] for dx, dy in nb], -1).reshape(-1, len(nb))
        return torch.cat([red(X[s:s + (1 << 18)]) for s in range(0, len(X), 1 << 18)]).view(H, W, -1)

    return (lambda E: logits(E).argmax(-1).to(torch.int8)), (lambda E: logits(E).softmax(-1))


def iou(a, b):
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else 1.0


@torch.no_grad()
def evaluar():
    dev = torch.device("cuda")
    paso, probas = red_de_pasos(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    res, filas = {}, []
    tot = {k: 0.0 for k in ("n", "ok", "copia", "cambia", "cambia_ok", "nuevo", "nuevo_ok", "p_nuevo", "queda",
                            "p_queda")}
    for nombre in PRUEBA:
        S = torch.from_numpy(cargar(nombre)).to(dev)
        dentro = S[0] != 0
        for t in range(len(S) - 1):                                    # un paso desde la imagen real
            pr = probas(S[t])
            p = pr.argmax(-1)[dentro]
            p_pl = pr[..., PLASMODIO:].sum(-1)[dentro]
            x, y = S[t][dentro], S[t + 1][dentro]
            cambia = x != y
            nuevo = (x < PLASMODIO) & (y >= PLASMODIO)
            queda = (x < PLASMODIO) & (y < PLASMODIO)
            tot["n"] += len(y)
            tot["ok"] += (p == y).sum().item()
            tot["copia"] += (x == y).sum().item()
            tot["cambia"] += cambia.sum().item()
            tot["cambia_ok"] += (p[cambia] == y[cambia]).sum().item()
            tot["nuevo"] += nuevo.sum().item()
            tot["nuevo_ok"] += (p[nuevo] >= PLASMODIO).sum().item()
            tot["p_nuevo"] += p_pl[nuevo].sum().item()
            tot["queda"] += queda.sum().item()
            tot["p_queda"] += p_pl[queda].sum().item()
        libres = {}
        for modo in ("argmax", "muestreo"):                            # libre desde la primera imagen real
            E = S[0].clone()
            ev = [E]
            for t in range(1, len(S)):
                if modo == "argmax":
                    E = paso(E)
                else:
                    pr = probas(E)
                    E = torch.multinomial(pr.view(-1, pr.shape[-1]), 1, generator=g).view(E.shape).to(torch.int8)
                ev.append(E)
            libres[modo] = torch.stack(ev)
            np.save(SALIDA / f"libre_{modo}_{nombre}.npy", libres[modo].cpu().numpy())
        n = dentro.sum().item()
        for t in range(len(S)):
            fila = {"caja": nombre, "h": t / 2, "real": round((S[t][dentro] >= PLASMODIO).sum().item() / n, 4)}
            for modo, L in libres.items():
                fila[modo] = round((L[t][dentro] >= PLASMODIO).sum().item() / n, 4)
                fila[f"iou_{modo}"] = round(iou(S[t] >= PLASMODIO, L[t] >= PLASMODIO), 4)
            filas.append(fila)
        horas = [h for h in (6, 12, 24, 48) if 2 * h < len(S)] + [(len(S) - 1) / 2]
        res[nombre] = {f"{h:g} h": {k: v for k, v in filas[-len(S) + int(2 * h)].items() if k not in ("caja", "h")}
                       for h in horas}
        res[nombre]["borde_conservado"] = bool((libres["argmax"][:, ~dentro] == 0).all().item())
        print(nombre, json.dumps(res[nombre]))
    res["un_paso"] = {"exactitud": tot["ok"] / tot["n"], "copia": tot["copia"] / tot["n"],
                      "exactitud_en_cambios": tot["cambia_ok"] / tot["cambia"],
                      "plasmodio_nuevo_detectado": tot["nuevo_ok"] / tot["nuevo"],
                      "p_plasmodio_en_nuevos": tot["p_nuevo"] / tot["nuevo"],
                      "p_plasmodio_en_sin_cambio": tot["p_queda"] / tot["queda"]}
    print("un paso:", {k: round(v, 4) for k, v in res["un_paso"].items()})
    (SALIDA / "resultados.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    with open(SALIDA / "area.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def video(tile=300, fps=10):
    from PIL import Image, ImageDraw, ImageFont

    fuente = lambda s: ImageFont.truetype("/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf", s)
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PALETA], np.uint8)
    filas = [("real", None), ("NCA argmax", "argmax"), ("NCA muestreo", "muestreo")]
    series = [[cargar(n) if m is None else np.load(SALIDA / f"libre_{m}_{n}.npy") for _, m in filas] for n in PRUEBA]
    T = max(len(s[0]) for s in series)
    LEFT, HEAD, LAB, sep = 20, 44, 24, 8
    W = LEFT + len(PRUEBA) * (tile + sep)
    H = HEAD + len(filas) * (tile + LAB) + 34
    W, H = W + W % 2, H + H % 2
    out = SALIDA / "real_vs_red.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libopenh264", "-b:v", "6M", "-pix_fmt", "yuv420p",
                           str(out)], stdin=subprocess.PIPE)
    f1, f2 = fuente(22), fuente(15)
    for t in range(T):
        img = Image.new("RGB", (W, H), (20, 20, 20))
        dr = ImageDraw.Draw(img)
        dr.text((12, 10), f"Physarum {VARIANTE}: real vs NCA libre desde t = 0   {t / 2:5.1f} h", fill=(235,) * 3,
                font=f1)
        for j, (n, evos) in enumerate(zip(PRUEBA, series)):
            k = min(t, len(evos[0]) - 1)
            fin = t >= len(evos[0])
            x = LEFT + j * (tile + sep)
            for i, E in enumerate(evos):
                y = HEAD + i * (tile + LAB)
                c = rgb[E[k]]
                if fin:
                    c = (c * 0.45).astype(np.uint8)
                img.paste(Image.fromarray(c).resize((tile, tile), Image.NEAREST), (x, y))
                dr.text((x + 4, y + tile + 3), f"{n[9:18]} {filas[i][0]}" + ("  fin" if fin else ""),
                        fill=(210,) * 3, font=f2)
        xl = LEFT
        for i, e in enumerate(ESTADOS):
            dr.rectangle([xl, H - 26, xl + 12, H - 14], fill=PALETA[i], outline=(120,) * 3)
            dr.text((xl + 16, H - 29), f"{i} {e}", fill=(220,) * 3, font=f2)
            xl += int(dr.textlength(f" {i} {e}  ", font=f2)) + 14
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {T} imágenes, {T / fps:.1f} s")


def video_uno(caja="physarum-008-dish5-r2c1", modo="muestreo", tile=480, sep=8, fps=10):
    """Una caja, real (izquierda) vs NCA libre (derecha), sin texto → <variante>/<caja>_real_vs_<modo>.mp4."""
    from PIL import Image

    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PALETA], np.uint8)
    S, L = cargar(caja), np.load(SALIDA / f"libre_{modo}_{caja}.npy")
    W, H = 2 * tile + sep, tile
    out = SALIDA / f"{caja}_real_vs_{modo}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libopenh264", "-b:v", "6M", "-pix_fmt", "yuv420p",
                           str(out)], stdin=subprocess.PIPE)
    for t in range(len(S)):
        img = Image.new("RGB", (W, H), (0, 0, 0))
        for j, E in enumerate((S, L)):
            img.paste(Image.fromarray(rgb[E[t]]).resize((tile, tile), Image.NEAREST), (j * (tile + sep), 0))
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {len(S)} imágenes, {len(S) / fps:.1f} s")


if __name__ == "__main__":
    configurar(sys.argv[1])
    pasos = {"datos": [datos], "entrenar": [entrenar], "evaluar": [evaluar], "video": [video],
             "uno": [video_uno], "todo": [datos, entrenar, evaluar, video, video_uno]}[
        sys.argv[2] if len(sys.argv) > 2 else "todo"]
    for f in pasos:
        f()
