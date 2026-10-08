# training/preprocessing/embryo_to_evolution.py
#
# Segmentación de secuencias time-lapse de embriones (CA/embryo/<id>/*RUN<n>.jpeg) al formato
# del simulador: (T, n, n) int32
#
#   1. Lee los frames en orden numérico de RUN. 211 embriones repiten las mismas imágenes en
#      F0/; se toma una sola copia por RUN.
#   2. Localiza la zona pelúcida UNA vez por embrión:
#        - círculo interior r_in: disco de alta desviación estándar temporal (las células se
#          mueven, la zona no), ajustado con RANSAC;
#        - círculo exterior R: donde termina la textura fina de la zona en la imagen media.
#      Los frames sin textura dentro del círculo (embrión fuera de cuadro, imagen en blanco) son
#      inválidos: se conserva el tramo continuo más largo de frames válidos y se recalcula el
#      círculo solo con ellos.
#   3. Recorta el cuadrado del círculo exterior y lo reduce a n×n (promedio por área).
#   4. Estados:
#        0 fuera del círculo exterior
#        1 zona pelúcida (r_in < r <= R)
#        2 interior
#        3 borde: Sobel sobre el frame n×n con histéresis (débil > Otsu, fuerte > 1.5·Otsu: un
#          borde débil se conserva solo si toca uno fuerte) y mediana temporal de 3 frames
#        4 brillo: intensidad > media + k·σ del interior, en celdas que no son borde
#      Los umbrales se calculan con todos los frames del embrión (no por frame).
#
# Uso mínimo:
#   python preprocessing/embryo_to_evolution.py CA/embryo/AA83-7
#
# Varios embriones (salida en CA/embryo_states/<id>.npy|.mat|.json):
#   python preprocessing/embryo_to_evolution.py CA/embryo/AA83-7 CA/embryo/AG782-8 --gif
#
# Ver README_embryo.md para la descripción completa del proceso.

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image
from scipy import ndimage as ndi

from celsius_to_evolution import downsample

STATE_NAMES = ["fuera", "zona pelúcida", "interior", "borde", "brillo"]
STATE_COLORS = ["#000000", "#3b82f6", "#2f7d4f", "#facc15", "#f472b6"]
RUN_RE = re.compile(r"RUN(\d+)\.jpe?g$", re.IGNORECASE)


# ── Lectura ───────────────────────────────────────────────────────────────────

def frame_paths(embryo_dir: Path) -> list:
    """Un archivo por número de RUN (prefiere la copia menos anidada), en orden numérico."""
    by_run = {}
    for p in sorted(embryo_dir.rglob("*"), key=lambda p: len(p.parts)):
        m = RUN_RE.search(p.name)
        if m:
            by_run.setdefault(int(m.group(1)), p)
    return [by_run[r] for r in sorted(by_run)]


def load_stack(paths: list) -> np.ndarray:
    return np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in paths])


def temporal_mean_std(stack: np.ndarray, chunk: int = 64) -> tuple:
    """Media y desviación estándar por píxel en [0, 1], por bloques para no duplicar memoria."""
    s = np.zeros(stack.shape[1:], np.float64)
    s2 = np.zeros_like(s)
    for i in range(0, len(stack), chunk):
        x = stack[i:i + chunk].astype(np.float64) / 255
        s += x.sum(0)
        s2 += (x ** 2).sum(0)
    mean = s / len(stack)
    return mean.astype(np.float32), np.sqrt(np.maximum(s2 / len(stack) - mean ** 2, 0)).astype(np.float32)


# ── Zona pelúcida ─────────────────────────────────────────────────────────────

def interior_circle(sd: np.ndarray) -> tuple:
    """Círculo (cx, cy, r) del interior de la zona: componente de alta variación temporal."""
    from skimage.filters import threshold_otsu
    from skimage.measure import CircleModel, find_contours, ransac

    s = ndi.gaussian_filter(sd, 3)
    lab, k = ndi.label(s > threshold_otsu(s))
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    areas = ndi.sum(np.ones_like(lab), lab, range(1, k + 1))
    candidates = [(a, i + 1) for i, a in enumerate(areas) if i + 1 not in border]
    if not candidates:
        raise RuntimeError("no se encontró una región interior que no toque el borde de la imagen")
    m = lab == max(candidates)[1]
    m = ndi.binary_opening(ndi.binary_fill_holes(ndi.binary_closing(m, iterations=5)), iterations=5)
    pts = max(find_contours(m.astype(float), 0.5), key=len)[:, ::-1]   # (x, y)
    model, _ = ransac(pts, CircleModel, min_samples=3, residual_threshold=3, max_trials=500, rng=0)
    cx, cy, r = model.params
    return float(cx), float(cy), float(r)


def radial_profile(img: np.ndarray, cx: float, cy: float, rmax: int) -> np.ndarray:
    yy, xx = np.indices(img.shape)
    rad = np.hypot(xx - cx, yy - cy).astype(int)
    ok = rad < rmax
    return np.bincount(rad[ok], img[ok], minlength=rmax) / np.maximum(np.bincount(rad[ok], minlength=rmax), 1)


def outer_radius(mean: np.ndarray, cx: float, cy: float, r_in: float, dmin: int, dmax: int) -> int:
    """Borde exterior: mayor caída radial de la textura fina (|I − G_2 * I|) de la imagen media."""
    hp = np.abs(mean - ndi.gaussian_filter(mean, 2))
    p = ndi.gaussian_filter1d(radial_profile(hp, cx, cy, int(r_in + dmax + 10)), 2)
    lo, hi = int(r_in + dmin), int(r_in + dmax)
    return lo + int(np.argmin(np.gradient(p)[lo:hi]))


def valid_segment(stack: np.ndarray, cx: float, cy: float, r: float, ratio: float) -> tuple:
    """
    Tramo continuo más largo de frames con textura dentro del círculo. Un frame es inválido si
    su textura media |I − G_2 * I| es menor que ratio × percentil 90 de la secuencia.
    """
    yy, xx = np.indices(stack.shape[1:])
    inside = np.hypot(xx - cx, yy - cy) <= r
    tex = np.array([np.abs(f - ndi.gaussian_filter(f, 2))[inside].mean()
                    for f in stack.astype(np.float32) / 255])
    ok = tex >= ratio * np.percentile(tex, 90)
    best, start = (0, 0), None
    for t, v in enumerate(np.append(ok, False)):
        if v and start is None:
            start = t
        elif not v and start is not None:
            best = max(best, (start, t), key=lambda b: b[1] - b[0])
            start = None
    return best, int((~ok).sum())


# ── Recorte, reducción y estados ──────────────────────────────────────────────

def crop_square(stack: np.ndarray, cx: float, cy: float, R: int) -> tuple:
    """Cuadrado de lado 2R centrado en el círculo; si sale de la imagen se rellena con el borde."""
    x0, y0 = int(round(cx - R)), int(round(cy - R))
    pad = max(0, -x0, -y0, x0 + 2 * R - stack.shape[2], y0 + 2 * R - stack.shape[1])
    if pad:
        stack = np.pad(stack, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    return stack[:, y0 + pad:y0 + pad + 2 * R, x0 + pad:x0 + pad + 2 * R], (x0, y0)


def grid_radius(size: int, R: int, origin: tuple, cx: float, cy: float) -> np.ndarray:
    """Distancia (px originales) del centro de cada celda n×n al centro del círculo."""
    c = (np.arange(size) + 0.5) * (2 * R / size)
    X, Y = np.meshgrid(origin[0] + c, origin[1] + c)
    return np.hypot(X - cx, Y - cy)


def segment(small: np.ndarray, rr: np.ndarray, r_in: float, R: int, margin: float, k_bright: float,
            hi_mult: float, t_median: int) -> tuple:
    from skimage.filters import apply_hysteresis_threshold, sobel, threshold_otsu

    interior = rr <= r_in
    edge_zone = rr <= r_in - margin           # sin bordes pegados a la zona pelúcida
    grad = np.stack([sobel(f) for f in small])

    th_edge = float(threshold_otsu(grad[:, edge_zone]))
    vals = small[:, interior]
    th_bright = float(vals.mean() + k_bright * vals.std())

    states = np.full(small.shape, 2, np.int32)
    states[:, rr > r_in] = 1
    states[:, rr > R] = 0
    edge = np.stack([apply_hysteresis_threshold(g, th_edge, hi_mult * th_edge) for g in grad]) & edge_zone
    if t_median > 1:                          # quita el parpadeo de bordes entre frames
        edge = ndi.median_filter(edge.astype(np.uint8), size=(t_median, 1, 1)).astype(bool)
    states[edge] = 3
    states[(small > th_bright) & interior & ~edge] = 4
    return states, th_edge, th_bright


# ── Figuras ───────────────────────────────────────────────────────────────────

def save_figures(fig_dir: Path, name: str, mean: np.ndarray, sd: np.ndarray, circle: tuple,
                 crop: np.ndarray, states: np.ndarray, runs: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig_dir.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#1f2328", "#59636e", "#d8dee4"
    plt.rcParams.update({
        "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })
    cmap = ListedColormap(STATE_COLORS)
    cx, cy, r_in, R = circle
    T = len(states)
    legend = [Patch(color=c, label=f"{i} {n}") for i, (c, n) in enumerate(zip(STATE_COLORS, STATE_NAMES))]

    # 1. Detección de la zona pelúcida
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in ((axes[0], mean, "Media temporal"), (axes[1], sd, "Desviación estándar temporal")):
        ax.imshow(img, cmap="gray", vmax=np.percentile(img, 99.5))
        ax.add_patch(plt.Circle((cx, cy), r_in, fill=False, color="#facc15", lw=1.2))
        ax.add_patch(plt.Circle((cx, cy), R, fill=False, color="#ef4444", lw=1.2))
        ax.set_title(title, color=ink, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}: r_in = {r_in:.0f} px (amarillo), R = {R} px (rojo)", color=ink)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_circle.png", dpi=110)
    plt.close(fig)

    # 2. Recorte vs. estados en 5 momentos
    idx = np.linspace(0, T - 1, 5).astype(int)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for j, t in enumerate(idx):
        axes[0, j].imshow(crop[t], cmap="gray")
        axes[0, j].set_title(f"t = {t} (RUN{runs[t]})", color=ink, fontsize=9)
        axes[1, j].imshow(states[t], cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("Recorte", color=ink)
    axes[1, 0].set_ylabel(f"Estados {states.shape[1]}×{states.shape[2]}", color=ink)
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(fig_dir / f"{name}_frames.png", dpi=130)
    plt.close(fig)

    # 3. Fracción de cada estado dentro del embrión (estados 2–4) a lo largo del tiempo
    inside = (states >= 2).sum(axis=(1, 2)).astype(float)
    frac = np.stack([(states == s).sum(axis=(1, 2)) / inside for s in (2, 3, 4)])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.stackplot(np.arange(T), frac, colors=STATE_COLORS[2:], edgecolor="white", linewidth=0.2,
                 labels=[f"{s} {STATE_NAMES[s]}" for s in (2, 3, 4)])
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Paso de tiempo (frame)")
    ax.set_ylabel("Fracción del interior")
    ax.set_title("Composición del interior de la zona pelúcida", color=ink, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_states.png", dpi=130)
    plt.close(fig)


def save_gif(path: Path, crop: np.ndarray, states: np.ndarray, scale: int = 2, ms: int = 80):
    """Animación lado a lado: recorte en grises | estados a color."""
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in STATE_COLORS], np.uint8)
    n = states.shape[1] * scale
    frames = []
    for g, s in zip(crop, states):
        left = np.repeat(np.repeat(np.stack([g] * 3, -1), scale, 0), scale, 1)
        right = np.repeat(np.repeat(pal[s], scale, 0), scale, 1)
        frames.append(Image.fromarray(np.concatenate([left, np.full((n, 4, 3), 255, np.uint8), right], 1)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def process(embryo_dir: Path, args) -> dict:
    name = embryo_dir.name
    paths = frame_paths(embryo_dir)
    if not paths:
        raise RuntimeError("sin imágenes *RUN<n>.jpeg")
    stack = load_stack(paths)
    n_total = len(paths)

    cx, cy, r_in = interior_circle(temporal_mean_std(stack)[1])
    (t0, t1), n_invalid = valid_segment(stack, cx, cy, r_in, args.blank_ratio)
    if t1 - t0 < 2:
        raise RuntimeError("menos de 2 frames válidos")
    if (t0, t1) != (0, n_total):
        print(f"{name}: {n_invalid} frames sin embrión; se conserva t = {t0}..{t1 - 1} de {n_total}")
        stack, paths = stack[t0:t1], paths[t0:t1]
    runs = [int(RUN_RE.search(p.name).group(1)) for p in paths]

    mean, sd = temporal_mean_std(stack)
    cx, cy, r_in = interior_circle(sd)
    R = outer_radius(mean, cx, cy, r_in, args.zona_min, args.zona_max)

    crop, origin = crop_square(stack, cx, cy, R)
    small = downsample(crop.astype(np.float32) / 255, args.size, "box")
    rr = grid_radius(args.size, R, origin, cx, cy)
    states, th_edge, th_bright = segment(small, rr, r_in, R, args.margin, args.k_bright,
                                         args.hi_mult, args.t_median)

    fractions = [float((states == s).mean()) for s in range(len(STATE_NAMES))]
    print(f"{name}: T = {len(paths)}  r_in = {r_in:.1f}  R = {R}  centro = ({cx:.1f}, {cy:.1f})  "
          f"umbral borde = {th_edge:.4f}  umbral brillo = {th_bright:.3f}  "
          + "  ".join(f"{s}:{100 * f:.1f}%" for s, f in enumerate(fractions)))

    base = Path(args.out_dir) / name
    base.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{base}.npy", states)
    scipy.io.savemat(f"{base}.mat", {"evolution": states})
    meta = {
        "source": str(embryo_dir.resolve()),
        "files": [str(p.relative_to(embryo_dir)) for p in paths],
        "runs": runs,
        "kept_frames": [t0, t1],
        "frames_total": n_total,
        "frames_invalid": n_invalid,
        "shape": list(states.shape),
        "size": args.size,
        "circle": {"cx": cx, "cy": cy, "r_in": r_in, "R": R},
        "crop_origin": list(origin),
        "crop_side": 2 * R,
        "margin": args.margin,
        "threshold_edge_sobel": th_edge,
        "hysteresis_high_mult": args.hi_mult,
        "temporal_median": args.t_median,
        "threshold_brightness": th_bright,
        "k_bright": args.k_bright,
        "states": STATE_NAMES,
        "state_fraction": fractions,
    }
    with open(f"{base}.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if not args.no_figures:
        crop_small = np.asarray([np.asarray(Image.fromarray(c).resize((200, 200), Image.BILINEAR)) for c in crop])
        save_figures(Path(args.fig_dir), name, mean, sd, (cx, cy, r_in, R), crop_small, states, runs)
    if args.gif:
        crop_gif = np.asarray([np.asarray(Image.fromarray(c).resize((args.size, args.size), Image.BILINEAR))
                               for c in crop])
        save_gif(Path(f"{base}.gif"), crop_gif[::args.gif_step], states[::args.gif_step])
    return meta


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Segmenta secuencias de embriones en evoluciones (T, n, n) de 5 estados."
    )
    parser.add_argument("inputs", nargs="+", help="Carpetas de embriones (ej. CA/embryo/AA83-7)")
    parser.add_argument("--out-dir", default=str(here.parent / "CA" / "embryo_states"),
                        help="Carpeta de salida (default: training/CA/embryo_states)")
    parser.add_argument("--size", type=int, default=100, help="Tamaño n de la malla n×n (default: 100)")
    parser.add_argument("--margin", type=float, default=4,
                        help="Px originales junto a la zona pelúcida sin detección de bordes (default: 4)")
    parser.add_argument("--hi-mult", type=float, default=1.5,
                        help="Histéresis: umbral fuerte = hi-mult × Otsu (default: 1.5)")
    parser.add_argument("--t-median", type=int, default=3,
                        help="Mediana temporal de los bordes en N frames; 1 la desactiva (default: 3)")
    parser.add_argument("--k-bright", type=float, default=1.5,
                        help="Brillo: intensidad > media + k·σ del interior (default: 1.5)")
    parser.add_argument("--zona-min", type=int, default=10, help="Grosor mínimo de la zona en px (default: 10)")
    parser.add_argument("--zona-max", type=int, default=35, help="Grosor máximo de la zona en px (default: 35)")
    parser.add_argument("--fig-dir", default=str(here / "docs" / "embryo"),
                        help="Carpeta para las figuras (default: training/preprocessing/docs/embryo)")
    parser.add_argument("--no-figures", action="store_true", help="No generar figuras")
    parser.add_argument("--blank-ratio", type=float, default=0.3,
                        help="Frame inválido si su textura interior < ratio × p90 de la secuencia (default: 0.3)")
    parser.add_argument("--gif", action="store_true",
                        help="Guardar también <out-dir>/<id>.gif (recorte | estados)")
    parser.add_argument("--gif-step", type=int, default=2, help="Usar 1 de cada N frames en el GIF (default: 2)")
    args = parser.parse_args()

    failed = []
    for d in map(Path, args.inputs):
        try:
            process(d, args)
        except Exception as e:  # un embrión defectuoso no detiene el lote
            print(f"{d.name}: ERROR {e}", file=sys.stderr)
            failed.append(d.name)
    print(f"Procesados: {len(args.inputs) - len(failed)}  Fallidos: {len(failed)}"
          + (f"  ({', '.join(failed)})" if failed else ""))
    print(f"Salida: {args.out_dir}/<id>.npy|.mat|.json")


if __name__ == "__main__":
    main()
