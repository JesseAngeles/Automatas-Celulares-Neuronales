# training/celsius_to_evolution.py
#
# Limpieza de imágenes térmicas radiométricas (FLAME 3, geo_thermal_tiff_celsius) al formato
# del simulador: (T, n, n) int32
#
#   1. Lee los GeoTIFF en °C (todos comparten la misma malla georreferenciada).
#   2. Marca 0 °C como "sin dato" y recorta el rectángulo más grande válido en TODOS los frames.
#   3. Reduce cada frame a n×n con un kernel de convolución separable (box / tent / gaussian).
#   4. Agrupa las temperaturas de toda la simulación con K-Means 1D; k por método del codo
#      (y silueta como verificación) o fijado con --k.
#   5. Ordena los clusters de frío a caliente y asigna a cada celda el índice de su cluster.
#
# Uso mínimo:
#   python preprocessing/celsius_to_evolution.py Dataset/plot_1/geo_thermal_tiff_celsius CA/FLAME3/flame3_plot1_celsius_100
#
# Ver README_thermal_to_evolution.md para la descripción completa del proceso.

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io

from thermal_to_evolution import EXIF_DATETIME, elbow_k, scan_k

KERNELS = ("box", "tent", "gaussian")


def load_stack(paths: list) -> np.ndarray:
    import rasterio

    frames, ref = [], None
    for p in paths:
        with rasterio.open(p) as r:
            if ref is None:
                ref = (r.shape, r.transform)
            elif (r.shape, r.transform) != ref:
                print(f"Error: {p.name} no comparte la malla del primer frame", file=sys.stderr)
                sys.exit(1)
            frames.append(r.read(1))
    return np.stack(frames)


def load_timestamps(paths: list, jpg_dir: Path) -> list:
    """Los GeoTIFF no traen fecha; se toma del JPG original con el mismo número (IRX_XXXX)."""
    from PIL import Image

    stamps = []
    for p in paths:
        jpg = jpg_dir / (p.name.split("_ref")[0] + ".jpg")
        raw = Image.open(jpg).getexif().get(EXIF_DATETIME) if jpg.exists() else None
        stamps.append(datetime.strptime(raw, "%Y:%m:%d %H:%M:%S") if raw else None)
    return stamps


def largest_rectangle(mask: np.ndarray) -> tuple:
    """Rectángulo alineado a los ejes de mayor área con todo True. Regresa (r0, r1, c0, c1)."""
    H, W = mask.shape
    heights = np.zeros(W, dtype=int)
    best_area, best = 0, (0, 0, 0, 0)
    for r in range(H):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []  # (columna inicial, altura), alturas crecientes
        for c in range(W + 1):
            h = heights[c] if c < W else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                if sh * (c - s) > best_area:
                    best_area, best = sh * (c - s), (r - sh + 1, r + 1, s, c)
                start = s
            stack.append((start, h))
    return best


def kernel_matrix(L: int, N: int, kind: str) -> np.ndarray:
    """
    Matriz (N, L) que convoluciona un eje de longitud L con el kernel y lo muestrea en N
    centros equiespaciados (paso s = L/N). Cada fila suma 1.
    """
    s = L / N
    centers = (np.arange(N) + 0.5) * s - 0.5
    j = np.arange(L)[None, :]
    d = j - centers[:, None]
    if kind == "box":
        # Promedio por área: traslape exacto de cada píxel con la ventana [c - s/2, c + s/2]
        w = np.minimum(j + 0.5, centers[:, None] + s / 2) - np.maximum(j - 0.5, centers[:, None] - s / 2)
        w = np.clip(w, 0, None)
    elif kind == "tent":
        w = np.clip(1 - np.abs(d) / s, 0, None)
    elif kind == "gaussian":
        sigma = s / 2
        w = np.exp(-d ** 2 / (2 * sigma ** 2)) * (np.abs(d) <= 3 * sigma)
    else:
        raise ValueError(kind)
    return w / w.sum(axis=1, keepdims=True)


def downsample(stack: np.ndarray, size: int, kind: str) -> np.ndarray:
    """Convolución separable + muestreo: Y_t = K_r · X_t · K_cᵀ para cada frame."""
    Kr = kernel_matrix(stack.shape[1], size, kind)
    Kc = kernel_matrix(stack.shape[2], size, kind)
    return (Kr @ stack.astype(np.float64) @ Kc.T).astype(np.float32)


def save_figures(fig_dir: Path, name: str, crop: np.ndarray, temps: np.ndarray, states: np.ndarray,
                 centers: np.ndarray, thresholds: np.ndarray, scan: dict | None, k: int, kernel: str,
                 size: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig_dir.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#1f2328", "#59636e", "#d8dee4"
    plt.rcParams.update({
        "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": grid, "grid.linewidth": 0.6, "font.size": 10,
    })
    # Estados ordenados = magnitud → rampa secuencial
    state_colors = plt.get_cmap("inferno")(np.linspace(0.12, 0.92, k))
    state_cmap = ListedColormap(state_colors)
    vmin, vmax = 20, float(np.percentile(temps, 99.9))
    T = len(temps)

    # 1. Codo y silueta
    if scan is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, key, title in ((a1, "inertia", "Inercia (método del codo)"),
                               (a2, "silhouette", "Coeficiente de silueta")):
            ax.plot(scan["k"], scan[key], color="#3b6ea5", lw=2, marker="o", ms=5)
            i = scan["k"].index(k)
            ax.plot(scan["k"][i], scan[key][i], "o", ms=10, mfc="none", mec="#c2410c", mew=2)
            ax.annotate(f"k = {k}", (scan["k"][i], scan[key][i]), textcoords="offset points",
                        xytext=(8, 8), color=ink)
            ax.set_title(title, color=ink, loc="left")
            ax.set_xlabel("k (número de clusters)")
            ax.set_xticks(scan["k"])
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}_k.png", dpi=150)
        plt.close(fig)

    # 2. Comparación de kernels en el frame con mayor temperatura
    t = int(np.argmax(crop.reshape(T, -1).max(axis=1)))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    panels = [(f"Recorte original {crop.shape[1]}×{crop.shape[2]}", crop[t])]
    panels += [(f"{kind} → {size}×{size}", downsample(crop[t:t + 1], size, kind)[0]) for kind in KERNELS]
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title + (f"\nmáx {img.max():.0f} °C"), color=ink, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="°C")
    fig.savefig(fig_dir / f"{name}_kernels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Histograma de temperaturas con los umbrales entre estados
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.hist(temps.ravel(), bins=200, color="#3b6ea5", log=True)
    for i, th in enumerate(thresholds):
        ax.axvline(th, color="#c2410c", lw=1.2, ls="--")
        ax.annotate(f"{th:.0f} °C", (th, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", color=ink, fontsize=8)
    ax.set_xlabel("Temperatura de celda (°C)")
    ax.set_ylabel("Celdas (escala log)")
    ax.set_title("Distribución de temperaturas y umbrales entre estados", color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_hist.png", dpi=150)
    plt.close(fig)

    # 4. Antes / después en 5 momentos
    idx = np.linspace(0, T - 1, 5).astype(int)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.4))
    for j, t in enumerate(idx):
        im0 = axes[0, j].imshow(temps[t], cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[0, j].set_title(f"t = {t}", color=ink)
        im1 = axes[1, j].imshow(states[t], cmap=state_cmap, vmin=-0.5, vmax=k - 0.5, interpolation="nearest")
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            axes[r, j].grid(False)
    axes[0, 0].set_ylabel(f"°C ({kernel} {size}×{size})", color=ink)
    axes[1, 0].set_ylabel(f"Estados (k = {k})", color=ink)
    fig.colorbar(im0, ax=axes[0, :], shrink=0.9, label="°C")
    cb = fig.colorbar(im1, ax=axes[1, :], shrink=0.9, ticks=range(k), label="Estado")
    fig.savefig(fig_dir / f"{name}_frames.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Proporción de cada estado en el tiempo
    frac = np.stack([(states == s).mean(axis=(1, 2)) for s in range(k)])
    edges = np.concatenate([[-np.inf], thresholds, [np.inf]])
    labels = []
    for s in range(k):
        lo, hi = edges[s], edges[s + 1]
        rng = f"< {hi:.0f} °C" if s == 0 else (f"≥ {lo:.0f} °C" if s == k - 1 else f"{lo:.0f}–{hi:.0f} °C")
        labels.append(f"Estado {s} ({rng})")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.stackplot(np.arange(T), frac, colors=state_colors, labels=labels, edgecolor="white", linewidth=0.3)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Paso de tiempo (frame)")
    ax.set_ylabel("Fracción de celdas")
    ax.set_title("Distribución de estados a lo largo de la simulación", color=ink, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_states.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Convierte GeoTIFF térmicos en °C en una evolución (T, n, n) por clusters de temperatura."
    )
    parser.add_argument("input", help="Carpeta con los GeoTIFF en °C")
    parser.add_argument("output", help="Ruta de salida sin extensión")
    parser.add_argument("--pattern", default="*.TIFF", help="Patrón glob (default: *.TIFF)")
    parser.add_argument("--size", type=int, default=100, help="Tamaño n de la malla n×n (default: 100)")
    parser.add_argument("--kernel", choices=KERNELS, default="gaussian",
                        help="Kernel de convolución para reducir resolución (default: gaussian)")
    parser.add_argument("--nodata", type=float, default=0.0,
                        help="Valores <= a este se consideran sin dato (default: 0)")
    parser.add_argument("--k", type=int, default=None, help="Número de clusters; si se omite se elige con el codo")
    parser.add_argument("--k-min", type=int, default=2, help="k mínimo a evaluar (default: 2)")
    parser.add_argument("--k-max", type=int, default=10, help="k máximo a evaluar (default: 10)")
    parser.add_argument("--sample", type=int, default=200_000,
                        help="Celdas muestreadas para ajustar K-Means (default: 200000)")
    parser.add_argument("--sil-size", type=int, default=10_000,
                        help="Celdas usadas para calcular la silueta (default: 10000)")
    parser.add_argument("--seed", type=int, default=0, help="Semilla (default: 0)")
    parser.add_argument("--jpg-dir", default=None,
                        help="Carpeta con los JPG originales para leer timestamps (default: ../raw_thermal_jpg)")
    parser.add_argument("--fig-dir", default=str(Path(__file__).parent / "docs" / "thermal"),
                        help="Carpeta para las figuras (default: training/preprocessing/docs/thermal)")
    args = parser.parse_args()

    from sklearn.cluster import KMeans

    input_path = Path(args.input)
    paths = sorted(input_path.glob(args.pattern))
    if not paths:
        print(f"No se encontraron archivos '{args.pattern}' en '{args.input}'", file=sys.stderr)
        sys.exit(1)

    # ── 1. Lectura ──────────────────────────────────────────────────────────
    print(f"Archivos encontrados: {len(paths)}")
    stack = load_stack(paths)
    T = len(stack)
    print(f"Malla común: {stack.shape[1]}×{stack.shape[2]}")

    jpg_dir = Path(args.jpg_dir) if args.jpg_dir else input_path.parent / "raw_thermal_jpg"
    stamps = load_timestamps(paths, jpg_dir)
    if all(stamps):
        dt = np.diff([s.timestamp() for s in stamps])
        print(f"Intervalo entre frames: mediana {np.median(dt):.0f} s, min {dt.min():.0f} s, max {dt.max():.0f} s")

    # ── 2. Recorte de la región válida en todos los frames ──────────────────
    valid = (stack > args.nodata).all(axis=0)
    r0, r1, c0, c1 = largest_rectangle(valid)
    crop = stack[:, r0:r1, c0:c1]
    print(f"Recorte válido: filas {r0}:{r1}, columnas {c0}:{c1}  →  {r1 - r0}×{c1 - c0}")

    # ── 3. Convolución + muestreo a n×n ─────────────────────────────────────
    temps = downsample(crop, args.size, args.kernel)
    print(f"Kernel {args.kernel} → {temps.shape}  °C: min {temps.min():.1f}, mediana {np.median(temps):.1f}, "
          f"máx {temps.max():.1f}")

    # ── 4. Clusters de temperatura sobre toda la simulación ─────────────────
    values = temps.reshape(-1, 1)
    rng = np.random.default_rng(args.seed)
    sample = values[rng.choice(len(values), size=min(args.sample, len(values)), replace=False)]

    scan = None
    if args.k is None:
        print(f"Evaluando k = {args.k_min}..{args.k_max} sobre {len(sample)} celdas:")
        scan = scan_k(sample, args.k_min, args.k_max, args.seed, args.sil_size)
        k = elbow_k(scan["k"], scan["inertia"])
        best_sil = scan["k"][int(np.argmax(scan["silhouette"]))]
        print(f"Codo → k = {k}   (máxima silueta en k = {best_sil})")
    else:
        k = args.k
        print(f"k fijado por el usuario: {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(sample)

    # ── 5. Ordenar frío → caliente y etiquetar ──────────────────────────────
    centers = np.sort(km.cluster_centers_.ravel())
    # En 1D el centroide más cercano equivale a umbrales en los puntos medios
    thresholds = (centers[:-1] + centers[1:]) / 2
    states = np.digitize(temps, thresholds).astype(np.int32)

    fractions = [(states == s).mean() for s in range(k)]
    edges = np.concatenate([[-np.inf], thresholds, [np.inf]])
    print(f"Temperaturas → estados ({k}):")
    for s in range(k):
        print(f"  Estado {s}: centro {centers[s]:6.1f} °C  rango [{edges[s]:6.1f}, {edges[s + 1]:6.1f})  "
              f"{100 * fractions[s]:5.1f}%")

    # ── Guardado ────────────────────────────────────────────────────────────
    base = args.output.removesuffix(".npy").removesuffix(".mat")
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    np.save(base + ".npy", states)
    np.save(base + "_celsius.npy", temps)
    scipy.io.savemat(base + ".mat", {"evolution": states})

    meta = {
        "source": str(input_path.resolve()),
        "files": [p.name for p in paths],
        "timestamps": [s.isoformat() if s else None for s in stamps],
        "shape": list(states.shape),
        "size": args.size,
        "crop_rows": [int(r0), int(r1)],
        "crop_cols": [int(c0), int(c1)],
        "kernel": args.kernel,
        "nodata": args.nodata,
        "k": k,
        "k_selection": "elbow" if args.k is None else "user",
        "centers_celsius": centers.tolist(),
        "thresholds_celsius": thresholds.tolist(),
        "state_fraction": [float(f) for f in fractions],
        "scan": scan,
        "sample": len(sample),
        "seed": args.seed,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    name = Path(base).name
    save_figures(Path(args.fig_dir), name, crop, temps, states, centers, thresholds, scan, k,
                 args.kernel, args.size)

    print(f"Shape: {states.shape}  estados: {k}")
    print(f"Guardado: {base}.npy")
    print(f"Guardado: {base}.mat")
    print(f"Guardado: {base}_celsius.npy  (temperaturas {args.size}×{args.size} antes de agrupar)")
    print(f"Guardado: {base}.json")
    print(f"Figuras:  {args.fig_dir}/{name}_*.png")


if __name__ == "__main__":
    main()
