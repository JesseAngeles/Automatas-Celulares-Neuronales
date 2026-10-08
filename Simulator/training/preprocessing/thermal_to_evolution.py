# training/thermal_to_evolution.py
#
# Limpieza de imágenes térmicas (FLAME 3, vista nadir) al formato del simulador: (T, n, n) int32
#
#   1. Escala cada frame a n×n (estirando, sin conservar aspecto).
#   2. Agrupa los colores de TODOS los frames de la simulación con K-Means; k se elige
#      con el método del codo (y se reporta el coeficiente de silueta) o se fija con --k.
#   3. Ordena los clusters de frío a caliente (luminancia del centroide) y asigna a cada
#      píxel el índice de su cluster → estado del autómata.
#
# Uso mínimo:
#   python preprocessing/thermal_to_evolution.py Dataset/plot_1/raw_thermal_jpg CA/FLAME3/flame3_plot1_100
#
# Con k fijo:
#   python preprocessing/thermal_to_evolution.py Dataset/plot_1/raw_thermal_jpg CA/FLAME3/flame3_plot1_100 --k 4
#
# Ver README_thermal_to_evolution.md para la descripción completa del proceso.

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

EXIF_DATETIME = 306


def load_frames(paths: list, size: int) -> tuple[np.ndarray, list]:
    """Carga y escala cada frame a size×size. Regresa (T, size, size, 3) uint8 y timestamps."""
    frames, stamps = [], []
    for p in paths:
        img = Image.open(p)
        raw = img.getexif().get(EXIF_DATETIME)
        stamps.append(datetime.strptime(raw, "%Y:%m:%d %H:%M:%S") if raw else None)
        # BOX = promedio por área: el filtro adecuado para reducir resolución sin aliasing
        small = img.convert("RGB").resize((size, size), Image.Resampling.BOX)
        frames.append(np.asarray(small, dtype=np.uint8))
    return np.stack(frames), stamps


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def elbow_k(ks: list, inertias: list) -> int:
    """Kneedle: k con la mayor distancia a la recta que une los extremos de la curva normalizada."""
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inertias) - inertias[-1]) / (inertias[0] - inertias[-1])
    # La curva de inercia es decreciente y convexa: la recta va de (0,1) a (1,0)
    dist = (1 - x) - y
    return ks[int(np.argmax(dist))]


def scan_k(sample: np.ndarray, k_min: int, k_max: int, seed: int, sil_size: int) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    ks, inertias, silhouettes = [], [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(sample)
        sil = silhouette_score(sample, km.labels_, sample_size=min(sil_size, len(sample)), random_state=seed)
        ks.append(k)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(sil))
        print(f"  k={k:2d}  inercia={km.inertia_:.4g}  silueta={sil:.4f}")
    return {"k": ks, "inertia": inertias, "silhouette": silhouettes}


def save_figures(fig_dir: Path, name: str, frames: np.ndarray, states: np.ndarray,
                 centroids: np.ndarray, scan: dict | None, k: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#1f2328", "#59636e", "#d8dee4"
    plt.rcParams.update({
        "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": grid, "grid.linewidth": 0.6, "font.size": 10,
    })
    hexes = ["#%02x%02x%02x" % tuple(c) for c in centroids]

    # 1. Codo y silueta: dos paneles, un eje cada uno (no doble eje)
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

    # 2. Paleta de estados: el color del centroide ES la identidad del estado
    fig, ax = plt.subplots(figsize=(1.2 * k + 1, 1.4))
    for i, h in enumerate(hexes):
        ax.add_patch(plt.Rectangle((i, 0), 0.94, 1, color=h))
        ax.text(i + 0.47, -0.25, f"{i}\n{h}", ha="center", va="top", fontsize=8, color=ink)
    ax.set_xlim(0, k)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    fig.savefig(fig_dir / f"{name}_palette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Antes / después en 5 momentos de la simulación
    T = len(frames)
    idx = np.linspace(0, T - 1, 5).astype(int)
    lut = centroids.astype(np.uint8)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for j, t in enumerate(idx):
        axes[0, j].imshow(frames[t], interpolation="nearest")
        axes[0, j].set_title(f"t = {t}", color=ink)
        axes[1, j].imshow(lut[states[t]], interpolation="nearest")
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            axes[r, j].grid(False)
    axes[0, 0].set_ylabel("Escalada", color=ink)
    axes[1, 0].set_ylabel(f"Estados (k = {k})", color=ink)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_frames.png", dpi=150)
    plt.close(fig)

    # 4. Proporción de cada estado en el tiempo (área apilada, colores = centroides)
    frac = np.stack([(states == s).mean(axis=(1, 2)) for s in range(k)])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.stackplot(np.arange(T), frac, colors=hexes, labels=[f"Estado {s}" for s in range(k)],
                 edgecolor="white", linewidth=0.3)
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
        description="Convierte imágenes térmicas JPG en una evolución (T, n, n) por clusters de color."
    )
    parser.add_argument("input", help="Carpeta con las imágenes térmicas")
    parser.add_argument("output", help="Ruta de salida sin extensión (se generan .npy, .mat y .json)")
    parser.add_argument("--pattern", default="*.jpg", help="Patrón glob (default: *.jpg)")
    parser.add_argument("--size", type=int, default=100, help="Tamaño n de la malla n×n (default: 100)")
    parser.add_argument("--k", type=int, default=None, help="Número de clusters; si se omite se elige con el codo")
    parser.add_argument("--k-min", type=int, default=2, help="k mínimo a evaluar (default: 2)")
    parser.add_argument("--k-max", type=int, default=10, help="k máximo a evaluar (default: 10)")
    parser.add_argument("--sample", type=int, default=200_000,
                        help="Píxeles muestreados para ajustar K-Means (default: 200000)")
    parser.add_argument("--sil-size", type=int, default=10_000,
                        help="Píxeles usados para calcular la silueta (default: 10000)")
    parser.add_argument("--seed", type=int, default=0, help="Semilla (default: 0)")
    parser.add_argument("--fig-dir", default=str(Path(__file__).parent / "docs" / "thermal"),
                        help="Carpeta para las figuras (default: training/preprocessing/docs/thermal)")
    args = parser.parse_args()

    from sklearn.cluster import KMeans

    paths = sorted(Path(args.input).glob(args.pattern))
    if not paths:
        print(f"No se encontraron archivos '{args.pattern}' en '{args.input}'", file=sys.stderr)
        sys.exit(1)

    # ── 1. Escalado ──────────────────────────────────────────────────────────
    print(f"Archivos encontrados: {len(paths)}  →  escalando a {args.size}×{args.size}")
    frames, stamps = load_frames(paths, args.size)
    T = len(frames)

    if all(stamps):
        if any(b < a for a, b in zip(stamps, stamps[1:])):
            print("Aviso: el orden por nombre no coincide con el orden temporal EXIF", file=sys.stderr)
        dt = np.diff([s.timestamp() for s in stamps])
        print(f"Intervalo entre frames: mediana {np.median(dt):.0f} s, min {dt.min():.0f} s, max {dt.max():.0f} s")

    # ── 2. Clusters de color sobre toda la simulación ───────────────────────
    pixels = frames.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    sample = pixels[rng.choice(len(pixels), size=min(args.sample, len(pixels)), replace=False)]

    scan = None
    if args.k is None:
        print(f"Evaluando k = {args.k_min}..{args.k_max} sobre {len(sample)} píxeles:")
        scan = scan_k(sample, args.k_min, args.k_max, args.seed, args.sil_size)
        k = elbow_k(scan["k"], scan["inertia"])
        best_sil = scan["k"][int(np.argmax(scan["silhouette"]))]
        print(f"Codo → k = {k}   (máxima silueta en k = {best_sil})")
    else:
        k = args.k
        print(f"k fijado por el usuario: {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(sample)

    # ── 3. Ordenar frío → caliente y etiquetar ──────────────────────────────
    order = np.argsort(luminance(km.cluster_centers_))
    remap = np.empty(k, dtype=np.int32)
    remap[order] = np.arange(k)
    centroids = np.clip(np.rint(km.cluster_centers_[order]), 0, 255).astype(int)

    states = remap[km.predict(pixels)].reshape(T, args.size, args.size).astype(np.int32)

    fractions = [(states == s).mean() for s in range(k)]
    print(f"Colores → estados ({k}):")
    for i, c in enumerate(centroids):
        print(f"  Estado {i}: RGB{tuple(c.tolist())}  #{c[0]:02x}{c[1]:02x}{c[2]:02x}  {100 * fractions[i]:5.1f}%")

    # ── Guardado ────────────────────────────────────────────────────────────
    base = args.output.removesuffix(".npy").removesuffix(".mat")
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    np.save(base + ".npy", states)
    scipy.io.savemat(base + ".mat", {"evolution": states})

    meta = {
        "source": str(Path(args.input).resolve()),
        "files": [p.name for p in paths],
        "timestamps": [s.isoformat() if s else None for s in stamps],
        "shape": list(states.shape),
        "size": args.size,
        "resample": "BOX (stretch, sin conservar aspecto)",
        "k": k,
        "k_selection": "elbow" if args.k is None else "user",
        "centroids_rgb": centroids.tolist(),
        "centroids_hex": ["#%02x%02x%02x" % tuple(c) for c in centroids],
        "state_fraction": [float(f) for f in fractions],
        "scan": scan,
        "sample": len(sample),
        "seed": args.seed,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    name = Path(base).name
    save_figures(Path(args.fig_dir), name, frames, states, centroids, scan, k)

    print(f"Shape: {states.shape}  estados: {k}")
    print(f"Guardado: {base}.npy")
    print(f"Guardado: {base}.mat")
    print(f"Guardado: {base}.json")
    print(f"Figuras:  {args.fig_dir}/{name}_*.png")


if __name__ == "__main__":
    main()
