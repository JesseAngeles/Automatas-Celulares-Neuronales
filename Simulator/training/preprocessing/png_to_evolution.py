# training/preprocessing/png_to_evolution.py
#
# Convierte una carpeta de PNGs segmentados al formato del simulador: (T, H, W) int32
#
# Uso mínimo:
#   python png_to_evolution.py ../Dataset/raw_2_out
#
# Con orden de colores explícito (estado 0 = negro, estado 1 = blanco, ...):
#   python png_to_evolution.py ../Dataset/raw_2_out --colors "#000000,#ffffff,#ff0000"
#
# Con salida en otra ruta:
#   python png_to_evolution.py ../Dataset/raw_2_out mi_evolucion

import argparse
import sys
import numpy as np
import scipy.io
from pathlib import Path


def is_color_image(path: Path) -> bool:
    from PIL import Image
    img = Image.open(path)
    return img.mode in ("RGB", "RGBA")


def load_grayscale(paths: list) -> np.ndarray:
    from PIL import Image
    frames, ref_shape = [], None
    for p in paths:
        arr = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            print(f"Error: {p.name} tiene forma {arr.shape}, se esperaba {ref_shape}", file=sys.stderr)
            sys.exit(1)
        frames.append(arr)
    return np.stack(frames)


def load_color(paths: list, color_order: list | None) -> np.ndarray:
    from PIL import Image
    raw_frames, ref_shape = [], None
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            print(f"Error: {p.name} tiene forma {arr.shape}, se esperaba {ref_shape}", file=sys.stderr)
            sys.exit(1)
        raw_frames.append(arr)

    if color_order is None:
        pixels = np.concatenate([f.reshape(-1, 3) for f in raw_frames], axis=0)
        unique = np.unique(pixels, axis=0)
        color_order = [tuple(c) for c in unique.tolist()]

    print(f"Colores → estados ({len(color_order)}):")
    for i, c in enumerate(color_order):
        print(f"  Estado {i}: RGB{c}  #{c[0]:02x}{c[1]:02x}{c[2]:02x}")

    color_index = {c: i for i, c in enumerate(color_order)}
    H, W = ref_shape[:2]
    frames = []
    for raw in raw_frames:
        label = np.zeros((H, W), dtype=np.uint8)
        for color, idx in color_index.items():
            mask = np.all(raw == np.array(color, dtype=np.uint8), axis=-1)
            label[mask] = idx
        frames.append(label)

    return np.stack(frames)


def parse_colors(s: str) -> list:
    colors = []
    for token in s.split(","):
        token = token.strip().lstrip("#")
        if len(token) != 6:
            print(f"Color inválido: '{token}' (se esperaba rrggbb)", file=sys.stderr)
            sys.exit(1)
        colors.append((int(token[0:2], 16), int(token[2:4], 16), int(token[4:6], 16)))
    return colors


def main():
    parser = argparse.ArgumentParser(
        description="Convierte PNGs segmentados al formato de evolución del simulador."
    )
    parser.add_argument("input", help="Carpeta con las imágenes PNG")
    parser.add_argument("output", nargs="?", default=None,
                        help="Ruta de salida sin extensión (default: nombre de la carpeta)")
    parser.add_argument("--pattern", default="*.png",
                        help="Patrón glob (default: *.png)")
    parser.add_argument("--colors", default=None,
                        help="Orden explícito de colores: '#000000,#ffffff,...'")
    args = parser.parse_args()

    input_path = Path(args.input)
    evolutions_dir = Path(__file__).resolve().parent.parent / "CA"
    output_base = args.output if args.output else str(evolutions_dir / input_path.resolve().name)

    paths = sorted(input_path.glob(args.pattern))
    if not paths:
        print(f"No se encontraron archivos '{args.pattern}' en '{args.input}'", file=sys.stderr)
        sys.exit(1)

    print(f"Archivos encontrados: {len(paths)}")

    color_order = parse_colors(args.colors) if args.colors else None

    # Auto-detectar si las imágenes son color o escala de grises
    if color_order is not None or is_color_image(paths[0]):
        arr = load_color(paths, color_order)
    else:
        print("Imágenes en escala de grises — valor de pixel = índice de estado")
        arr = load_grayscale(paths)

    print(f"Shape: {arr.shape}  estados: {int(arr.max()) + 1}")

    base = output_base.removesuffix(".npy").removesuffix(".mat")
    np.save(base + ".npy", arr.astype(np.int32))
    scipy.io.savemat(base + ".mat", {"evolution": arr.astype(np.int32)})
    print(f"Guardado: {base}.npy")
    print(f"Guardado: {base}.mat")


if __name__ == "__main__":
    main()
