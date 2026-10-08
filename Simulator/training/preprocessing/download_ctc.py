# training/preprocessing/download_ctc.py
#
# Descarga los datasets 2D+t de entrenamiento del Cell Tracking Challenge
# (https://celltrackingchallenge.net/2d-datasets/) y los descomprime en Dataset/CTC/<nombre>/.
#
# Cada dataset trae dos secuencias (01/, 02/) de TIFF t000.tif, t001.tif, … y sus anotaciones:
#   <seq>_GT/SEG  segmentación de referencia (solo algunos frames)
#   <seq>_GT/TRA  marcadores de seguimiento (todos los frames)
#   <seq>_ST/SEG  segmentación "silver truth" (todos los frames; no en todos los datasets)
#
# Uso:
#   python preprocessing/download_ctc.py                 # los 8 datasets ligeros (~635 MB)
#   python preprocessing/download_ctc.py DIC-C2DH-HeLa   # solo los indicados
#   python preprocessing/download_ctc.py --all           # incluye BF-C2DL-HSC y BF-C2DL-MuSC (+2.8 GB)

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://data.celltrackingchallenge.net/training-datasets/{}.zip"

LIGHT = [
    "DIC-C2DH-HeLa",      # 37 MB, HeLa en vidrio (DIC)
    "Fluo-C2DL-Huh7",     # 36 MB, hepatocarcinoma humano
    "Fluo-C2DL-MSC",      # 72 MB, células madre mesenquimales de rata
    "Fluo-N2DH-GOWT1",    # 53 MB, células madre de ratón GFP-GOWT1
    "Fluo-N2DL-HeLa",     # 182 MB, HeLa con H2b-GFP
    "PhC-C2DH-U373",      # 40 MB, glioblastoma-astrocitoma U373
    "PhC-C2DL-PSC",       # 124 MB, células madre pancreáticas
    "Fluo-N2DH-SIM+",     # 91 MB, núcleos HL60 simulados (segmentación perfecta)
]
HEAVY = [
    "BF-C2DL-HSC",        # 1.6 GB, células madre hematopoyéticas de ratón
    "BF-C2DL-MuSC",       # 1.2 GB, células madre musculares de ratón
]


def download(name: str, dest: Path):
    out_dir = dest / name
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{name}: ya existe, se omite")
        return
    zip_path = dest / f"{name}.zip"
    url = BASE_URL.format(urllib.request.quote(name))
    print(f"{name}: descargando {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)

    tmp = dest / f".{name}.tmp"
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():                     # evita rutas fuera de la carpeta destino
            if m.startswith("/") or ".." in Path(m).parts:
                raise RuntimeError(f"ruta sospechosa en el zip: {m}")
        z.extractall(tmp)
    # El zip trae una carpeta raíz con el nombre del dataset
    inner = tmp / name if (tmp / name).is_dir() else tmp
    shutil.move(str(inner), str(out_dir))
    shutil.rmtree(tmp, ignore_errors=True)
    zip_path.unlink()
    print(f"{name}: listo en {out_dir}")


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Descarga datasets 2D del Cell Tracking Challenge")
    p.add_argument("names", nargs="*", help="Datasets a descargar (default: los 8 ligeros)")
    p.add_argument("--all", action="store_true", help="Incluir también los datasets BF (2.8 GB)")
    p.add_argument("--dest", default=str(here.parent / "Dataset" / "CTC"),
                   help="Carpeta destino (default: training/Dataset/CTC)")
    args = p.parse_args()

    names = args.names or (LIGHT + HEAVY if args.all else LIGHT)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for n in names:
        try:
            download(n, dest)
        except Exception as e:
            print(f"{n}: ERROR {e}", file=sys.stderr)
            failed.append(n)
    print(f"Descargados: {len(names) - len(failed)}  Fallidos: {len(failed)}"
          + (f"  ({', '.join(failed)})" if failed else ""))


if __name__ == "__main__":
    main()
