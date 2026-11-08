"""Entrena alumnas con scripts/main.py (estados combinados, one-hot 24, frontera abierta).

    python entrenar.py <nombre> <capas: 16x128> [épocas=30] [semilla=42] [datos=datos] [schedule=exp]
→ redes/<nombre>/red.npz
"""
import subprocess
import sys

import comun as C

PY = sys.executable
MAIN = C.TRAINING / "scripts" / "main.py"


def entrenar(nombre, capas, epocas=30, semilla=42, datos="datos", schedule="exp"):
    n, ancho = map(int, capas.split("x"))
    out = C.CARPETA / "redes" / nombre
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", str(MAIN), "--data", *map(str, sorted((C.CARPETA / datos).glob("*.npy"))),
           "--output", str(out / "red.npz"), "--neighbourhood", (C.CARPETA / "vecindad.txt").read_text(),
           "--layers", " ".join([f"{ancho}:relu"] * n), "--out-size", str(C.NE), "--out-act", "linear",
           "--one-hot", str(C.NE), "--epochs", str(epocas), "--batch", "8192", "--lr", "1e-3",
           "--lr-schedule", schedule, "--lr-min", "1e-5", "--loss-target", "1e-9", "--print-every", "1",
           "--seed", str(semilla), "--open-boundary", "--log-csv", str(out / "perdida.csv")]
    with open(out / "entrenamiento.log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=C.TRAINING)
    if r.returncode:
        raise RuntimeError(f"falló main.py, ver {out / 'entrenamiento.log'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    entrenar(a[0], a[1], *(int(x) for x in a[2:4]), *a[4:6])
