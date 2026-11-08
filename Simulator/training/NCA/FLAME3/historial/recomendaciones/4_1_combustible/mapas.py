"""4.1 / 4.2 — Mapas del terreno sobre la malla térmica de cada escenario.

Lee de Dataset/FLAME3/hanna_hammock_extra/ (bajado de Kaggle, brycehopkins/flame-3-nadir-thermal-plot-subset) el ortomosaico
RGB antes y después de la quema, el DTM y la nube de puntos antes de la quema (plot 1 → p1, plot 2 → p3; todo en
EPSG:26916, igual que los GeoTIFF térmicos) y los lleva a la malla de 100×100 y 200×200 de
Dataset/FLAME3/seleccion1/flame_plot_N/geo_thermal_tiff_celsius_aligned_<n> (el transform del primer frame, que es la referencia de la
alineación) promediando lo que cae en cada celda.

Por celda:
    brillo_pre, brillo_post   (R + G + B) / 3 del ortomosaico (0–255)
    exg_pre, exg_post         verdor: 2g − r − b con r, g, b = R, G, B / (R + G + B)
    altura                    altura de la vegetación: percentil 95 de la Z de los puntos de la celda menos el DTM
    dtm                       elevación del terreno (m)
    cobertura                 fracción de la celda con dato en el ortomosaico (canal alfa)
Además mide si la térmica y el ortomosaico coinciden: busca el desplazamiento (en celdas) que maximiza la
correlación entre el oscurecimiento (brillo_pre − brillo_post) y la temperatura máxima del escenario completo.

Escribe datos/<escenario>_<n>.npz, alineacion.csv, mapas_<escenario>.png y registro.log.

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/4_1_combustible/mapas.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent))
import comun_r as cr                                                      # noqa: E402

EXTRA = cr.TRAINING / "Dataset" / "FLAME3" / "hanna_hammock_extra"
PLOT = {"p1": 1, "p3": 2}
DESPLAZ = 6                                     # búsqueda de ±6 celdas


def malla(k: str, lado: int):
    archivo = json.load(open(cr.TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo.json"))["files"][0]
    with rasterio.open(cr.TRAINING / f"Dataset/FLAME3/seleccion1/flame_plot_{k[1]}/geo_thermal_tiff_celsius_aligned_{lado}/{archivo}") as r:
        return r.transform, r.crs, (r.height, r.width)


def a_malla(ruta: Path, bandas, transform, crs, forma, nodata=None) -> np.ndarray:
    with rasterio.open(ruta) as src:
        salida = np.zeros((len(bandas),) + forma, dtype=np.float32)
        for i, b in enumerate(bandas):
            reproject(rasterio.band(src, b), salida[i], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=crs, resampling=Resampling.average,
                      src_nodata=nodata if nodata is not None else src.nodata, dst_nodata=np.nan)
    return salida


def ortomosaico(ruta, transform, crs, forma):
    rgba = a_malla(ruta, [1, 2, 3, 4], transform, crs, forma)
    R, G, B, A = rgba
    suma = np.maximum(R + G + B, 1e-6)
    return {"brillo": (R + G + B) / 3, "exg": (2 * G - R - B) / suma, "cobertura": A / 255}


def leer_las(ruta: Path):
    """X, Y, Z (m) de un .las sin comprimir (formato de punto cualquiera: X, Y, Z son int32 al inicio)."""
    with open(ruta, "rb") as f:
        h = f.read(375)
    off_datos = int(np.frombuffer(h, "<u4", 1, 96)[0])
    largo = int(np.frombuffer(h, "<u2", 1, 105)[0])
    n = int(np.frombuffer(h, "<u4", 1, 107)[0])
    version = (h[24], h[25])
    if n == 0 and version >= (1, 4):
        n = int(np.frombuffer(h, "<u8", 1, 247)[0])
    escala = np.frombuffer(h, "<f8", 3, 131)
    origen = np.frombuffer(h, "<f8", 3, 155)
    m = np.memmap(ruta, dtype=np.uint8, mode="r", offset=off_datos, shape=(n, largo))
    xyz = np.ascontiguousarray(m[:, :12]).view("<i4").reshape(n, 3)
    return xyz[:, 0] * escala[0] + origen[0], xyz[:, 1] * escala[1] + origen[1], xyz[:, 2] * escala[2] + origen[2]


def altura_vegetacion(ruta_las, dtm, transform, forma, q: float = 95):
    x, y, z = leer_las(ruta_las)
    inv = ~transform
    col, fila = inv * (x, y)
    col, fila = np.floor(col).astype(np.int64), np.floor(fila).astype(np.int64)
    ok = (fila >= 0) & (fila < forma[0]) & (col >= 0) & (col < forma[1])
    celda = fila[ok] * forma[1] + col[ok]
    zc = z[ok]
    orden = np.argsort(celda, kind="stable")
    celda, zc = celda[orden], zc[orden]
    cortes = np.flatnonzero(np.diff(celda)) + 1
    grupos = np.split(zc, cortes)
    ids = celda[np.r_[0, cortes]]
    alto = np.full(forma[0] * forma[1], np.nan, dtype=np.float32)
    alto[ids] = [np.percentile(g, q) for g in grupos]
    return alto.reshape(forma) - dtm, int(ok.sum())


def mejor_desplazamiento(oscuro, tmax):
    """(dy, dx, correlación) que maximiza corr(oscuro desplazado, tmax) en la zona común."""
    mejor = (0, 0, -1.0)
    H, W = tmax.shape
    d = DESPLAZ
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            a = oscuro[d + dy:H - d + dy, d + dx:W - d + dx]
            b = tmax[d:H - d, d:W - d]
            m = np.isfinite(a) & np.isfinite(b)
            r = np.corrcoef(a[m], b[m])[0, 1]
            if r > mejor[2]:
                mejor = (dy, dx, float(r))
    return mejor


def figura(k, mapas, tmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    campos = [("brillo_pre", "gray"), ("exg_pre", "Greens"), ("altura", "viridis"), ("dtm", "terrain"),
              ("oscurecimiento", "magma"), (None, "inferno")]
    fig, ax = plt.subplots(1, len(campos), figsize=(4 * len(campos), 4))
    for a, (c, cmap) in zip(ax, campos):
        v = tmax if c is None else mapas[c]
        im = a.imshow(v, cmap=cmap, vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98))
        a.set_title(f"{k}: {'T máx (°C), escenario completo' if c is None else c}", fontsize=9)
        a.axis("off")
        plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    plt.savefig(CARPETA / f"mapas_{k}.png", dpi=80)
    plt.close(fig)


def main():
    cr.log(CARPETA, "Inicio mapas del terreno (4.1 / 4.2)")
    (CARPETA / "datos").mkdir(exist_ok=True)
    for k, p in PLOT.items():
        completo = np.load(cr.TRAINING / f"CA/FLAME3/seleccion1/flame_plot_{k[1]}/evo_celsius.npy")
        for lado in (100, 200):
            destino = CARPETA / "datos" / f"{k}_{lado}.npz"
            if destino.exists():
                continue
            transform, crs, forma = malla(k, lado)
            pre = ortomosaico(EXTRA / f"plot{p}_preburn_ortho.tif", transform, crs, forma)
            post = ortomosaico(EXTRA / f"plot{p}_postburn_ortho.tif", transform, crs, forma)
            dtm = a_malla(EXTRA / f"plot{p}_preburn_dtm.tif", [1], transform, crs, forma)[0]
            alto, npts = altura_vegetacion(EXTRA / f"plot{p}_preburn_pointcloud.las", dtm, transform, forma)
            mapas = {"brillo_pre": pre["brillo"], "exg_pre": pre["exg"], "cobertura_pre": pre["cobertura"],
                     "brillo_post": post["brillo"], "exg_post": post["exg"], "cobertura_post": post["cobertura"],
                     "dtm": dtm, "altura": alto, "oscurecimiento": pre["brillo"] - post["brillo"]}
            np.savez(destino, **mapas)
            cr.log(CARPETA, f"{k} {lado}×{lado}: {npts:,} puntos en la malla; cobertura pre/post "
                            f"{np.nanmean(pre['cobertura']):.3f}/{np.nanmean(post['cobertura']):.3f}; altura mediana "
                            f"{np.nanmedian(alto):.2f} m (p90 {np.nanpercentile(alto, 90):.2f}); DTM "
                            f"{np.nanmin(dtm):.2f}–{np.nanmax(dtm):.2f} m")
            if lado == 100:
                tmax = completo.max(0)
                dy, dx, r = mejor_desplazamiento(mapas["oscurecimiento"], tmax)
                a, b = mapas["oscurecimiento"], tmax
                m = np.isfinite(a)
                r_sin = float(np.corrcoef(a[m], b[m])[0, 1])
                cr.escribir(CARPETA, {"escenario": k, "lado": lado, "corr_sin_desplazar": round(r_sin, 4),
                                      "mejor_dy": dy, "mejor_dx": dx, "corr_mejor": round(r, 4)}, "alineacion.csv")
                cr.log(CARPETA, f"{k}: corr(oscurecimiento, T máx) sin desplazar {r_sin:.3f}; mejor desplazamiento "
                                f"dy={dy}, dx={dx} celdas con corr {r:.3f}")
                figura(k, mapas, tmax)
    cr.log(CARPETA, "Fin")


if __name__ == "__main__":
    main()
