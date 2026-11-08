# Opción 4 — ¿FLAME 3 trae RGB, viento o información del terreno?

Revisión del 2026-10-14. No es un experimento: se revisó qué contiene el dataset de Kaggle
`brycehopkins/flame-3-nadir-thermal-plot-subset` (listado de archivos con `kagglesdk`, sin descargar
todo) y el artículo de FLAME 3 (arXiv 2412.02831). Se descargaron solo los archivos chicos a
`Simulator/training/Dataset/FLAME3/hanna_hammock_meta/`.

## Lo que hay en el dataset (4630 archivos, 37.5 GB)

| Carpeta | Contenido | Tamaño |
|---|---|---|
| `plot N/duringburn/raw_rgb_jpg` | Foto RGB de cada frame durante la quema (297 en plot 1, 521 en plot 2) | — |
| `plot N/duringburn/raw_thermal_*`, `geo_thermal_tiff_celsius` | Lo que ya usamos | — |
| `plot N/preburn/` | `plotN_preburn_ortho.tif` (ortomosaico RGB), `plotN_preburn_dtm.tif` (modelo de elevación), `plotN_preburn_pointcloud.las` (nube de puntos), 83–87 fotos RGB | plot 1: 245 + 59 + 598 MB; plot 2: 135 + 34 + 397 MB |
| `plot N/postburn/` | Lo mismo después de la quema | plot 1: 351 + 86 + 1475 MB; plot 2: 215 + 50 + 504 MB |
| `GCPs/`, `plotN_gcp_coords.txt` | Puntos de control en tierra (georreferencia) | < 1 MB |
| `metadata/Hammock Wx.jpeg` | Clima (ver abajo) | 0.1 MB |
| `metadata/BriefMap.pdf` | Mapa de la unidad de quema con cobertura de suelo y **líneas de ignición** | 0.6 MB |
| `hannah_hammok_README.rtf` | Notas de captura y procesamiento | 2 KB |

## Clima (`Hammock_Wx.jpeg`)

Nota de campo del 28 de febrero de 2023 (lecturas cada hora):

| Hora | Temperatura | Humedad | Viento | Cielo |
|---|---|---|---|---|
| 13:00 | 81 °F (27 °C) | 58 % | 4 mph (1.8 m/s) | nubes dispersas |
| 14:00 | 84 °F (29 °C) | 54 % | 4 mph (1.8 m/s) | nubes dispersas |
| 15:00 | 84 °F (29 °C) | 52 % | 1 mph (0.4 m/s) | nubes dispersas |

No hay dirección del viento. El viento fue débil, así que la propagación debió depender más del
combustible y del patrón de ignición que del viento.

## Notas del README

- Las marcas de tiempo de las imágenes están 4 h atrasadas (09:15 en el archivo = 13:15 real).
- Plot 2 tiene una pausa de 9 min 10 s entre la parte 1 y la parte 2 porque "no hubo crecimiento
  significativo del fuego" y había que cuidar la batería (coincide con lo poco que pasa en p2).
- Las imágenes térmicas se estabilizaron con ECC (afín) frame a frame y se georreferenciaron con
  los GCP del primer frame; °C = DN · 0.1 − 275.
- Ortomosaico, DSM y nube de puntos procesados en Agisoft Metashape 1.8.1.

## Lo que se podría usar (no se ha hecho)

1. **Mapa de combustible** a partir del ortomosaico RGB antes de la quema (vegetación vs suelo) o
   de la nube de puntos (altura de la vegetación). Es lo más prometedor: la opción 3 mostró que la
   térmica antes de la ignición casi no distingue lo que se quema.
2. **Pendiente** a partir del DTM.
3. **Fotos RGB de cada frame**: humo y llama visibles; habría que alinearlas con la térmica.
4. **Líneas de ignición** del mapa: confirman dónde prendieron el fuego.

Para usar 1 y 2 hay que bajar el ortomosaico o el DTM (cientos de MB por parcela) y registrarlos
contra la malla térmica recortada y alineada (las térmicas están georreferenciadas, pero el
recorte cuadrado y la alineación propia hay que llevarlos a coordenadas UTM).

Fuentes: [FLAME 3 (arXiv 2412.02831)](https://arxiv.org/html/2412.02831v1),
[IEEE DataPort](https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management).
