#!/bin/sh
# Copia los videos de la presentación a media/videos/ en WebM (VP9).
# Firefox y Chrome decodifican VP9 sin códecs del sistema; los MP4 H.264 originales
# fallan en el Firefox de Flatpak (MEDIA_ERR_DECODE). Los originales no se modifican.
# Uso (desde cualquier carpeta): sh Presentacion/media/convertir.sh
# langton_diez_colores_una_hormiga.webm no sale de aquí: lo genera langton_diez_colores_una_hormiga.py.
cd "$(dirname "$0")" || exit 1
R=../..
mkdir -p videos

# destino (sin extensión)            origen relativo a la raíz del repo
while read -r destino origen; do
	[ -z "$destino" ] && continue
	salida="videos/$destino.webm"
	if [ -f "$salida" ] && [ "$salida" -nt "$R/$origen" ]; then
		echo "ya existe  $salida"
		continue
	fi
	# -nostdin: sin él ffmpeg lee la lista de abajo y se come parte de los renglones
	ffmpeg -nostdin -hide_banner -loglevel error -y -i "$R/$origen" \
		-c:v libvpx-vp9 -crf 24 -b:v 0 -row-mt 1 -deadline good -cpu-used 4 \
		-pix_fmt yuv420p -an "$salida" && echo "listo      $salida"
done <<'EOF'
gol_nave                       Entregables/Animation/GameOfLife/nave_comparacion.mp4
gol_aleatoria                  Entregables/Animation/GameOfLife/aleatoria_comparacion.mp4
sandpile_dos_generadores       Entregables/Animation/Sandpile/dos_generadores_abierta_comparacion.mp4
sandpile_periodica             Entregables/Animation/Sandpile/un_generador_periodica_comparacion.mp4
langton_una_hormiga            Entregables/Animation/LangtonAnt/una_hormiga_RL_comparacion.mp4
langton_dos_hormigas           Entregables/Animation/LangtonAnt/dos_hormigas_RL_comparacion.mp4
langton_diez_colores           Entregables/Animation/LangtonAnt/diez_colores_RLRLRLRRRL_comparacion.mp4
langton_ocho_hormigas          Entregables/Animation/LangtonAnt/ocho_hormigas_RL_aleatorias_comparacion.mp4
flame_plot_1                   Entregables/Animation/FLAME3/flame_plot_1_abierta_comparacion.mp4
flame_plot_2                   Entregables/Animation/FLAME3/flame_plot_2_abierta_comparacion.mp4
flame_centro                   Entregables/Animation/FLAME3/centro_200x200_periodica_nca.mp4
sevir_mejor                    Simulator/training/NCA/SEVIR/barrido/videos/mejor.mp4
physarum_ruido                 Simulator/training/NCA/Physarum/dado/videos/b16x128_008-dish5_6.mp4
physarum_mas_probable          Simulator/training/NCA/Physarum/prueba/320_6estados/real_vs_red.mp4
physarum_edad                  Simulator/training/NCA/Physarum/edad/videos/c_e6edad_s43_008-dish5_suave.mp4
EOF
