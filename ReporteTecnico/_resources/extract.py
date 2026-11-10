import re
import os
import sys
import pyperclip

def procesar_latex(ruta_entrada):
    if not os.path.exists(ruta_entrada):
        print(f"❌ Error: El archivo '{ruta_entrada}' no existe.")
        return None

    try:
        with open(ruta_entrada, 'r', encoding='utf-8') as f:
            contenido = f.read()

        # 1. Eliminar bloques \begin{comment} ... \end{comment}
        contenido = re.sub(r'\\begin\{comment\}.*?\\end\{comment\}', '', contenido, flags=re.DOTALL)

        # 2. Eliminar comentarios % (respetando \%)
        contenido = re.sub(r'(?<!\\)%.*', '', contenido)

        # 3. Manejar Figuras (Caption)
        def extraer_caption(match):
            cap = re.search(r'\\caption\{(.*?)\}', match.group(0))
            return f"\n[FIGURA: {cap.group(1) if cap else 'Sin descripción'}]\n"
        
        contenido = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', extraer_caption, contenido, flags=re.DOTALL)

        # 4. Citas IEEE numéricas [1], [2]...
        citas = {}
        def num_cita(m):
            c = m.group(1)
            if c not in citas: citas[c] = len(citas) + 1
            return f"[{citas[c]}]"
        
        contenido = re.sub(r'\\cite\{(.*?)\}', num_cita, contenido)

        # 5. Limpieza de comandos y matemáticas
        contenido = re.sub(r'\$\$.*?\$\$|\$.*?\$', '', contenido, flags=re.DOTALL)
        
        # Doble pasada para comandos anidados como \section{\textbf{Texto}}
        for _ in range(2): 
            contenido = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', contenido)
        
        # Eliminar comandos sueltos (ej. \clearpage)
        contenido = re.sub(r'\\[a-zA-Z]+', '', contenido)
        
        # Limpieza de llaves huerfanas y espacios
        contenido = contenido.replace('{', '').replace('}', '')
        resultado = "\n".join([l.strip() for l in contenido.splitlines() if l.strip()])
        
        return resultado

    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python extract.py ruta/al/archivo.tex")
    else:
        ruta_completa = sys.argv[1]
        texto_limpio = procesar_latex(ruta_completa)
        
        if texto_limpio:
            # COPIAR AL PORTAPAPELES DIRECTAMENTE
            pyperclip.copy(texto_limpio)
            
            print(f"---File copied")