# Trabajo Terminal

Simulador de autómatas celulares con interfaz gráfica en Python y núcleo de cómputo en C++ (pybind11).

## Tecnologías

- Python (interfaz gráfica)
- C++ (núcleo del simulador)
- pybind11 (enlace Python-C++)
- CMake

## Instalación

**Fedora (recomendado)**

```bash
cd core
mkdir build && cd build
cmake ..
make
```

**Windows**

Activar el entorno virtual:

```powershell
.\venv\Scripts\Activate.ps1
```

Compilar:

```powershell
cd core
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=c++ -DCMAKE_C_COMPILER=gcc
cmake --build .
```

## Ejecución

```bash
python -m gui.MainWindow
```

## Estructura

```
Codigo/         # Código fuente (Python + C++)
Documentacion/  # Documentación del proyecto
```
