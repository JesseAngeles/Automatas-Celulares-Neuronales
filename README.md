Para ejecutar es necesario el comando:
```
python -m gui.MainWindow
```

Para compilar CASimulador.++ se ejecutan los siguientes comandos desde ./core/
```
mkdir build && cd build
cmake ..
make
```

**Ejecución en Windows**

Activar el entorno virtual: 
```
.\venv\Scripts\Activate.ps1
```

Desde la raíz del proyecto:
```
cd core
rmdir -Recurse -Force build
mkdir build
cd build
```

Compilar:
```
cmake .. -G "MinGW Makefiles" `
  -DCMAKE_CXX_COMPILER=c++ `
  -DCMAKE_C_COMPILER=gcc `
  -Dpybind11_DIR="C:/Users/redix/OneDrive/Documentos/Trabajo-Terminal/Codigo/simulator/venv/Lib/site-packages/pybind11/share/cmake/pybind11"
```

Compilar el modulo:
```
cmake --build .
```