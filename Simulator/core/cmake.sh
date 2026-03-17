source ~/Documents/ESCOM/Simulator/frontend/venv/bin/activate

mkdir build
cd build
rm -rf *
cmake .. -DPython_EXECUTABLE=$(which python)
make

cp core*.so ../../frontend/