# mkdir build
# cd build
# cmake ..
# make

# rm -rf ~/Documents/ESCOM/Simulator/backend/build/*

cd build
rm -rf *
cmake .. -DPython_EXECUTABLE=$(which python)
make
cp ca_backend*.so ../../frontend/