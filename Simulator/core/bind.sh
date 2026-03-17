source bind/bin/activate

pip install -r requirements.txt

mkdir build
cd build
rm -rf *
cmake .. -DPython_EXECUTABLE=$(which python)
make

cp CA*.so ../../GUI/