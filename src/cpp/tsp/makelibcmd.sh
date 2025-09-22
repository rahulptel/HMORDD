#!/bin/bash

for i in {3..3}; do
    sed -i "s/PYBIND11_MODULE(libtspenv, m)/PYBIND11_MODULE(libtspenvo$i, m)/g" libtspenv.cpp

    makelib=$(echo "g++ -m64 -O3 -DIL_STD -DNOBJS=$i -Wall -shared -std=c++17 -fPIC -I. $(python3 -m pybind11 --includes) libtspenv.cpp tspenv.cpp dd/*.cpp instance/*.cpp ../common/util/*.cpp $(python3-config --ldflags) -lm -ldl -pthread -o libtspenvo$i.cpython-38-x86_64-linux-gnu.so")

    eval $makelib

    sed -i "s/PYBIND11_MODULE(libtspenvo$i, m)/PYBIND11_MODULE(libtspenv, m)/g" libtspenv.cpp
done
