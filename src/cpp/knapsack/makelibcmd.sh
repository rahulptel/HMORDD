#!/bin/bash

for i in {3..3}; do
    sed -i "s/PYBIND11_MODULE(libknapsackenv, m)/PYBIND11_MODULE(libknapsackenvo$i, m)/g" libknapsackenv.cpp

    makelib=$(echo "g++ -m64 -O3 -DIL_STD -DNOBJS=$i -Wall -shared -std=c++17 -fPIC -I. $(python3 -m pybind11 --includes) libknapsackenv.cpp knapsackenv.cpp bdd/*.cpp instances/*.cpp ../common/util/*.cpp $(python3-config --ldflags) -lm -ldl -pthread -o libknapsackenvo$i.cpython-38-x86_64-linux-gnu.so")

    eval $makelib

    sed -i "s/PYBIND11_MODULE(libknapsackenvo$i, m)/PYBIND11_MODULE(libknapsackenv, m)/g" libknapsackenv.cpp
done
