#!/bin/bash

for i in {3..3}; 
do
	sed -i "s/PYBIND11_MODULE(libsetpackingenv, m)/PYBIND11_MODULE(libsetpackingenvo$i, m)/g" libsetpackingenv.cpp
	
	makelib=$(echo "g++ -m64 -O3 -DIL_STD -DNOBJS=$i -Wall -shared -std=c++17 -fPIC -I./include $(python3 -m pybind11 --includes) libsetpackingenv.cpp setpackingenv.cpp ../common/bdd/*.cpp ../common/util/*.cpp ./bdd/*.cpp ./instances/*.cpp $(python3-config --ldflags) -lm -ldl -pthread -o libsetpackingenvo$i.cpython-38-x86_64-linux-gnu.so")

	eval $makelib
	
	sed -i "s/PYBIND11_MODULE(libsetpackingenvo$i, m)/PYBIND11_MODULE(libsetpackingenv, m)/g" libsetpackingenv.cpp
done