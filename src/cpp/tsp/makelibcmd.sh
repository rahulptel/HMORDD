#!/bin/bash

SUFFIX_STATIC=".cpython-38-x86_64-linux-gnu.so"

if [ "${machine}" = "cc" ] || [ "${machine}" = "desktop" ]; then
    EXT_SUFFIX="$(python3-config --extension-suffix)"
    echo "[makelibcmd] machine='${machine}' -> using dynamic suffix '${EXT_SUFFIX}'"
else
    EXT_SUFFIX="${SUFFIX_STATIC}"
    echo "[makelibcmd] machine='${machine}' -> using static suffix '${EXT_SUFFIX}'"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(realpath "${SCRIPT_DIR}/../../..")"
DEST_DIR="${ROOT_DIR}/resources/bin/tsp"
mkdir -p "${DEST_DIR}"

for i in {3..7}; do
    sed -i "s/PYBIND11_MODULE(libtspenv, m)/PYBIND11_MODULE(libtspenvo$i, m)/g" libtspenv.cpp

    makelib=$(echo "g++ -m64 -O3 -DIL_STD -DNOBJS=$i -Wall -shared -std=c++17 -fPIC -I. $(python3 -m pybind11 --includes) libtspenv.cpp tspenv.cpp dd/*.cpp instance/*.cpp ../common/util/*.cpp $(python3-config --ldflags) -lm -ldl -pthread -o libtspenvo$i${EXT_SUFFIX}")

    eval $makelib

    BUILD_ARTIFACT="libtspenvo$i${EXT_SUFFIX}"
    if [ -f "${BUILD_ARTIFACT}" ]; then
        mv -f "${BUILD_ARTIFACT}" "${DEST_DIR}/"
        echo "[makelibcmd] moved ${BUILD_ARTIFACT} -> ${DEST_DIR}"
    fi

    sed -i "s/PYBIND11_MODULE(libtspenvo$i, m)/PYBIND11_MODULE(libtspenv, m)/g" libtspenv.cpp
done
