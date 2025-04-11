#!/bin/bash

mkdir -p support/build
cd support/build

[ -f Makefile ] || (module load python/3.10 pytorch/2; cmake -DTORCH_CUDA_ARCH_LIST="7.0;8.0;8.0+PTX" -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.__path__[0])')" ..)

make -j4
