#!/bin/bash

set -e
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON ../
make -j$(nproc)

file_array=("c432" "c499" "c880" "c1355" "c1908" "c2670" "c3540" "c5315" "c6288" "c7552")

echo "--- Starting Test all C circuits ---"
for file in "${file_array[@]}"; do
    echo "Processing: ${file}"
    ./main/MA_cuda_partition "../benchmark/${file}.ckt" "../benchmark/${file}.flst" "../benchmark/${file}.ptn"
    echo "Finished ${file}"
    echo "--------------------------------"
done
