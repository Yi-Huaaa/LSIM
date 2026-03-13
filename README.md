# Logic simulation on GPU for ECE757 benchmark

## Build and run

1. Clone this branch: ECE757
```
git clone -b ECE757 https://github.com/Yi-Huaaa/LSIM.git
cd LSIM
```

2. Build the project
```
mkdir -p build
cp run.sh build/
cd build
cmake ../
make -j
```

3. Run Benchmarks (Note: under this path: `LSIM/build`)
```
bash run.sh
```

