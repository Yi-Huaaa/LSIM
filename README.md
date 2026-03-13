# Logic simulation on GPU for ECE757 benchmark

## Build and Run

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

4. Outputs:
```
--- Starting Test all C circuits ---
Processing: c432
_num_PIs: 36, _num_POs: 7, _num_inner_gates 134
prepare_GPU_simulation: 50.241
Level   0 | Gates:       36 | Runtime: 3599.008 us
Level   1 | Gates:       21 | Runtime:    5.120 us
Level   2 | Gates:        9 | Runtime:    5.120 us
Level   3 | Gates:        3 | Runtime:    4.096 us
Level   4 | Gates:        1 | Runtime:    4.096 us
Level   5 | Gates:        1 | Runtime:    4.096 us
Level   6 | Gates:       19 | Runtime:    4.224 us
Level   7 | Gates:        9 | Runtime:    6.144 us
Level   8 | Gates:        3 | Runtime:    5.088 us
Level   9 | Gates:        1 | Runtime:    3.360 us
Level  10 | Gates:        1 | Runtime:    4.096 us
Level  11 | Gates:        1 | Runtime:    4.096 us
Level  12 | Gates:       19 | Runtime:    4.096 us
Level  13 | Gates:        9 | Runtime:    5.120 us
Level  14 | Gates:        1 | Runtime:    4.096 us
Level  15 | Gates:        1 | Runtime:    4.096 us
Level  16 | Gates:        1 | Runtime:    4.032 us
Level  17 | Gates:       10 | Runtime:    4.096 us
Level  18 | Gates:        9 | Runtime:    4.896 us
Level  19 | Gates:        8 | Runtime:    5.088 us
Level  20 | Gates:        5 | Runtime:    4.096 us
Level  21 | Gates:        4 | Runtime:    4.416 us
Level  22 | Gates:        4 | Runtime:    4.224 us
Level  23 | Gates:        1 | Runtime:    4.800 us
Finished c432
--------------------------------
(...)
Processing: c7552
_num_PIs: 206, _num_POs: 107, _num_inner_gates 1147
prepare_GPU_simulation: 41.237
Level   0 | Gates:      206 | Runtime: 3591.168 us
Level   1 | Gates:      145 | Runtime:    6.144 us
Level   2 | Gates:      150 | Runtime:    5.120 us
Level   3 | Gates:      107 | Runtime:    6.144 us
Level   4 | Gates:      156 | Runtime:    5.760 us
Level   5 | Gates:      114 | Runtime:    6.144 us
Level   6 | Gates:       99 | Runtime:    6.144 us
Level   7 | Gates:       69 | Runtime:    7.104 us
Level   8 | Gates:       49 | Runtime:    6.144 us
Level   9 | Gates:       52 | Runtime:    7.168 us
Level  10 | Gates:       41 | Runtime:    6.848 us
Level  11 | Gates:       38 | Runtime:    8.192 us
Level  12 | Gates:       34 | Runtime:    8.192 us
Level  13 | Gates:       32 | Runtime:    7.168 us
Level  14 | Gates:       26 | Runtime:    7.168 us
Level  15 | Gates:       34 | Runtime:    7.168 us
Level  16 | Gates:       35 | Runtime:    7.936 us
Level  17 | Gates:       26 | Runtime:    8.000 us
Level  18 | Gates:       25 | Runtime:    7.072 us
Level  19 | Gates:       14 | Runtime:    5.120 us
Level  20 | Gates:        7 | Runtime:    4.096 us
Level  21 | Gates:        1 | Runtime:    4.800 us
Finished c7552
--------------------------------
```

---

## Notes

### Available Benchmarks
The repository includes the following circuit graphs for simulation:
* **Small:** `c17`, `c432`, `c499`, `c880`
* **Medium:** `c1355`, `c1908`, `c2670`, `c3540`
* **Large:** `c5315`, `c6288`, `c7552`
* Files are located in the `../benchmark/` directory relative to your build folder.

### Environment Requirements

* **GPU Architecture:** This simulator is optimized for NVIDIA GPUs (available on the Euler workstations).
* **Compiler:** Ensure you are using `nvcc` (CUDA toolkit 12.0+) and `gcc/g++` 11 or 12.

### Common Issues
* **Permission Denied:** If you cannot execute `run.sh`, run `chmod +x run.sh` inside the build folder.
* **File Not Found:** Ensure you run `bash run.sh` **inside** the `build/` directory so the script can correctly find the `../benchmark/` folder.
* **Build Errors:** If you change any `.cu` or `.cpp` files, run `make -j` again to recompile the project.
