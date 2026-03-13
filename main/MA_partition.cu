#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <iostream>

#include <fsim/cuda_MA_partition/fsim_cuda_MA_partition.cuh>

constexpr std::chrono::duration<double> chro_zero() {
  return std::chrono::duration<double>::zero();
}

double round_to(double value, double precision = 1.0){
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <ckt_path> <flst_path> <ptn_path>" << std::endl;
    return 1;
  }

  // Init recording time 
  std::chrono::duration<double> dur_sim = chro_zero();

  // Inputs 
  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);

  // user perspective
  CUDAMAPartitioner cudaMAPartitioner;

  // Read data and Run cudaMAPartitioner
  cudaMAPartitioner.read(ckt_path, flst_path, ptn_path);
  
  // Prepare Simulator
  cudaMAPartitioner.prepare_gpu_simulation();

  // Input for number of simulation rounds
  const size_t _NUM_SIMULATION_RDS = 1; 
  cudaMAPartitioner.run(_NUM_SIMULATION_RDS);
  
  // free memory
  cudaMAPartitioner.freeMem();
  
  return 0;
}