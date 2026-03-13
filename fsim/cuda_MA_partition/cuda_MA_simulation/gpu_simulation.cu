#include "./gpu_simulation.cuh"

constexpr int _NUM_THREADS = 512;
constexpr uint32_t UINT32T_BITS = std::numeric_limits<uint32_t>::digits;
// #define PRINT_SIMULATION_OUTPUTS_DSP

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void GPUSimulator::_run_gates_DSP_gpu(const int _total_num_levels,
  const std::vector<int> &_numGates_per_level,
  const int *_numGates_per_level_gpu,
  const int *_invAdj_gpu,
  const int *_invAdj_index_table_gpu,
  const int *_pi_gate_po_gate_type_gpu,
  const uint32_t *_patterns_gpu,
  const std::vector<Pattern> _patterns,
  uint32_t *_pi_gate_po_output_res_gpu) 
{

  _num_rounds = 1;  // use for profiling
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
      ((_num_pattern / (UINT32T_BITS * (rd + 1))))
      ? (UINT32T_BITS)
      : (_num_pattern % UINT32T_BITS);

    int num_blocks, num_threads; int num_accumGates = 0;

    for (int level = 0; level < _total_num_levels; level++) {
      const int num_gates_per_level = (_numGates_per_level[level]);
      num_blocks  = (num_gates_per_level > _NUM_THREADS) ? 
                    (num_gates_per_level + _NUM_THREADS - 1)/_NUM_THREADS : 
                    (1);
      num_threads = (num_gates_per_level > _NUM_THREADS) ? 
                    (_NUM_THREADS) : 
                    (num_gates_per_level);

      CUDA_CHECK(cudaEventRecord(start));
      
      _run_gate_DSP <<< num_blocks, num_threads >>> (num_accumGates, _numGates_per_level_gpu, 
        _invAdj_gpu, _invAdj_index_table_gpu, 
        _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
        _patterns_gpu, rd, num_gates_per_level, _num_PIs);
      
      num_accumGates += num_gates_per_level; 

      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float msec = 0;
      CUDA_CHECK(cudaEventElapsedTime(&msec, start, stop));
      printf("Level %3d | Gates: %8d | Runtime: %8.3f us\n", 
            level, num_gates_per_level, msec * 1000.0);
      // cudaDeviceSynchronize();
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    #ifdef PRINT_SIMULATION_OUTPUTS_DSP
      if (rd == 0) {
        // shift and copy answer to the good results
        int num_blocks = (_sum_pi_gates_pos > _NUM_THREADS) ? 
          (_sum_pi_gates_pos + _NUM_THREADS - 1)/_NUM_THREADS : (1);
        int num_threads = (_sum_pi_gates_pos > _NUM_THREADS) ? 
          (_NUM_THREADS) : (_sum_pi_gates_pos);    
        _write_and_shift_to_array_gpu <<< num_blocks, num_threads >>> (num_testcases_this_round, 
          _pi_gate_po_output_res_gpu, _sum_pi_gates_pos);
        cudaCheckErrors("CUDA: _write_and_shift_to_array_gpu launch- Failure");

        cudaDeviceSynchronize();
        cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
        
        std::cout << "GOOD resutls ans:" << std::endl;
        _print_simulation_results <<< 1, 1 >>> (_pi_gate_po_output_res_gpu, _sum_pi_gates_pos);
        cudaDeviceSynchronize();
        cudaCheckErrors("CUDA: _print_simulation_results cudaDeviceSynchronize - Failure");
      }
    #endif
  }
}

__device__ __forceinline__ void _apply_INV(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0];
  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_AND(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_OR(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];

  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_XOR(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_NAND(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_NOR(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_XNOR(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_MUX(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  const uint32_t a = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  const uint32_t b = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+1]];
  const uint32_t s = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+2]];

  uint32_t ret = ((s & b) | ( a & (!s)));

  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_CLKBUF(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_PI(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,  const uint32_t pattern_val) 
{
  _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
}

__device__ __forceinline__ void _apply_PO(const int gate_idx, const int *_invAdj_gpu, 
  const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) 
{
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
}

__global__ void _run_gate_DSP(const int num_accumGates, const int *_numGates_per_level_gpu,
  const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
  const int *_pi_gate_po_gate_type_gpu, uint32_t *_pi_gate_po_output_res_gpu, 
  const uint32_t *_patterns_gpu, const size_t rd, 
  const int num_gates_per_level, const int _num_PIs) {
  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (num_gates_per_level > t_idx) {
    int real_g_idx = num_accumGates+t_idx;
    int type = _pi_gate_po_gate_type_gpu[real_g_idx];

    switch (type) { 
      case 0:
      _apply_INV(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 1:
      _apply_AND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 2:
      _apply_OR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 3:
      _apply_XOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 4:
      _apply_NAND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 5:
      _apply_NOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 6:
      _apply_XNOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 7:
      _apply_MUX(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 8:
      _apply_CLKBUF(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 9:
      _apply_PI(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
      break;
    case 10:
      _apply_PO(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 11:
      break;
    }      
  }
}

__global__ void _write_and_shift_to_array_gpu(const size_t bits, 
  uint32_t *_pi_gate_po_output_res_gpu, 
  const int total_num_gates) 
{
  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (t_idx < total_num_gates) {
    // printf("pre: _pi_gate_po_output_res_gpu[%d] = %u\n", t_idx, _pi_gate_po_output_res_gpu[t_idx]);
    _pi_gate_po_output_res_gpu[t_idx] = (_pi_gate_po_output_res_gpu[t_idx] << (UINT32T_BITS - bits));
    _pi_gate_po_output_res_gpu[t_idx] >>= (UINT32T_BITS - bits);
    // printf("post: _pi_gate_po_output_res_gpu[%d] = %lu\n", t_idx, _pi_gate_po_output_res_gpu[t_idx]);
  } 
}

__global__  void _print_simulation_results(const uint32_t *_pi_gate_po_output_res_gpu, 
  const int total_num_gates)
{
  for (int i = 0; i < total_num_gates; i++)
    printf("gate_%d.output = %u\n", i, _pi_gate_po_output_res_gpu[i]);
}
