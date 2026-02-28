# CMakeLists.txt in the main directory
add_executable(simple simple.cpp)
add_executable(MA_cuda_partition MA_partition.cu)

# Link fsim libraries
target_link_libraries(simple PRIVATE fsim)
target_link_libraries(MA_cuda_partition PRIVATE fsim_MA_cuda_partition)

# Set compiler flags for the "main" target
target_compile_options(simple PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3
)

# Set compiler flags for the "main" target
target_compile_options(MA_cuda_partition PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -lineinfo
  -O3
)
