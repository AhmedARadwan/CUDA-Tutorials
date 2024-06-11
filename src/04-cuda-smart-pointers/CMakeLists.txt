cmake_minimum_required(VERSION 3.0.2)
project(simple_add)
add_compile_options(-std=c++14)


find_package(CUDA REQUIRED QUIET)
include_directories(${CUDA_INCLUDE_DIRS})
if (CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

list(APPEND LINK_LIBRARY
            ${CUDA_cusparse_LIBRARY}
            ${CUDA_LIBRARIES}
            ${CUDA_curand_LIBRARY}            
            ${CUDA_cusparse_LIBRARY}
            )


# simple add gpu
cuda_add_executable(simple_cuda_smart_pointers src/example.cu)
target_link_libraries(simple_cuda_smart_pointers ${LINK_LIBRARIES})
