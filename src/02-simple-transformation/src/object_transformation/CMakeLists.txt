cmake_minimum_required(VERSION 3.0.2)
project(simple_add)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  autoware_msgs
)

find_package(CUDA REQUIRED QUIET)

include_directories(${CUDA_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
)

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


# simple transformation gpu
cuda_add_executable(transform_objects src/transform_objects.cu)
target_link_libraries(transform_objects ${LINK_LIBRARIES} ${catkin_LIBRARIES})
