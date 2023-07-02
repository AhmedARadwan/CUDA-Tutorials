cmake_minimum_required(VERSION 3.0.2)
project(object_transformation)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  autoware_msgs
)

find_package(Eigen3 REQUIRED) 
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(CUDA REQUIRED QUIET)

include_directories(include
                    ${CUDA_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
)

if (CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lidar_detector
#  CATKIN_DEPENDS roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

list(APPEND LINK_LIBRARY
            ${CUDA_cusparse_LIBRARY}
            ${CUDA_curand_LIBRARY}            
            ${CUDA_cusparse_LIBRARY}
            )

# simple transformation gpu
cuda_add_executable(transform_objects_node src/transform_objects.cu)
target_link_libraries(transform_objects_node ${LINK_LIBRARIES} ${CUDA_LIBRARIES} ${catkin_LIBRARIES})

add_executable(transform_objects_cpu src/transform_objects.cpp)
target_link_libraries(transform_objects_cpu ${catkin_LIBRARIES})

