#include <chrono>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/count.h>


// function to sort elements using thrust (GPU)
void _sort_by_key(float* keys, int* values,int size) {

        thrust::sequence(thrust::device, values, values+size);
        thrust::sort_by_key(thrust::device, keys, keys + size,   values,  thrust::greater<float>());

}

int main(void)
{
    bool DEBUG = false;
    int N = 1000000;
    float* scores;
    int* index;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&scores, N*sizeof(float));
    cudaMallocManaged(&index, N*sizeof(int));

    // initialize scores array on the host
    for (int i = 0; i < N; i++) {
        scores[i] = 0.3 + static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX / (0.9 - 0.3)));
    }

    if (DEBUG){
        std::cout << "scores after: ";
        for (int i=0 ; i < N ; i++){
            std::cout << ", " << scores[i];
        }
        std::cout << "\n";
    }
    

    
    auto startTime = std::chrono::high_resolution_clock::now();
    _sort_by_key(scores, index, N);
    auto endTime = std::chrono::high_resolution_clock::now();
    double sort_time = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    if (DEBUG){
        std::cout << "scores after: ";
        for (int i=0 ; i < N ; i++){
            std::cout << ", " << scores[i];
        }
        std::cout << "\n";

    }
    
    std::cout << "Sort Duration: " << sort_time << "\n";

    // Free memory
    cudaFree(scores);
    cudaFree(index);

    return 0;
}