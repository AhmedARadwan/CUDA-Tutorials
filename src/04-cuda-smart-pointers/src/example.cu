#include <iostream>
#include "cudaMem/CudaUniquePointer.h"
#include "cudaMem/CudaSharedPointer.h"

// CUDA kernel example
__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    // Using CudaUniquePointer
    {
        CudaUniquePointer<int> dev_a(arraySize);
        CudaUniquePointer<int> dev_b(arraySize);
        CudaUniquePointer<int> dev_c(arraySize);

        dev_a.copyToDevice(a, arraySize);
        dev_b.copyToDevice(b, arraySize);

        addKernel<<<1, arraySize>>>(dev_c.get(), dev_a.get(), dev_b.get());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed to launch addKernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
            return 1;
        }

        dev_c.copyToHost(c, arraySize);

        std::cout << "Result with CudaUniquePointer: ";
        for (int i = 0; i < arraySize; ++i) {
            std::cout << c[i] << " ";
        }
        std::cout << std::endl;
    }

    // Using CudaSharedPointer
    {
        CudaSharedPointer<int> dev_a(arraySize);
        CudaSharedPointer<int> dev_b(arraySize);
        CudaSharedPointer<int> dev_c(arraySize);

        dev_a.copyToDevice(a, arraySize);
        dev_b.copyToDevice(b, arraySize);

        addKernel<<<1, arraySize>>>(dev_c.get(), dev_a.get(), dev_b.get());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed to launch addKernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
            return 1;
        }

        dev_c.copyToHost(c, arraySize);

        std::cout << "Result with CudaSharedPointer: ";
        for (int i = 0; i < arraySize; ++i) {
            std::cout << c[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
