#ifndef CUDA_UNIQUE_POINTER_H
#define CUDA_UNIQUE_POINTER_H

#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
class CudaUniquePointer {
private:
    T* device_ptr;

public:
    // Constructor: Allocates memory on the device
    explicit CudaUniquePointer(size_t size) {
        cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory");
        }
    }

    // Destructor: Frees the allocated memory on the device
    ~CudaUniquePointer() {
        cudaFree(device_ptr);
    }

    // Access the raw device pointer
    T* get() const {
        return device_ptr;
    }

    // Copy data from host to device
    void copyToDevice(const T* host_ptr, size_t size) {
        cudaError_t err = cudaMemcpy(device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device");
        }
    }

    // Copy data from device to host
    void copyToHost(T* host_ptr, size_t size) const {
        cudaError_t err = cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to host");
        }
    }

    // Prevent copying
    CudaUniquePointer(const CudaUniquePointer&) = delete;
    CudaUniquePointer& operator=(const CudaUniquePointer&) = delete;

    // Allow moving
    CudaUniquePointer(CudaUniquePointer&& other) noexcept : device_ptr(other.device_ptr) {
        other.device_ptr = nullptr;
    }

    CudaUniquePointer& operator=(CudaUniquePointer&& other) noexcept {
        if (this != &other) {
            cudaFree(device_ptr);
            device_ptr = other.device_ptr;
            other.device_ptr = nullptr;
        }
        return *this;
    }
};

#endif // CUDA_UNIQUE_POINTER_H