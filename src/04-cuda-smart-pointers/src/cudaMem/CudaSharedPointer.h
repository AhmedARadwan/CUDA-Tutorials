#ifndef CUDA_SHARED_POINTER_H
#define CUDA_SHARED_POINTER_H

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <atomic>

template <typename T>
class CudaSharedPointer {
private:
    T* device_ptr;
    std::atomic<int>* ref_count;

public:
    // Constructor: Allocates memory on the device
    explicit CudaSharedPointer(size_t size) : ref_count(new std::atomic<int>(1)) {
        cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(T));
        if (err != cudaSuccess) {
            delete ref_count;
            throw std::runtime_error("Failed to allocate device memory");
        }
    }

    // Destructor: Frees the allocated memory on the device if no more references
    ~CudaSharedPointer() {
        if (--(*ref_count) == 0) {
            cudaFree(device_ptr);
            delete ref_count;
        }
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

    // Copy constructor
    CudaSharedPointer(const CudaSharedPointer& other) : device_ptr(other.device_ptr), ref_count(other.ref_count) {
        ++(*ref_count);
    }

    // Copy assignment
    CudaSharedPointer& operator=(const CudaSharedPointer& other) {
        if (this != &other) {
            if (--(*ref_count) == 0) {
                cudaFree(device_ptr);
                delete ref_count;
            }
            device_ptr = other.device_ptr;
            ref_count = other.ref_count;
            ++(*ref_count);
        }
        return *this;
    }
};

#endif // CUDA_SHARED_POINTER_H
