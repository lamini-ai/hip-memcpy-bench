
#include <iostream>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <chrono>

void benchmark_gemm(rocblas_handle, int, int);

// This is a benchmarking program for Memcpy operations on MI210 using rocBLAS library
int main(int argc, const char** argv)
{
    // Initialize rocBLAS
    rocblas_handle handle;
    rocblas_status status = rocblas_create_handle(&handle);

    // Check for errors
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS initialization failed" << std::endl;
        return -1;
    }

    // Parse the matrix sizes to be used for benchmarking from the command line
    // The arguments are a list of matrix sizes separated by spaces, e.g. 1024 2048 4096
    std::vector<int> matrix_sizes;
    for (int i = 1; i < argc; i++)
    {
        int m = 0;
        std::sscanf(argv[i], "%d", &m);
        matrix_sizes.push_back(m);
    }

    // Benchmarking GEMM operations
    for (auto& matrix_size : matrix_sizes)
    {
        benchmark_memcpy(handle, matrix_size.first, matrix_size.second);
    }

    return 0;
}

// Benchmarking Memcpy operations
void benchmark_memcpy(rocblas_handle handle, int bytes)
{
    // Print the matrix size
    std::cout << "Benchmarking Memcpy operation on " << bytes << " bytes" << std::endl;

    // Allocate the matrices on the device
    data_type* dA;
    data_type* dB;
    hipError_t error = hipMalloc((void**)&dA, bytes);
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS device memory allocation failed for A" << std::endl;
        return;
    }

    error = hipMalloc((void**)&dB, bytes);
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS device memory allocation failed for B" << std::endl;
        return;
    }

    // Synchronize the device
    error = hipDeviceSynchronize();
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS device synchronization failed" << std::endl;
        return;
    }

    // Start a timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the memcpy operation on the device
    error = hipMemcpy(dB, dA, bytes, hipMemcpyDeviceToDevice);

    // Synchronize the device
    error = hipDeviceSynchronize();
    if (error != hipSuccess)
    {
        std::cout << "hip memcpy failed" << std::endl;
        return;
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Print the time taken
    std::cout << "Time taken for memcpy operation on " << bytes << " bytes: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Print the GB/s
    std::cout << "GB/s: " << (bytes) / (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e6) << std::endl;

    // Free the device memory
    error = hipFree(dA);
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS free failed" << std::endl;
        return;
    }
    error = hipFree(dB);
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS free failed" << std::endl;
        return;
    }

}


