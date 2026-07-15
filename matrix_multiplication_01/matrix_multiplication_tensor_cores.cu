#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::cerr << #call << " failed: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << #call << " failed with cublas status " << status << std::endl; \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

void fillMatrix(std::vector<float>& values, int N) {
    for (int i = 0; i < N * N; ++i) {
        values[i] = static_cast<float>((rand() % 100) - 50) / 10.0f;
    }
}

void multiplyOnCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i + k * N] * B[k + j * N];
            }
            C[i + j * N] = sum;
        }
    }
}

int main() {
    const int N = 256;
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> h_A(N * N), h_B(N * N), h_C_cpu(N * N), h_C_gpu(N * N);
    fillMatrix(h_A, N);
    fillMatrix(h_B, N);

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        N,
        N,
        &alpha,
        d_A,
        CUDA_R_32F,
        N,
        d_B,
        CUDA_R_32F,
        N,
        &beta,
        d_C,
        CUDA_R_32F,
        N,
        CUDA_R_32F,
        CUBLAS_COMPUTE_32F_FAST_TF32));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    multiplyOnCPU(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double gpu_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    double max_diff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        max_diff = std::max(max_diff, std::fabs(h_C_gpu[i] - h_C_cpu[i]));
    }

    std::cout << "GPU time: " << gpu_ms << " ms" << std::endl;
    std::cout << "CPU time: " << cpu_ms << " ms" << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
