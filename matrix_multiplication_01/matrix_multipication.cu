// vector_add.cu --------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cmath>

// ---------------------------------------------------------------
// Simple error‑check macro (prints and exits on failure)
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "%s failed: %s\n", #call,               \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#define CUBLAS_CHECK(call)                                          \
    do {                                                            \
        cublasStatus_t status = (call);                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                       \
            fprintf(stderr, "%s failed: %d\n", #call, status);    \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)


void multiplyMatricesOnCPU(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// GPU kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row + k * N] * B[k + col * N];
        }
        C[row + col * N] = sum;
    }
}

void multiplyMatricesOnGPU(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, int N)
{
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i + j * N] = A[i][j];
            h_B[i + j * N] = B[i][j];
        }
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = h_C[i + j * N];
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void tensorMultiplication(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, int N) {
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i + j * N] = A[i][j];
            h_B[i + j * N] = B[i][j];
        }
    }

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
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        N,
        N,
        &alpha,
        d_A,
        N,
        d_B,
        N,
        &beta,
        d_C,
        N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = h_C[i + j * N];
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ---------------------------------------------------------------
int main()
{
    // Define matrix size and allocate host memory
    const int N = 256; // Size of the matrices (N x N)
    std::vector<std::vector<float>> matrixA(N, std::vector<float>(N));
    std::vector<std::vector<float>> matrixB(N, std::vector<float>(N));
    std::vector<std::vector<float>> cpuMatrixSumResult(N, std::vector<float>(N));
    std::vector<std::vector<float>> gpuMatrixSumResult(N, std::vector<float>(N));
    std::vector<std::vector<float>> tensorMatrixSumResult(N, std::vector<float>(N));

    
     // Initialize matrices with random values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrixA[i][j] = static_cast<float>(rand()) / RAND_MAX;
            matrixB[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    int gpuMUltiplicationTime = 0;
    int cpuMultiplicationTime = 0;
    int tensorMultiplicationTime = 0;
    bool areEqual = true;
    const float tolerance = 1e-4f;

    // Multiply matrices on CPU
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    multiplyMatricesOnCPU(matrixA, matrixB, cpuMatrixSumResult, N);
    auto cpuEndTime = std::chrono::high_resolution_clock::now();
    cpuMultiplicationTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEndTime - cpuStartTime).count();

    // Multiply matrices on GPU
    auto gpuStartTime = std::chrono::high_resolution_clock::now();
    multiplyMatricesOnGPU(matrixA, matrixB, gpuMatrixSumResult, N);
    auto gpuEndTime = std::chrono::high_resolution_clock::now();
    gpuMUltiplicationTime = std::chrono::duration_cast<std::chrono::milliseconds>(gpuEndTime - gpuStartTime).count();
    
    // Multiply using Tensor Cores (cuBLAS)
    auto tensorStartTime = std::chrono::high_resolution_clock::now();
    tensorMultiplication(matrixA, matrixB, tensorMatrixSumResult, N);
    auto tensorEndTime = std::chrono::high_resolution_clock::now();
    tensorMultiplicationTime = std::chrono::duration_cast<std::chrono::milliseconds>(tensorEndTime - tensorStartTime).count();

    // Compare results
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::fabs(cpuMatrixSumResult[i][j] - gpuMatrixSumResult[i][j]) > tolerance) {
                areEqual = false;
                break;
            } else if (std::fabs(cpuMatrixSumResult[i][j] - tensorMatrixSumResult[i][j]) > tolerance) {
                areEqual = false;
                break;
            }
        }
        if (!areEqual) {
            break;
        }
    }

    std::cout << "GPU Multiplication Time: " << gpuMUltiplicationTime << " ms" << std::endl;
    std::cout << "CPU Multiplication Time: " << cpuMultiplicationTime << " ms" << std::endl;
    std::cout << "Tensor Multiplication Time: " << tensorMultiplicationTime << " ms" << std::endl;

    if (areEqual) {
        std::cout << "Results are equal." << std::endl;
    } else {
        std::cout << "Results are not equal." << std::endl;
    }
}

