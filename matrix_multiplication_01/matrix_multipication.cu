// vector_add.cu --------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

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
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void multiplyMatricesOnGPU(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, int N)
{
    // Allocate device memory
    float* d_A, * d_B, * d_C;
    size_t bytes = N * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy data to device
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = A[i][j];
            h_B[i * N + j] = B[i][j];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Setup grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    float* h_C = new float[N * N];
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = h_C[i * N + j];
        }
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// ---------------------------------------------------------------
int main()
{
    // Define matrix size and allocate host memory
    const int N = 2000; // Size of the matrices (N x N)
    std::vector<std::vector<float>> matrixA(N, std::vector<float>(N));
    std::vector<std::vector<float>> matrixB(N, std::vector<float>(N));
    std::vector<std::vector<float>> cpuMatrixSumResult(N, std::vector<float>(N));
    std::vector<std::vector<float>> gpuMatrixSumResult(N, std::vector<float>(N));

    
     // Initialize matrices with random values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrixA[i][j] = static_cast<float>(rand()) / RAND_MAX;
            matrixB[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    int gpuMUltiplicationTime = 0;
    int cpuMultiplicationTime = 0;
    bool areEqual = true;

    // Multiply matrices on CPU
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    multiplyMatricesOnGPU(matrixA, matrixB, cpuMatrixSumResult, N);
    auto cpuEndTime = std::chrono::high_resolution_clock::now();
    cpuMultiplicationTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEndTime - cpuStartTime).count();

    // Multiply matrices on GPU
    auto gpuStartTime = std::chrono::high_resolution_clock::now();
    multiplyMatricesOnCPU(matrixA, matrixB, gpuMatrixSumResult, N);
    auto gpuEndTime = std::chrono::high_resolution_clock::now();
    gpuMUltiplicationTime = std::chrono::duration_cast<std::chrono::milliseconds>(gpuEndTime - gpuStartTime).count();

    // Compare results
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (cpuMatrixSumResult[i][j] != gpuMatrixSumResult[i][j]) {
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

    if (areEqual) {
        std::cout << "Results are equal." << std::endl;
    } else {
        std::cout << "Results are not equal." << std::endl;
    }
}

