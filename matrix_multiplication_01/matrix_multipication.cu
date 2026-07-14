// vector_add.cu --------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

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

void multiplyMatricesOnGPU(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, int N)
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

// ---------------------------------------------------------------
int main()
{
    // Define matrix size and allocate host memory
    const int N = 1024; // Size of the matrices (N x N)
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

    // Multiply matrices on CPU
    multiplyMatricesOnCPU(matrixA, matrixB, cpuMatrixSumResult, N);

    // Multiply matrices on GPU
    multiplyMatricesOnGPU(matrixA, matrixB, gpuMatrixSumResult, N);

    // Compare results
    bool areEqual = true;
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

    if (areEqual) {
        std::cout << "The results are equal." << std::endl;
    }
    else {
        std::cout << "The results are not equal." << std::endl;
    }


}

