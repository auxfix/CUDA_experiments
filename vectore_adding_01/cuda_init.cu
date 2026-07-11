#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    std::printf("CUDA devices found: %d\n", deviceCount);
    if (deviceCount == 0) {
        std::printf("No CUDA devices available.\n");
        return EXIT_SUCCESS;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    std::printf("Using device 0: %s\n", prop.name);
    std::printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    std::printf("  Total global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));

    const int N = 1024;
    size_t size = N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for d_a: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for d_b: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        return EXIT_FAILURE;
    }
    err = cudaMalloc(&d_c, size);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for d_c: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed for d_a: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed for d_b: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed for d_c: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            std::fprintf(stderr, "Result mismatch at index %d: host %f device %f\n", i, expected, h_c[i]);
            success = false;
            break;
        }
    }

    if (success) {
        std::printf("Vector add completed successfully.\n");
        std::printf("Sample output: c[0]=%f c[1]=%f c[2]=%f\n", h_c[0], h_c[1], h_c[2]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
