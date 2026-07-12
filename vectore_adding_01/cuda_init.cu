// vector_add.cu --------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

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

// ---------------------------------------------------------------
// Kernel: each thread adds one element of a and b into c
__global__ void vectorAdd(const float* a, const float* b,
                          float* c, float* res, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    if (i < n) res[i] = a[i] + b[i] + c[i];
}

// ---------------------------------------------------------------
int main()
{
    // 1️⃣ Choose device 0 (first GPU) and print its name
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Running on %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);

    // 2️⃣ Problem size
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // 3️⃣ Host vectors (managed by std::vector → auto‑free)
    std::vector<float> h_a(N), h_b(N), h_c(N), h_res(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);       // 0,1,2,...
        h_b[i] = static_cast<float>(i * 2);   // 0,2,4,...
        h_c[i] = static_cast<float>(i * 3);   // 0,3,6,...
    }

    // 4️⃣ Allocate device memory
    float *d_a, *d_b, *d_c, *d_res;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_res, bytes));

    // 5️⃣ Copy inputs from host → device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice));

    // 6️⃣ Launch kernel (256 threads per block is a common choice)
    const int TPB = 256;
    const int BPG = (N + TPB - 1) / TPB;          // ceil(N/TPB)
    vectorAdd<<<BPG, TPB>>>(d_a, d_b, d_c, d_res, N);
    CUDA_CHECK(cudaGetLastError());               // catch launch errors
    CUDA_CHECK(cudaDeviceSynchronize());          // wait for kernel

    // 7️⃣ Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, bytes, cudaMemcpyDeviceToHost));

    // 8️⃣ Verify (use a tolerance for floating‑point compare)
    const float eps = 1e-5f;
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i] + h_c[i];
        if (fabsf(h_res[i] - expected) > eps) {
            fprintf(stderr, "Mismatch @%d: %f vs %f\n", i, h_res[i], expected);
            ok = false;
            break;
        }
    }

    if (ok) {
        printf("Success! Sample: c[0]=%f c[1]=%f c[2]=%f\n",
               h_res[0], h_res[1], h_res[2]);
    }

    // 9️⃣ Clean up device memory (host vectors free automatically)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_res);
    cudaDeviceReset();   // optional, helps profiling tools

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
