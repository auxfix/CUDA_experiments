#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit + 1 < 5) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << " " << units[unit];
    return oss.str();
}

int estimateFp32CoresPerSM(int major, int minor) {
    switch (major) {
        case 2: return 32;   // Fermi
        case 3: return 192;  // Kepler
        case 5: return 128;  // Maxwell
        case 6: return 64;   // Pascal
        case 7: return 64;   // Volta / Turing
        case 8: return 64;   // Ampere
        case 9: return 64;   // Hopper
        default: return 64;
    }
}

int estimateTensorCoresPerSM(int major, int minor) {
    if (major >= 7) return 8;
    return 0;
}

int estimateRtCoresPerSM(int major, int minor) {
    if (major >= 8) return 2;
    if (major == 7) return 2;
    return 0;
}

__global__ void helloKernel() {
    printf("Hello from GPU thread! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}

int getDeviceAttributeOrMinusOne(int device, cudaDeviceAttr attr) {
    int value = -1;
    cudaError_t err = cudaDeviceGetAttribute(&value, attr, device);
    if (err != cudaSuccess) {
        return -1;
    }
    return value;
}

}  // namespace

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices were found.\n";
        return 1;
    }

    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "=== CUDA Hello World ===\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "SMs (streaming multiprocessors): " << props.multiProcessorCount << "\n";
    std::cout << "Warp size: " << props.warpSize << "\n";
    std::cout << "Max threads per block: " << props.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per SM: " << props.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Global memory: " << formatBytes(props.totalGlobalMem) << "\n";
    std::cout << "Shared memory per block: " << formatBytes(props.sharedMemPerBlock) << "\n";
    std::cout << "Shared memory per SM: " << formatBytes(props.sharedMemPerMultiprocessor) << "\n";
    std::cout << "L2 cache size: " << formatBytes(props.l2CacheSize) << "\n";
    std::cout << "Memory bus width: " << props.memoryBusWidth << " bits\n";
    const int clockRateKHz = getDeviceAttributeOrMinusOne(device, cudaDevAttrClockRate);
    const int memoryClockRateKHz = getDeviceAttributeOrMinusOne(device, cudaDevAttrMemoryClockRate);
    std::cout << "Memory clock rate: " << (memoryClockRateKHz >= 0 ? memoryClockRateKHz / 1000000.0 : -1.0) << " GHz\n";
    std::cout << "Clock rate: " << (clockRateKHz >= 0 ? clockRateKHz / 1000000.0 : -1.0) << " GHz\n";
    std::cout << "Registers per block: " << props.regsPerBlock << "\n";
    std::cout << "Max grid size: (" << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << ")\n";
    std::cout << "Approx. FP32 cores per SM: " << estimateFp32CoresPerSM(props.major, props.minor) << "\n";
    std::cout << "Approx. FP32 cores total: " << estimateFp32CoresPerSM(props.major, props.minor) * props.multiProcessorCount << "\n";
    std::cout << "Approx. tensor cores per SM: " << estimateTensorCoresPerSM(props.major, props.minor) << "\n";
    std::cout << "Approx. RT cores per SM: " << estimateRtCoresPerSM(props.major, props.minor) << "\n";
    std::cout << "CUDA runtime version: " << CUDART_VERSION << "\n";

    std::cout << "Launching a tiny kernel...\n";
    helloKernel<<<1, 1>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;
}
