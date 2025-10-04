#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> 

__global__ void helloFromGPU()
{ 
    printf("Hello CUDA\n"); 
}

int main() {
    // 2 blocks, 10 threads
    helloFromGPU<<<2, 10>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}