#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> 

__global__ void helloFromGPU()
{ 
    printf("Hello CUDA\n"); 
}

int main() {
    
    dim3 grid(8); 
    dim3 block(4);

    helloFromGPU<<<grid, block>>>();
    
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}