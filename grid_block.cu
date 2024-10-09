#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> 

__global__ void helloFromGPU()
{ 
    printf("Hello CUDA\n"); 
}

int main() {
    
    int nx, ny; 
    nx = 16; 
    ny = 4;

    dim3 block(8, 2); // 8 threads in x axis, 2 thread in y axis
    dim3 grid(nx/block.x, ny/block.y); // 2 block in x axis, 2 block in y axis

    helloFromGPU<<<grid, block>>>();
    
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}