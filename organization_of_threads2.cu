#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>  

__global__ void printThreadIds()
{ 
    printf("blockIdx.x: %d | blockIdx.y: %d | blockDim.x: %d | blockDim.y: %d | gridDim.x: %d | gridDim.y: %d |\n", 
            blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y
        ); 
} 

int main()
{ 
    int nx, ny; 
    nx = 16; 
    ny = 16; 

    dim3 block(8, 8); 
    dim3 grid(nx / block.x, ny / block.y);  // 2, 2

    printThreadIds<<<grid, block>>>();
    
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}