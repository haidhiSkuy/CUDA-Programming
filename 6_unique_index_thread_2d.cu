#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void unique_gid_calc(int* input, int width) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    int gid = row * width + col; 

    if (gid < width * width) {
        printf("Block(%d,%d) Thread(%d,%d) => Row:%d Col:%d GID:%d Value:%d\n",
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
            row, col, gid, input[gid]);
    }
}



int main() {
    int h_data[10][10] = {
        {0, 10, 15, 28, 36, 40, 47, 56, 77},
        {1, 18, 42, 45, 48, 58, 60, 64, 90},
        {9, 10, 60, 67, 79, 81, 90, 92, 94, 97},
        {2, 17, 25, 27, 58, 60, 78, 83, 84, 90},
        {1, 32, 43, 44, 60, 79, 80, 85, 99},
        {7, 21, 37, 39, 49, 72, 79, 93, 95, 98},
        {7, 17, 18, 22, 23, 37, 42, 58, 69, 96},
        {7, 22, 25, 58, 64, 65, 70, 81, 86, 87},
        {23, 27, 36, 49, 77, 82, 89, 90, 100},
        {10, 21, 59, 61, 63, 68, 75, 79, 96, 97}
    };
    int array_size = sizeof(h_data) / sizeof(h_data[0][0]);
    int array_byte_size = sizeof(int) * array_size; 
    printf("Total array elements: %d\n", array_size); // 100
    printf("array byte size: %d\n", array_byte_size); // 400

    // Allocate data
    int* d_data; 
    cudaMalloc((void**)&d_data, array_byte_size);  
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    // 2 kolom, 5 baris per block = 10 thread/block
    dim3 block(2, 5); 
    // 5 block di X, 2 block di Y = total 10x10 thread
    dim3 grid(2, 5);

    unique_gid_calc <<<grid, block>>> (d_data, 10);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}