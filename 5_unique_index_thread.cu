#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h> 
#include <stdlib.h>

__global__ void unique_idx_calc_threadIdx(int* input)
{
    int tid = threadIdx.x;
	printf("threadIdx : %d, value : %d \n", tid, input[tid]);
} 

__global__ void unique_gid_calc(int* input)
{
    int tid = threadIdx.x;
    int gid = tid + (blockIdx.x * blockDim.x); 
    printf("blockIdx.x : %d, threadIdx.x : %d, gid : %d, value : %d \n",
		blockIdx.x, tid, gid, input[gid]);

}


int main()
{
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 87, 45, 23, 12, 342, 56, 44, 99};
    int array_size = sizeof(h_data) / sizeof(h_data[0]); // 16
    int array_byte_size = sizeof(int) * array_size; // 64
    printf("array byte size: %d\n", array_byte_size);
    printf("Total array elements: %d\n", array_size);
    printf("Array elements: ");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n============\n");

    int* d_data;
    // allocates memory in GPU
	cudaMalloc((void**)&d_data, array_byte_size); 
    // Transfer data to GPU
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);


    dim3 block(4);
    dim3 grid(4); 

    unique_gid_calc <<<grid, block>>> (d_data);
    
	cudaDeviceSynchronize();
	cudaDeviceReset();

    return 0;

}