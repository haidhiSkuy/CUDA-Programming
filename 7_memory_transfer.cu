#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void mem_trs_test(int* input)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; 
    printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
}

int main(){
    int size = 128; 
    int byte_size = size * sizeof(int); 

    // Host input
    int* h_input;
    h_input = (int*)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_input[i] = int(rand() & 0xff);
    }

    // Device input
    int* d_input;

    /* 
    Minta alokasi byte_size byte di device. cudaMalloc expect void**, makanya cast (void**)&d_input.
    Setelah sukses, d_input berisi alamat di memory GPU. CPU nggak bisa baca isi memory itu langsung.
    */
    cudaMalloc((void**)&d_input, byte_size);

    /* 
    Copy byte_size byte dari h_input (host/CPU memory) ke d_input (device/GPU memory). 
    cudaMemcpyHostToDevice nunjukin arah transfer.
    */
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid(2);

    mem_trs_test <<<grid, block>>> (d_input);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    free(h_input);

    cudaDeviceReset();
    return 0;
}