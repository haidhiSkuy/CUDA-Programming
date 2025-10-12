#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

void initialize_data(float* ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));
    for (size_t i = 1; i < size; i++) 
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void sum_array(float* a, float* b, float* c)
{
    int i = threadIdx.x; 
    c[i] = a[i] + b[i]; 
    printf("a =%f  b = %f c = %f \n",a[i],b[i],c[i]);
}

int main()
{
    int element_count = 32;
    size_t number_bytes = element_count * sizeof(float); // 128

    float *host_a, *host_b, *host_ref, *device_ref;

    host_a = (float*)malloc(number_bytes);
    host_b = (float*)malloc(number_bytes);

    host_ref = (float*)malloc(number_bytes);
    device_ref = (float*)malloc(number_bytes); 

    initialize_data(host_a, element_count);
    initialize_data(host_b, element_count);

    memset(host_ref, 0, number_bytes);
    memset(device_ref, 0, number_bytes);

    float *device_a, *device_b, *device_c; 
    cudaMalloc((float**)&device_a, number_bytes);
    cudaMalloc((float**)&device_b, number_bytes); 
    cudaMalloc((float**)&device_c, number_bytes); 

    cudaMemcpy(device_a, host_a, number_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, number_bytes, cudaMemcpyHostToDevice); 
    
    dim3 block(element_count);
    dim3 grid(element_count / block.x); 

    sum_array <<<grid, block>>> (device_a, device_b, device_c);


    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_ref);
    free(device_ref); 

    return 0;
}