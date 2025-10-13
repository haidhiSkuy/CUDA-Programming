#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cstring>
#include <cstdlib> 


/**
 * @brief Macro untuk melakukan pengecekan error CUDA dengan mudah.
 *
 * Macro ini secara otomatis memanggil fungsi 'gpuAssert()' dan menambahkan
 * informasi file serta nomor baris di mana error terjadi.
 *
 * Contoh penggunaan:
 *     gpuErrorCheck(cudaMalloc((void**)&ptr, size));
 */
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * @brief Fungsi untuk memeriksa hasil operasi CUDA dan menampilkan pesan error.
 *
 * @param code  Nilai hasil pengembalian (return value) dari fungsi CUDA (cudaError_t)
 * @param file  Nama file sumber di mana error terjadi (otomatis diisi oleh __FILE__)
 * @param line  Nomor baris di mana error terjadi (otomatis diisi oleh __LINE__)
 * @param abort Jika true, program akan langsung dihentikan (exit) ketika terjadi error
 *
 * Fungsi ini akan:
 *  - Mengecek apakah `code` sama dengan `cudaSuccess`
 *  - Jika tidak, mencetak pesan error ke `stderr` yang berisi:
 *      * Deskripsi error dari `cudaGetErrorString(code)`
 *      * Nama file tempat error terjadi
 *      * Nomor baris error
 *  - Jika `abort == true`, maka program akan langsung keluar dengan kode error.
 */
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){
    if (code != cudaSuccess) // jika hasil eksekusi CUDA tidak sukses
    {   
        // Menampilkan pesan error lengkap ke terminal (stderr)
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        // Jika mode abort aktif, hentikan program agar tidak lanjut dengan state GPU rusak
        if (abort) exit(code);
    }
}

void initialize_data(float* ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));
    
    for (size_t i = 0; i < size; i++)
    {

        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}


__global__ void sum_array(float* a, float* b, float* c)
{
    int i = threadIdx.x; 
    c[i] = a[i] + b[i]; 
    printf("a = %f  b = %f  c = %f\n", a[i], b[i], c[i]);
}


int main()
{
    int element_count = 32; // Jumlah elemen array
    size_t number_bytes = element_count * sizeof(float);

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
    
    gpuErrorCheck(cudaMalloc((float**)&device_a, number_bytes)); 
    gpuErrorCheck(cudaMalloc((float**)&device_b, number_bytes));
    gpuErrorCheck(cudaMalloc((float**)&device_c, number_bytes));

    gpuErrorCheck(cudaMemcpy(device_a, host_a, number_bytes, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(device_b, host_b, number_bytes, cudaMemcpyHostToDevice)); 

    dim3 block(element_count); 
    
    dim3 grid(1); 

    sum_array<<<grid, block>>>(device_a, device_b, device_c);

    cudaDeviceSynchronize(); 

    gpuErrorCheck(cudaMemcpy(device_ref, device_c, number_bytes, cudaMemcpyDeviceToHost));

    printf("\nHasil penjumlahan di host:\n");
    for (int i = 0; i < element_count; i++)
    {
        printf("%2d: %f + %f = %f\n", i, host_a[i], host_b[i], device_ref[i]);
    }

    gpuErrorCheck(cudaFree(device_a));
    gpuErrorCheck(cudaFree(device_b));
    gpuErrorCheck(cudaFree(device_c));
    
    free(host_a);
    free(host_b);
    free(host_ref);
    free(device_ref);

    return 0;
}