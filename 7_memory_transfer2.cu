#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

// ============================================================
// KERNEL: Fungsi ini dijalankan di GPU secara paralel
// __global__ artinya function ini dipanggil dari CPU, tapi dijalankan di GPU
// ============================================================
__global__ void mem_trs_test(int* input, int size)
{
    // Hitung Global Thread ID = index unik tiap thread di seluruh grid
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Cek biar thread gak keluar dari batas array
    if (gid < size) {
        // Print data dari GPU (dijalankan dari tiap thread)
        printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
    }
}

// ============================================================
// MAIN PROGRAM (Host code - dijalankan di CPU)
// ============================================================
int main(){
    int size = 150;                       // Jumlah elemen array
    int byte_size = size * sizeof(int);   // Total byte yang dibutuhkan

    // ======================
    // 1️⃣ ALOKASI DI HOST (CPU)
    // ======================
    int* h_input;
    h_input = (int*)malloc(byte_size);    // Alokasi memori di CPU

    // Inisialisasi random seed
    time_t t;
    srand((unsigned)time(&t));

    // Isi array dengan angka random 0–255
    for (int i = 0; i < size; i++)
    {
        h_input[i] = int(rand() & 0xff);
    }

    // ======================
    // 2️⃣ ALOKASI DI DEVICE (GPU)
    // ======================
    int* d_input;
    cudaMalloc((void**)&d_input, byte_size);   // Minta memori di GPU sebesar byte_size

    // ======================
    // 3️⃣ TRANSFER DATA Host → Device
    // ======================
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
    // → Copy data array dari RAM (h_input) ke VRAM GPU (d_input)
    // → cudaMemcpyHostToDevice = arah transfernya

    // ======================
    // 4️⃣ KONFIGURASI GRID & BLOCK
    // ======================
    dim3 block(32);    // 1 blok berisi 32 thread
    dim3 grid(5);      // 5 blok total
    // Total thread = grid.x * block.x = 5 * 32 = 160 thread
    // Tapi array cuma 150 elemen, jadi 10 thread terakhir bakal gagal lewat (dicegah if)

    // ======================
    // 5️⃣ JALANKAN KERNEL DI GPU
    // ======================
    mem_trs_test <<<grid, block>>> (d_input, size);
    // <<<grid, block>>> = konfigurasi paralel CUDA
    // GPU bakal spawn 160 thread yang masing-masing ngeprint 1 elemen

    // Tunggu semua thread GPU selesai (sinkronisasi)
    cudaDeviceSynchronize();

    // ======================
    // 6️⃣ DEALOKASI MEMORY
    // ======================
    cudaFree(d_input); // Hapus alokasi GPU
    free(h_input);     // Hapus alokasi CPU

    // Reset GPU (optional tapi recommended)
    cudaDeviceReset();

    return 0;
}
