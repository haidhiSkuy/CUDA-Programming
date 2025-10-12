#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cstring>
#include <cstdlib> // Untuk rand() dan srand()

/**
 * @brief Menginisialisasi array float dengan nilai acak.
 * * Nilai acak dihasilkan dalam rentang tertentu.
 * * @param ip Pointer ke array float yang akan diinisialisasi.
 * @param size Jumlah elemen dalam array.
 */
void initialize_data(float* ip, int size)
{
    // Menggunakan waktu saat ini sebagai seed untuk generator angka acak
    time_t t;
    srand((unsigned) time(&t));
    
    // Looping untuk mengisi setiap elemen array
    for (size_t i = 0; i < size; i++)
    {
        // Menghasilkan nilai acak, di-masking dengan 0xFF (0-255), 
        // lalu dibagi 10.0f agar hasilnya berupa float dan tidak terlalu besar
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

/**
 * @brief Kernel CUDA untuk menjumlahkan dua array elemen-per-elemen.
 * * Fungsi ini dijalankan di GPU oleh setiap thread. Setiap thread 
 * bertanggung jawab untuk menjumlahkan satu pasang elemen.
 * * @param a Pointer ke array input pertama di memori device (GPU).
 * @param b Pointer ke array input kedua di memori device (GPU).
 * @param c Pointer ke array output (hasil) di memori device (GPU).
 */
__global__ void sum_array(float* a, float* b, float* c)
{
    // Mengambil ID thread saat ini dalam blok. 
    // Karena kita hanya menggunakan 1 blok, ini sama dengan indeks elemen.
    int i = threadIdx.x; 
    
    // Operasi inti: penjumlahan elemen ke-i
    c[i] = a[i] + b[i]; 
    
    // Debugging: mencetak hasil penjumlahan langsung dari thread GPU
    printf("a = %f  b = %f  c = %f\n", a[i], b[i], c[i]);
}

/**
 * @brief Fungsi utama program (Host code).
 * * Mengatur alokasi memori, inisialisasi data, transfer data ke GPU, 
 * meluncurkan kernel, dan mengambil hasil kembali.
 */
int main()
{
    // --- Pengaturan Ukuran dan Alokasi Memori di Host (CPU) ---
    int element_count = 32; // Jumlah elemen array
    size_t number_bytes = element_count * sizeof(float); // Total ukuran memori dalam byte

    // Deklarasi pointer untuk memori Host (CPU)
    float *host_a, *host_b, *host_ref, *device_ref;
    
    // Alokasi memori Host menggunakan malloc()
    host_a = (float*)malloc(number_bytes); // Array input A
    host_b = (float*)malloc(number_bytes); // Array input B
    host_ref = (float*)malloc(number_bytes); // Array untuk hasil referensi (tidak dipakai di sini, tapi bagus untuk verifikasi)
    device_ref = (float*)malloc(number_bytes); // Array untuk menampung hasil yang di-copy kembali dari GPU

    // --- Inisialisasi Data ---
    initialize_data(host_a, element_count); // Isi A dengan data acak
    initialize_data(host_b, element_count); // Isi B dengan data acak

    // Mengosongkan (set ke 0) memori hasil di Host sebelum dipakai
    memset(host_ref, 0, number_bytes);
    memset(device_ref, 0, number_bytes);

    // --- Alokasi Memori di Device (GPU) ---
    float *device_a, *device_b, *device_c; 
    
    // Alokasi memori di Device (GPU) menggunakan cudaMalloc()
    cudaMalloc((float**)&device_a, number_bytes); // Device array A
    cudaMalloc((float**)&device_b, number_bytes); // Device array B
    cudaMalloc((float**)&device_c, number_bytes); // Device array C (Hasil)

    // --- Transfer Data Host ke Device ---
    // Menyalin array A dan B dari Host ke Device
    cudaMemcpy(device_a, host_a, number_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, number_bytes, cudaMemcpyHostToDevice); 
    
    // --- Konfigurasi dan Peluncuran Kernel ---
    // Konfigurasi Blok: 32 thread per blok
    dim3 block(element_count); 
    
    // Konfigurasi Grid: 1 blok (cuma satu blok aja, karena threadnya udah 32)
    dim3 grid(1); 

    // Peluncuran Kernel: memanggil fungsi __global__ sum_array di GPU
    sum_array<<<grid, block>>>(device_a, device_b, device_c);

    // Sinkronisasi: menunggu semua thread kernel selesai dieksekusi di GPU
    cudaDeviceSynchronize(); 

    // --- Transfer Data Device ke Host ---
    // Menyalin hasil (Array C) dari Device kembali ke Host (ke device_ref)
    cudaMemcpy(device_ref, device_c, number_bytes, cudaMemcpyDeviceToHost);

    // --- Verifikasi dan Pencetakan Hasil di Host ---
    printf("\nHasil penjumlahan di host:\n");
    for (int i = 0; i < element_count; i++)
    {
        // Mencetak elemen A, elemen B, dan hasilnya (elemen C yang ada di device_ref)
        printf("%2d: %f + %f = %f\n", i, host_a[i], host_b[i], device_ref[i]);
    }

    // --- Pembersihan Memori ---
    // Membebaskan memori yang dialokasikan di Device (GPU)
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    
    // Membebaskan memori yang dialokasikan di Host (CPU)
    free(host_a);
    free(host_b);
    free(host_ref);
    free(device_ref);

    return 0;
}