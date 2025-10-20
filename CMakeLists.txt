cmake_minimum_required(VERSION 4.0)

project(CUDA_Programming LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 20)

# Add executables
add_executable(hello_cuda 1_hello_cuda.cu)
add_executable(grid_block 2_grid_block.cu)
add_executable(organization_of_threads 3_organization_of_threads.cu)
add_executable(organization_of_threads2 4_organization_of_threads2.cu)
add_executable(unique_index_thread 5_unique_index_thread.cu)
add_executable(unique_index_thread_2d 6_unique_index_thread_2d.cu)
add_executable(memory_transfer 7_memory_transfer.cu)
add_executable(memory_transfer2 7_memory_transfer2.cu)
add_executable(array_sum 8_array_sum.cu)
add_executable(error_handling 9_error_handling.cu)
add_executable(device_query 10_device_query.cu)
add_executable(array_sum_time 11_array_sum_time.cu)

# Set target properties for CUDA
set_target_properties(
        hello_cuda grid_block organization_of_threads organization_of_threads2
        unique_index_thread unique_index_thread_2d memory_transfer memory_transfer2
        array_sum error_handling device_query array_sum_time

        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)
