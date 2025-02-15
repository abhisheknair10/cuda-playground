#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

#define WMMA_M 32
#define WMMA_N 32
#define WMMA_K 32

using namespace nvcuda;

// matmul kernel
__global__ void matmul_tensor_core(half *A, half *B, float *C, int N);

half *init_matrix_half(int N, int reverse = 0) {
    half *mat = (half *)malloc(N * N * sizeof(half));

    for (int i = 0; i < N * N; i++) {
        mat[i] = reverse == 0 ? __float2half(i % 1000) : __float2half(((N * N) - i) % 1000);
    }

    return mat;
}

float *init_matrix_float(int N, int reverse = 0) {
    float *mat = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        mat[i] = reverse == 0 ? i % 1000 : ((N * N) - i) % 1000;
    }

    return mat;
}

void matprint(float *mat, int N) {
    for (int i = 0; i < N * N; i++) {
        if (i % N == 0) {
            printf("\n");
        }
        
        printf("%.0f, ", mat[i]);
    }
}

int main() {
    int N = 32;
    half *d_A, *d_B;
    float *d_C;

    half *A = init_matrix_half(N, 0);
    half *B = init_matrix_half(N, 1);
    float *C = init_matrix_float(N, 0);

    cudaMalloc((void **)&d_A, N * N * sizeof(half));
    cudaMalloc((void **)&d_B, N * N * sizeof(half));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, A, sizeof(half) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(half) * N * N, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(N / 32, N / 32);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_tensor_core<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken by kernel: %.3f ms\n", elapsedTime);

    cudaMemcpy(C, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    matprint(C, N);

    cudaDeviceSynchronize();

    return 0;
}

__global__ void matmul_tensor_core(half *A, half *B, float *C, int N) {
    // Declare fragment containers for A, B, and C
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accum_frag;

    // Load matrix tiles into WMMA fragments
    wmma::load_matrix_sync(a_frag, A, N);
    wmma::load_matrix_sync(b_frag, B, N);

    // Initialize accumulator fragment
    wmma::fill_fragment(accum_frag, 0.0f);

    // Perform matrix multiplication using Tensor Cores
    wmma::mma_sync(accum_frag, a_frag, b_frag, accum_frag);

    // Store the result back to global memory
    wmma::store_matrix_sync(C, accum_frag, N, wmma::mem_row_major);
}