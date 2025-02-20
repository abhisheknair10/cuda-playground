#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA_ERROR()                                       \
    {                                                            \
        cudaError_t err = cudaGetLastError();                    \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s in file '%s' in line %i\n",   \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

#define TILE 2

void display_matrix(float *d_mat, float *h_mat, int row, int col) {
    cudaMemcpy(h_mat, d_mat, sizeof(float) * row * col, cudaMemcpyDeviceToHost);

    for (int i = 0; i < row; i++) {
        printf("[");
        for (int j = 0; j < col; j++) {
            printf("%.2f", h_mat[i * col + j]);
            if (j < col - 1) {
                printf(", ");
            }
        }

        printf("]");
        if (i < row - 1) {
            printf(", ");
        }
        printf("\n");
    }

    printf("\n");
}

float *init_matrix(int row, int col) {
    float *matrix = (float *)malloc(sizeof(float) * row * col);

    float mid = (row * col) / 2;
    for (int i = 0; i < row * col; i++) {
        matrix[i] = i - mid;
    }

    return matrix;
}

__global__ void transpose_kernel(float *d_A, float *d_B, int m, int n) {
    __shared__ float shared_mem[TILE * TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row < m && col < n) {
        shared_mem[threadIdx.y * TILE + threadIdx.x] = d_A[row * n + col];
    }
    __syncthreads();

    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    if (row < n && col < m) {
        d_B[row * m + col] = shared_mem[threadIdx.x * TILE + threadIdx.y];
    }
}

void transpose(float *d_A, float *d_B, int m, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    transpose_kernel<<<grid, block>>>(d_A, d_B, m, n);
    cudaDeviceSynchronize();
}

int main() {
    int m = 4;
    int n = 6;

    float *h_A = init_matrix(m, n);
    float *h_B = init_matrix(n, m);

    float *d_A, *d_B;
    cudaMalloc((void **)&d_A, sizeof(float) * m * n);
    cudaMalloc((void **)&d_B, sizeof(float) * n * m);

    cudaMemcpy(d_A, h_A, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    display_matrix(d_A, h_A, m, n);
    transpose(d_A, d_B, m, n);
    display_matrix(d_B, h_B, n, m);

    CHECK_CUDA_ERROR();

    return 0;
}