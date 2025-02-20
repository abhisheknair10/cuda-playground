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

__global__ void naive_matmul_kernel(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = idx / n;
    int col = idx % n;

    if (row >= m || col >= n) return;

    float _res = 0.0;
    for (int i = 0; i < k; i++) {
        _res += d_A[row * k + i] * d_B[i * n + col];
    }

    d_C[idx] = _res;
}

__global__ void tiled_matmul_kernel(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    __shared__ float shmem_A[TILE * TILE];
    __shared__ float shmem_B[TILE * TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float _res = 0.0f;
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        if (row < m && t * TILE + threadIdx.x < k) {
            int Aidx = row * k + t * TILE + threadIdx.x;
            shmem_A[threadIdx.y * TILE + threadIdx.x] = d_A[Aidx];
        } else {
            shmem_A[threadIdx.y * TILE + threadIdx.x] = 0.0;
        }

        if (col < n && (t * TILE + threadIdx.y) < k) {
            int Bidx = (t * TILE + threadIdx.y) * n + col;
            shmem_B[threadIdx.y * TILE + threadIdx.x] = d_B[Bidx];
        } else {
            shmem_B[threadIdx.y * TILE + threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            _res += shmem_A[threadIdx.y * TILE + i] * shmem_B[i * TILE + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        d_C[row * n + col] = _res;
    }
}

void naive_matmul(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    int THREADS_PER_BLOCK = 1024;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(((m * n) + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);
    naive_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
}

void tiled_matmul(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    tiled_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
}

int main() {
    int m = 4;
    int n = 4;
    int k = 6;

    float *h_A = init_matrix(m, k);
    float *h_B = init_matrix(k, n);
    float *h_C = init_matrix(m, n);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * m * k);
    cudaMalloc((void **)&d_B, sizeof(float) * n * k);
    cudaMalloc((void **)&d_C, sizeof(float) * m * n);

    cudaMemcpy(d_A, h_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    display_matrix(d_A, h_A, m, k);
    display_matrix(d_B, h_B, k, n);

    // printf("Starting Naive Kernel\n");
    naive_matmul(d_A, d_B, d_C, m, n, k);
    display_matrix(d_C, h_C, m, n);
    // printf("Finished Naive Kernel\n");

    // printf("Starting Tiled Kernel\n");
    tiled_matmul(d_A, d_B, d_C, m, n, k);
    display_matrix(d_C, h_C, m, n);
    // printf("Finished Tiled Kernel\n");

    CHECK_CUDA_ERROR();

    return 0;
}