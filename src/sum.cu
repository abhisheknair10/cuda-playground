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

#define TILE 1024

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

    for (int i = 0; i < row * col; i++) {
        matrix[i] = i;
    }

    return matrix;
}

__global__ void sum_reduction_kernel(float *res, float *d_v, int v_len) {
    __shared__ float buffer[TILE];

    float _res = 0.0f;
    for (int i = 0; i < (v_len + TILE - 1) / TILE; i++) {
        if (i * TILE + threadIdx.x < v_len) {
            _res += d_v[i * TILE + threadIdx.x];
        }
    }
    __syncthreads();

    buffer[threadIdx.x] = _res;
    for (int offset = TILE / 2; offset >= 32; offset /= 2) {
        if (threadIdx.x < offset) {
            buffer[threadIdx.x] += buffer[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        float val = buffer[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) *res = val;
    }
}

void sum_reduction(float *res, float *d_v, int v_len) {
    dim3 block(TILE);
    dim3 grid(1);

    sum_reduction_kernel<<<grid, block>>>(res, d_v, v_len);
    cudaDeviceSynchronize();
}

int main() {
    int len = 2048;

    float *h_v = init_matrix(1, len);

    float *d_v, *d_res, h_res;
    cudaMalloc((void **)&d_v, sizeof(float) * len);
    cudaMalloc((void **)&d_res, sizeof(float));

    cudaMemcpy(d_v, h_v, sizeof(float) * len, cudaMemcpyHostToDevice);

    sum_reduction(d_res, d_v, len);

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    display_matrix(d_v, h_v, 1, len);
    printf("Sum: %.2f\n", h_res);

    CHECK_CUDA_ERROR();

    return 0;
}