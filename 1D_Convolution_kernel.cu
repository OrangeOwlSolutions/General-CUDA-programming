#include <stdio.h>
#include <stdlib.h>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

#define RG          10
#define BLOCKSIZE   8

/****************/
/* CPU FUNCTION */
/****************/
void h_convolution_1D(const float * __restrict__ h_Signal, const float * __restrict__ h_ConvKernel, float * __restrict__ h_Result_CPU, 
                      const int N, const int K) {

    for (int i = 0; i < N; i++) {

        float temp = 0.f;

        int N_start_point = i - (K / 2);

        for (int j = 0; j < K; j++) if (N_start_point + j >= 0 && N_start_point + j < N) {
            temp += h_Signal[N_start_point+ j] * h_ConvKernel[j];
        }

        h_Result_CPU[i] = temp;
    }
}

/********************/
/* BASIC GPU KERNEL */
/********************/
__global__ void d_convolution_1D_basic(const float * __restrict__ d_Signal, const float * __restrict__ d_ConvKernel, float * __restrict__ d_Result_GPU, 
                                       const int N, const int K) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.f;

    int N_start_point = i - (K / 2);

    for (int j = 0; j < K; j++) if (N_start_point + j >= 0 && N_start_point + j < N) {
        temp += d_Signal[N_start_point+ j] * d_ConvKernel[j];
    }

    d_Result_GPU[i] = temp;
}

/***************************/
/* GPU KERNEL WITH CACHING */
/***************************/
__global__ void d_convolution_1D_caching(const float * __restrict__ d_Signal, const float * __restrict__ d_ConvKernel, float * __restrict__ d_Result_GPU, 
                                         const int N, const int K) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float d_Tile[BLOCKSIZE];

    d_Tile[threadIdx.x] = d_Signal[i];
    __syncthreads();

    float temp = 0.f;

    int N_start_point = i - (K / 2);

    for (int j = 0; j < K; j++) if (N_start_point + j >= 0 && N_start_point + j < N) {

            if ((N_start_point + j >= blockIdx.x * blockDim.x) && (N_start_point + j < (blockIdx.x + 1) * blockDim.x))

                // --- The signal element is in the tile loaded in the shared memory
                temp += d_Tile[threadIdx.x + j - (K / 2)] * d_ConvKernel[j]; 

            else

                // --- The signal element is not in the tile loaded in the shared memory
                temp += d_Signal[N_start_point + j] * d_ConvKernel[j];

    }

    d_Result_GPU[i] = temp;
}

/********/
/* MAIN */
/********/
int main(){

    const int N = 15;           // --- Signal length
    const int K = 5;            // --- Convolution kernel length

    float *h_Signal         = (float *)malloc(N * sizeof(float));
    float *h_Result_CPU     = (float *)malloc(N * sizeof(float));
    float *h_Result_GPU     = (float *)malloc(N * sizeof(float));
    float *h_ConvKernel     = (float *)malloc(K * sizeof(float));

    float *d_Signal;        gpuErrchk(cudaMalloc(&d_Signal,     N * sizeof(float)));
    float *d_Result_GPU;    gpuErrchk(cudaMalloc(&d_Result_GPU, N * sizeof(float)));
    float *d_ConvKernel;    gpuErrchk(cudaMalloc(&d_ConvKernel, K * sizeof(float)));

    for (int i=0; i < N; i++) { h_Signal[i] = (float)(rand() % RG); }

    for (int i=0; i < K; i++) { h_ConvKernel[i] = (float)(rand() % RG); }

    gpuErrchk(cudaMemcpy(d_Signal,      h_Signal,       N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_ConvKernel,  h_ConvKernel,   K * sizeof(float), cudaMemcpyHostToDevice));

    h_convolution_1D(h_Signal, h_ConvKernel, h_Result_CPU, N, K);

    d_convolution_1D_basic<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_Signal, d_ConvKernel, d_Result_GPU, N, K);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_Result_GPU, d_Result_GPU, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) if (h_Result_CPU[i] != h_Result_GPU[i]) {printf("mismatch2 at %d, cpu: %d, gpu %d\n", i, h_Result_CPU[i], h_Result_GPU[i]); return 1;}

    printf("Test basic passed\n");

    d_convolution_1D_caching<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_Signal, d_ConvKernel, d_Result_GPU, N, K);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_Result_GPU, d_Result_GPU, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) if (h_Result_CPU[i] != h_Result_GPU[i]) {printf("mismatch2 at %d, cpu: %d, gpu %d\n", i, h_Result_CPU[i], h_Result_GPU[i]); return 1;}

    printf("Test caching passed\n");

    return 0;
}
