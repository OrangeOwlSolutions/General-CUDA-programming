#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

#define BLOCKSIZE 1024

//#define DEBUG

/***********************************************/
/* MEMCPY1 - EACH THREAD COPIES ONE FLOAT ONLY */
/***********************************************/
__global__ void memcpy1(float *src, float *dst, unsigned int N)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {
		float a0 = src[tid];
		dst[tid] = a0;
	}
}

/*******************************************/
/* MEMCPY2 - EACH THREAD COPIES TWO FLOATS */
/*******************************************/
__global__ void memcpy2(float *src, float *dst, unsigned int N)
{
	const int tid = threadIdx.x + blockIdx.x * (2 * blockDim.x);

	if (tid < N) {
		float a0 = src[tid];
		float a1 = src[tid + blockDim.x];
		dst[tid] = a0;
		dst[tid + blockDim.x] = a1;
	}

}

/********************************************/
/* MEMCPY4 - EACH THREAD COPIES FOUR FLOATS */
/********************************************/
__global__ void memcpy4(float *src, float *dst, unsigned int N)
{
	const int tid = threadIdx.x + blockIdx.x * (4 * blockDim.x);

	if (tid < N) {

		float a0 = src[tid];
		float a1 = src[tid + blockDim.x];
		float a2 = src[tid + 2 * blockDim.x];
		float a3 = src[tid + 3 * blockDim.x];

		dst[tid] = a0;
		dst[tid + blockDim.x] = a1;
		dst[tid + 2 * blockDim.x] = a2;
		dst[tid + 3 * blockDim.x] = a3;

	}

}

/***********************************************/
/* MEMCPY4_2 - EACH THREAD COPIES FOUR FLOATS2 */
/***********************************************/
__global__ void memcpy4_2(float2 *src, float2 *dst, unsigned int N)
{
	const int tid = threadIdx.x + blockIdx.x * (4 * blockDim.x);

	if (tid < N / 2) {

		float2 a0 = src[tid];
		float2 a1 = src[tid + blockDim.x];
		float2 a2 = src[tid + 2 * blockDim.x];
		float2 a3 = src[tid + 3 * blockDim.x];

		dst[tid] = a0;
		dst[tid + blockDim.x] = a1;
		dst[tid + 2 * blockDim.x] = a2;
		dst[tid + 3 * blockDim.x] = a3;

	}

}

/********/
/* MAIN */
/********/
int main()
{
	//const int N = 131072;
	const int N = 8192 * 64;

	const int N_iter = 20;

	TimingGPU timerGPU;

	// --- Setting host data and memory space for result
	float* h_vect = (float*)malloc(N*sizeof(float));
	float* h_result = (float*)malloc(N*sizeof(float));
	for (int i = 0; i<N; i++) h_vect[i] = i;

	// --- Setting device data and memory space for result
	float* d_src;  gpuErrchk(cudaMalloc((void**)&d_src, N*sizeof(float)));
	float* d_dest1; gpuErrchk(cudaMalloc((void**)&d_dest1, N*sizeof(float)));
	float* d_dest2; gpuErrchk(cudaMalloc((void**)&d_dest2, N*sizeof(float)));
	float* d_dest4; gpuErrchk(cudaMalloc((void**)&d_dest4, N*sizeof(float)));
	float* d_dest4_2; gpuErrchk(cudaMalloc((void**)&d_dest4_2, N*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_src, h_vect, N*sizeof(float), cudaMemcpyHostToDevice));

	// --- Warmup
	for (int i = 0; i<N_iter; i++) memcpy1 << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_src, d_dest1, N);

	/***********/
	/* MEMCPY1 */
	/***********/
	timerGPU.StartCounter();
	for (int i = 0; i<N_iter; i++) {
		memcpy1 << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_src, d_dest1, N);
#ifdef DEGUB
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}
	printf("GB/s = %f\n", (1.e-6)*((float)N*(float)N_iter*(float)sizeof(float)) / timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_result, d_dest1, N*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i<N; i++) if (h_result[i] != h_vect[i]) { printf("Error at i=%i! Host = %f; Device = %f\n", i, h_vect[i], h_result[i]); return 0; }

	/***********/
	/* MEMCPY2 */
	/***********/
	timerGPU.StartCounter();
	for (int i = 0; i<N_iter; i++) {
		memcpy2 << <iDivUp(N / 2, BLOCKSIZE), BLOCKSIZE >> >(d_src, d_dest2, N);
#ifdef DEGUB
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}
	printf("GB/s = %f\n", (1.e-6)*((float)N*(float)N_iter*(float)sizeof(float)) / timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_result, d_dest2, N*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i<N; i++) if (h_result[i] != h_vect[i]) { printf("Error at i=%i! Host = %f; Device = %f\n", i, h_vect[i], h_result[i]); return 0; }

	/***********/
	/* MEMCPY4 */
	/***********/
	timerGPU.StartCounter();
	for (int i = 0; i<N_iter; i++) {
		memcpy4 << <iDivUp(N / 4, BLOCKSIZE), BLOCKSIZE >> >(d_src, d_dest4, N);
#ifdef DEGUB
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}
	printf("GB/s = %f\n", (1.e-6)*(float)((float)N*(float)N_iter*(float)sizeof(float)) / timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_result, d_dest4, N*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i<N; i++) if (h_result[i] != h_vect[i]) { printf("Error at i=%i! Host = %f; Device = %f\n", i, h_vect[i], h_result[i]); return 0; }

	/*************/
	/* MEMCPY4_2 */
	/*************/
	timerGPU.StartCounter();
	for (int i = 0; i<N_iter; i++) {
		memcpy4_2 << <iDivUp(N / 8, BLOCKSIZE), BLOCKSIZE >> >((float2*)d_src, (float2*)d_dest4_2, N);
#ifdef DEGUB
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}
	printf("GB/s = %f\n", (1.e-6)*(float)(N*N_iter*sizeof(float)) / timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_result, d_dest4_2, N*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i<N; i++) if (h_result[i] != h_vect[i]) { printf("Error at i=%i! Host = %f; Device = %f\n", i, h_vect[i], h_result[i]); return 0; }

	return 0;

}
