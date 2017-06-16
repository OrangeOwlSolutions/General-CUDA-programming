#include<stdio.h>

#define N_ITERATIONS 8192

#include "Utilities.cuh"
#include "TimingGPU.cuh"

#define BLOCKSIZE	512

//#define DEBUG

/********************************************************/
/* KERNEL0 - NO INSTRUCTION LEVEL PARALLELISM (ILP = 0) */
/********************************************************/
__global__ void kernel0(int * __restrict__ d_a, const int * __restrict__ d_b, const int * __restrict__ d_c, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {

		int a = d_a[tid];
		int b = d_b[tid];
		int c = d_c[tid];

		for (unsigned int i = 0; i < N_ITERATIONS; i++) {
			a = a * b + c;
		}

		d_a[tid] = a;
	}

}

/*****************************************************/
/* KERNEL1 - INSTRUCTION LEVEL PARALLELISM (ILP = 2) */
/*****************************************************/
__global__ void kernel1(int * __restrict__ d_a, const int * __restrict__ d_b, const int * __restrict__ d_c, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N / 2) {

		int a1 = d_a[tid];
		int b1 = d_b[tid];
		int c1 = d_c[tid];

		int a2 = d_a[tid + N / 2];
		int b2 = d_b[tid + N / 2];
		int c2 = d_c[tid + N / 2];

		for (unsigned int i = 0; i < N_ITERATIONS; i++) {
			a1 = a1 * b1 + c1;
			a2 = a2 * b2 + c2;
		}

		d_a[tid] = a1;
		d_a[tid + N / 2] = a2;
	}

}

/*****************************************************/
/* KERNEL2 - INSTRUCTION LEVEL PARALLELISM (ILP = 4) */
/*****************************************************/
__global__ void kernel2(int * __restrict__ d_a, const int * __restrict__ d_b, const int * __restrict__ d_c, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N / 4) {

		int a1 = d_a[tid];
		int b1 = d_b[tid];
		int c1 = d_c[tid];

		int a2 = d_a[tid + N / 4];
		int b2 = d_b[tid + N / 4];
		int c2 = d_c[tid + N / 4];

		int a3 = d_a[tid + N / 2];
		int b3 = d_b[tid + N / 2];
		int c3 = d_c[tid + N / 2];

		int a4 = d_a[tid + 3 * N / 4];
		int b4 = d_b[tid + 3 * N / 4];
		int c4 = d_c[tid + 3 * N / 4];

		for (unsigned int i = 0; i < N_ITERATIONS; i++) {
			a1 = a1 * b1 + c1;
			a2 = a2 * b2 + c2;
			a3 = a3 * b3 + c3;
			a4 = a4 * b4 + c4;
		}

		d_a[tid] = a1;
		d_a[tid + N / 4] = a2;
		d_a[tid + N / 2] = a3;
		d_a[tid + 3 * N / 4] = a4;
	}

}

/********/
/* MAIN */
/********/
int main() {

	const int N = 8192 * 4;

	TimingGPU timerGPU;

	int *h_a = (int*)malloc(N*sizeof(int));
	int *h_a_result_host = (int*)malloc(N*sizeof(int));
	int *h_a_result_device = (int*)malloc(N*sizeof(int));
	int *h_b = (int*)malloc(N*sizeof(int));
	int *h_c = (int*)malloc(N*sizeof(int));

	for (int i = 0; i<N; i++) {
		h_a[i] = 2;
		h_b[i] = 1;
		h_c[i] = 2;
		h_a_result_host[i] = h_a[i];
		for (unsigned int k = 0; k < N_ITERATIONS; k++) {
			h_a_result_host[i] = h_a_result_host[i] * h_b[i] + h_c[i];
		}
	}

	int *d_a; gpuErrchk(cudaMalloc((void**)&d_a, N*sizeof(int)));
	int *d_b; gpuErrchk(cudaMalloc((void**)&d_b, N*sizeof(int)));
	int *d_c; gpuErrchk(cudaMalloc((void**)&d_c, N*sizeof(int)));

	gpuErrchk(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_c, h_c, N*sizeof(int), cudaMemcpyHostToDevice));

	/***********/
	/* KERNEL0 */
	/***********/
	timerGPU.StartCounter();
	kernel0 << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, d_c, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	// --- Remember: timing is in ms
	printf("GFlops = %f\n", (1.e-6)*((float)N*(float)N_ITERATIONS) / timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_a_result_device, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i<N; i++) if (h_a_result_device[i] != h_a_result_host[i]) { printf("Error at i=%i! Host = %i; Device = %i\n", i, h_a_result_host[i], h_a_result_device[i]); return 1; }

	/***********/
	/* KERNEL1 */
	/***********/
	gpuErrchk(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
	timerGPU.StartCounter();
	kernel1 << <iDivUp(N / 2, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, d_c, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	// --- Remember: timing is in ms
	printf("GFlops = %f\n", (1.e-6)*((float)N*(float)N_ITERATIONS) / timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_a_result_device, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i<N; i++) if (h_a_result_device[i] != h_a_result_host[i]) { printf("Error at i=%i! Host = %i; Device = %i\n", i, h_a_result_host[i], h_a_result_device[i]); return 1; }

	/***********/
	/* KERNEL2 */
	/***********/
	gpuErrchk(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
	timerGPU.StartCounter();
	kernel2 << <iDivUp(N / 4, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, d_c, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	// --- Remember: timing is in ms
	printf("GFlops = %f\n", (1.e-6)*(float)((float)N*(float)N_ITERATIONS) / timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_a_result_device, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i<N; i++) if (h_a_result_device[i] != h_a_result_host[i]) { printf("Error at i=%i! Host = %i; Device = %i\n", i, h_a_result_host[i], h_a_result_device[i]); return 1; }

	return 0;

}
