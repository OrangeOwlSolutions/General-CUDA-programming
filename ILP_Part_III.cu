#include<stdio.h>
#include<iostream>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

#define BLOCKSIZE	 512

//#define DEBUG

/****************************************/
/* INSTRUCTION LEVEL PARALLELISM KERNEL */
/****************************************/
__global__ void ILPKernel(const int * __restrict__ d_a, int * __restrict__ d_b, const int ILP, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x * ILP;

	if (tid >= N) return;

	for (int j = 0; j < ILP; j++) d_b[tid + j * blockDim.x] = d_a[tid + j * blockDim.x];

}

/********/
/* MAIN */
/********/
int main() {

	const int N = 2097152 * 64;
	//const int N = 1048576;
	//const int N = 262144;
	//const int N = 2048;

	const int numITER = 100;

	const int ILP = 1;

	TimingGPU timerGPU;

	int *h_a = (int *)malloc(N * sizeof(int));
	int *h_b = (int *)malloc(N * sizeof(int));

	for (int i = 0; i<N; i++) {
		h_a[i] = 2;
		h_b[i] = 1;
	}

	int *d_a; gpuErrchk(cudaMalloc(&d_a, N * sizeof(int)));
	int *d_b; gpuErrchk(cudaMalloc(&d_b, N * sizeof(int)));

	gpuErrchk(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

	/**************/
	/* ILP KERNEL */
	/**************/
	float timeTotal = 0.f;
	for (int k = 0; k < numITER; k++) {
		timerGPU.StartCounter();
		ILPKernel << <iDivUp(N / ILP, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, ILP, N);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		timeTotal = timeTotal + timerGPU.GetCounter();
	}

	printf("Bandwidth = %f GB / s; Num blocks = %d\n", (4.f * N * numITER) / (1e6 * timeTotal), iDivUp(N / ILP, BLOCKSIZE));
	gpuErrchk(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < N; i++) if (h_a[i] != h_b[i]) { printf("Error at i = %i for kernel0! Host = %i; Device = %i\n", i, h_a[i], h_b[i]); return 1; }

	return 0;

}
