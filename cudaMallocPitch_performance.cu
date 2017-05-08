#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

/******************/
/* TEST KERNEL 2D */
/******************/
__global__ void test_kernel_2D(float * __restrict__ devPtrA, float * __restrict__ devPtrB, float * __restrict__ devPtrC, const int Nrows, const int Ncols)
{
	int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int    tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if ((tidx < Ncols) && (tidy < Nrows))
		devPtrA[tidy * Ncols + tidx] = devPtrA[tidy * Ncols + tidx] + devPtrB[tidy * Ncols + tidx] + devPtrC[tidy * Ncols + tidx];
}

/**************************/
/* TEST KERNEL PITCHED 2D */
/**************************/
__global__ void test_kernel_Pitched_2D(float * __restrict__ devPtrA, float * __restrict__ devPtrB, float * __restrict__ devPtrC, const size_t pitchA, const size_t pitchB, const size_t pitchC, const int Nrows, const int Ncols)
{
	int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int    tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if ((tidx < Ncols) && (tidy < Nrows))
	{
		float *row_a = (float *)((char*)devPtrA + tidy * pitchA);
		float *row_b = (float *)((char*)devPtrB + tidy * pitchB);
		float *row_c = (float *)((char*)devPtrC + tidy * pitchC);
		row_a[tidx] = row_a[tidx] + row_b[tidx] + row_c[tidx];
	}
}

/********/
/* MAIN */
/********/
int main()
{
	const int Nrows = 7100;
	const int Ncols = 2300;

	TimingGPU timerGPU;

	float *hostPtrA = (float *)malloc(Nrows * Ncols * sizeof(float));
	float *hostPtrB = (float *)malloc(Nrows * Ncols * sizeof(float));
	float *hostPtrC = (float *)malloc(Nrows * Ncols * sizeof(float));
	float *devPtrA, *devPtrPitchedA;
	float *devPtrB, *devPtrPitchedB;
	float *devPtrC, *devPtrPitchedC;
	size_t pitchA, pitchB, pitchC;

	for (int i = 0; i < Nrows; i++)
		for (int j = 0; j < Ncols; j++) {
		hostPtrA[i * Ncols + j] = 1.f;
		hostPtrB[i * Ncols + j] = 2.f;
		hostPtrC[i * Ncols + j] = 3.f;
		//printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
		}

	// --- 2D non-pitched allocation and host->device memcopy
	gpuErrchk(cudaMalloc(&devPtrA, Nrows * Ncols * sizeof(float)));
	gpuErrchk(cudaMalloc(&devPtrB, Nrows * Ncols * sizeof(float)));
	gpuErrchk(cudaMalloc(&devPtrC, Nrows * Ncols * sizeof(float)));
	gpuErrchk(cudaMemcpy(devPtrA, hostPtrA, Nrows * Ncols * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devPtrB, hostPtrB, Nrows * Ncols * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devPtrC, hostPtrC, Nrows * Ncols * sizeof(float), cudaMemcpyHostToDevice));

	// --- 2D pitched allocation and host->device memcopy
	gpuErrchk(cudaMallocPitch(&devPtrPitchedA, &pitchA, Ncols * sizeof(float), Nrows));
	gpuErrchk(cudaMallocPitch(&devPtrPitchedB, &pitchB, Ncols * sizeof(float), Nrows));
	gpuErrchk(cudaMallocPitch(&devPtrPitchedC, &pitchC, Ncols * sizeof(float), Nrows));
	gpuErrchk(cudaMemcpy2D(devPtrPitchedA, pitchA, hostPtrA, Ncols * sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy2D(devPtrPitchedB, pitchB, hostPtrB, Ncols * sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy2D(devPtrPitchedC, pitchC, hostPtrC, Ncols * sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice));

	dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

	timerGPU.StartCounter();
	test_kernel_2D << <gridSize, blockSize >> >(devPtrA, devPtrB, devPtrC, Nrows, Ncols);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	printf("Non-pitched - Time = %f; Memory = %i bytes \n", timerGPU.GetCounter(), Nrows * Ncols * sizeof(float));

	timerGPU.StartCounter();
	test_kernel_Pitched_2D << <gridSize, blockSize >> >(devPtrPitchedA, devPtrPitchedB, devPtrPitchedC, pitchA, pitchB, pitchC, Nrows, Ncols);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	printf("Pitched - Time = %f; Memory = %i bytes \n", timerGPU.GetCounter(), Nrows * pitchA);

	//gpuErrchk(cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtrPitched, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hostPtrA, devPtrA, Nrows * Ncols * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hostPtrB, devPtrB, Nrows * Ncols * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hostPtrC, devPtrC, Nrows * Ncols * sizeof(float), cudaMemcpyDeviceToHost));

	//for (int i = 0; i < Nrows; i++) 
	//	for (int j = 0; j < Ncols; j++) 
	//		printf("row %i column %i value %f \n", i, j, hostPtr[i * Ncols + j]);

	return 0;

}
