#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include "Utilities.cuh"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#define Nrows 3
#define Ncols 5

/******************/
/* TEST KERNEL 2D */
/******************/
__global__ void test_kernel_2D(float *devPtr, size_t pitch)
{
	int    tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int    tidy = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ((tidx < Ncols) && (tidy < Nrows))
	{
		float *row_a = (float *)((char*)devPtr + tidy * pitch);
		row_a[tidx] = row_a[tidx] * tidx * tidy;
	}
}

/********/
/* MAIN */
/********/
int main()
{
	float hostPtr[Nrows][Ncols];
	float *devPtr;
	size_t pitch;

	for (int i = 0; i < Nrows; i++)
		for (int j = 0; j < Ncols; j++) {
			hostPtr[i][j] = 1.f;
			//printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
		}

	// --- 2D pitched allocation and host->device memcopy
	gpuErrchk(cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows));
	gpuErrchk(cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice));

	dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

	test_kernel_2D << <gridSize, blockSize >> >(devPtr, pitch);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost));

	for (int i = 0; i < Nrows; i++) 
		for (int j = 0; j < Ncols; j++) 
			printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);

	return 0;

}
