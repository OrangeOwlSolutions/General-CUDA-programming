#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
 
#include "Utilities.cuh"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16
 
#define N 256
#define M 256
 
/******************/
/* TEST KERNEL 2D */
/******************/
__global__ void test_kernel_2D(float* d_a, size_t pitch) {

	int tidx = blockIdx.x*blockDim.x+threadIdx.x;
	int tidy = blockIdx.y*blockDim.y+threadIdx.y;
 
	if ((tidx<M) && (tidy<N)) {
		float* row_a = (float*)((char*)d_a + tidx*pitch);
		row_a[tidy] = row_a[tidy] * row_a[tidy];
	}
}
 
/********/
/* MAIN */
/********/

int main() {
	
	float a[N][M];
	float *d_a;
	size_t pitch;
 

	for (int i=0; i<N; i++)
		for (int j=0; j<M; j++) {
			a[i][j] = 3.f;
			printf("row %i column %i value %f \n",i,j,a[i][j]);
		}

	// --- 2D pitched allocation and host->device memcopy
	gpuErrchk(cudaMallocPitch(&d_a,&pitch,M*sizeof(float),N));
    gpuErrchk(cudaMemcpy2D(d_a,pitch,a,M*sizeof(float),M*sizeof(float),N,cudaMemcpyHostToDevice));

	dim3 GridSize1(iDivUp(M,BLOCKSIZE_x),iDivUp(N,BLOCKSIZE_y));
	dim3 BlockSize1(BLOCKSIZE_y,BLOCKSIZE_x);
	test_kernel_2D<<<GridSize1,BlockSize1>>>(d_a,pitch);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy2D(a,M*sizeof(float),d_a,pitch,M*sizeof(float),N,cudaMemcpyDeviceToHost));

	for (int i=0; i<N; i++) for (int j=0; j<M; j++) printf("row %i column %i value %f\n",i,j,a[i][j]);

	return 0;
}
