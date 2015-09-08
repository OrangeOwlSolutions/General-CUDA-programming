#include <cstdio>

#include <thrust/pair.h>

#include "Matlab_like.cuh"
#include "Utilities.cuh"

#define BLOCKSIZE_MESHGRID_X	16
#define BLOCKSIZE_MESHGRID_Y	16

#define DEBUG

/*******************/
/* MESHGRID KERNEL */
/*******************/
template <class T>
__global__ void meshgrid_kernel(const T * __restrict__ x, size_t Nx, const float * __restrict__ y, size_t Ny, T * __restrict__ X, T * __restrict__ Y) 
{
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if ((tidx < Nx) && (tidy < Ny)) {	
		X[tidy * Nx + tidx] = x[tidx];
		Y[tidy * Nx + tidx] = y[tidy];
	}
}

/************/
/* MESHGRID */
/************/
template <class T>
thrust::pair<T *,T *> meshgrid(const T *x, const unsigned int Nx, const T *y, const unsigned int Ny) {
	
	T *X; gpuErrchk(cudaMalloc((void**)&X, Nx * Ny * sizeof(T)));
	T *Y; gpuErrchk(cudaMalloc((void**)&Y, Nx * Ny * sizeof(T)));

	dim3 BlockSize(BLOCKSIZE_MESHGRID_X, BLOCKSIZE_MESHGRID_Y);
	dim3 GridSize (iDivUp(Nx, BLOCKSIZE_MESHGRID_X), iDivUp(BLOCKSIZE_MESHGRID_Y, BLOCKSIZE_MESHGRID_Y));
	
	meshgrid_kernel<<<GridSize, BlockSize>>>(x, Nx, y, Ny, X, Y);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return thrust::make_pair(X, Y);
}

/********/
/* MAIN */
/********/
int main()
{
	const int Nx = 3;
	const int Ny = 4;

	float *h_x = (float *)malloc(Nx * sizeof(float));
	float *h_y = (float *)malloc(Ny * sizeof(float));

	float *h_X = (float *)malloc(Nx * Ny * sizeof(float));
	float *h_Y = (float *)malloc(Nx * Ny * sizeof(float));

	for (int i = 0; i < Nx; i++) h_x[i] = i;
	for (int i = 0; i < Ny; i++) h_y[i] = i + 4.f;
	
	float *d_x;	gpuErrchk(cudaMalloc(&d_x, Nx * sizeof(float)));
	float *d_y;	gpuErrchk(cudaMalloc(&d_y, Ny * sizeof(float)));

	gpuErrchk(cudaMemcpy(d_x, h_x, Nx * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_y, h_y, Ny * sizeof(float), cudaMemcpyHostToDevice));
	
	thrust::pair<float *, float *> meshgrid_pointers = meshgrid(d_x, Nx, d_y, Ny);
	float *d_X = (float *)meshgrid_pointers.first;
	float *d_Y = (float *)meshgrid_pointers.second;

	gpuErrchk(cudaMemcpy(h_X, d_X, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_Y, d_Y, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost));
 
	for (int j = 0; j < Ny; j++) {
		for (int i = 0; i < Nx; i++) {
			printf("i = %i; j = %i; x = %f; y = %f\n", i, j, h_X[j * Nx + i], h_Y[j * Nx + i]);
		}
	}

	return 0;

}
