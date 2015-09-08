#include <cstdio>

#include <thrust/pair.h>

#include "Matlab_like.cuh"
#include "Utilities.cuh"

#define DEBUG

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
