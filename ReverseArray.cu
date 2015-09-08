#include <stdio.h>
#include <assert.h>
 
#include "Utilities.cuh"

/********/
/* MAIN */
/********/
int main() 
{
    // --- Number of elements
	const int N = 6;								

	// --- Input host array
    float *h_in = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = i;
 
	// --- Output host array
    float *h_out = (float *)malloc(N * sizeof(float));

	// --- Input device array
	float *d_in;	gpuErrchk(cudaMalloc(&d_in, N * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

	// --- Output device array
	float *d_out;	gpuErrchk(cudaMalloc(&d_out, N * sizeof(float)));
	
    reverseArray(d_in, d_out, N);
 
    gpuErrchk(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
 
    // --- Result check
    for (int i = 0; i < N; i++) assert(h_out[i] == N - 1 - i );
    for (int i = 0; i < N; i++) printf("%i %f\n", i, h_out[i]);
 
    // --- Free device memory
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
 
    // --- Free host memory
    free(h_in);
 
    printf("Test passed!\n");
 
    return 0;
}
 
