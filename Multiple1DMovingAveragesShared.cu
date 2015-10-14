#include <thrust/device_vector.h>

#define RADIUS        3
#define BLOCK_SIZE_X  8
#define BLOCK_SIZE_Y  8

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**********/
/* KERNEL */
/**********/
__global__ void moving_average(unsigned int *in, unsigned int *out, unsigned int M, unsigned int N) {

    __shared__ unsigned int temp[BLOCK_SIZE_Y][BLOCK_SIZE_X + 2 * RADIUS];

    unsigned int gindexx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int gindexy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int gindex  = gindexy * N + gindexx;

    unsigned int lindexx = threadIdx.x + RADIUS;
    unsigned int lindexy = threadIdx.y;

    // --- Read input elements into shared memory
    temp[lindexy][lindexx] = ((gindexx < N)&&(gindexy < M))? in[gindex] : 0;
    if (threadIdx.x < RADIUS) {
        temp[lindexy][threadIdx.x] = ((gindexx >= RADIUS)&&(gindexx < (N + RADIUS))&&(gindexy < M)) ? in[gindex - RADIUS] : 0;
        temp[lindexy][threadIdx.x + (RADIUS + min(BLOCK_SIZE_X, N - blockIdx.x * BLOCK_SIZE_X))] = (((gindexx + min(BLOCK_SIZE_X, N - blockIdx.x * BLOCK_SIZE_X)) < N)&&(gindexy < M))? in[gindexy * N + gindexx + min(BLOCK_SIZE_X, N - blockIdx.x * BLOCK_SIZE_X)] : 0;
        if ((threadIdx.y == 0)&&(gindexy < M)&&((gindexx + BLOCK_SIZE_X) < N)&&(gindexy < M)) printf("Inside 2 - tidx = %i; bidx = %i; tidy = %i; bidy = %i; lindexx = %i; temp = %i\n", threadIdx.x, blockIdx.x, threadIdx.y, blockIdx.y, threadIdx.x + (RADIUS + BLOCK_SIZE_X), temp[lindexy][threadIdx.x + (RADIUS + BLOCK_SIZE_X)]);
    }

    __syncthreads();

    // --- Apply the stencil
    unsigned int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
        result += temp[lindexy][lindexx + offset];
    }

    // --- Store the result
    out[gindexy * N + gindexx] = result;
}

/********/
/* MAIN */
/********/
int main() {

    const unsigned int M        = 2;
    const unsigned int N        = 4 + 2 * RADIUS;

    const unsigned int constant = 3;

    thrust::device_vector<unsigned int> d_in(M * N, constant);
    thrust::device_vector<unsigned int> d_out(M * N);

    dim3 GridSize(iDivUp(N, BLOCK_SIZE_X), iDivUp(M, BLOCK_SIZE_Y));
    dim3 BlockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    moving_average<<<GridSize, BlockSize>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), M, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::host_vector<unsigned int> h_out = d_out;

    for (int j=0; j<M; j++) {
        for (int i=0; i<N; i++)
            printf("Element j = %i; i = %i; h_out = %i\n", j, i, h_out[N*j+i]);
    }

    return 0;

}
