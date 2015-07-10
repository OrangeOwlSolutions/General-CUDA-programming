#include <thrust\device_vector.h>

#define BLOCKSIZE 256

#define warpSize 32

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  if (abort) { getchar(); exit(code); }
   }
}

/*******************************************************/
/* CALCULATING THE NEXT POWER OF 2 OF A CERTAIN NUMBER */
/*******************************************************/
unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/*************************************/
/* CHECK IF A NUMBER IS A POWER OF 2 */
/*************************************/
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

/******************/
/* REDUCE0 KERNEL */
/******************/
/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i		= blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

    // --- Loading data to shared memory
    sdata[tid] = (i < N) ? g_idata[i] : 0;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

    // --- Reduction in shared memory
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // --- Only the threads with index multiple of 2*s perform additions. Furthermore, modulo arithmetic is slow.       
		if ((tid % (2*s)) == 0) { sdata[tid] += sdata[tid + s]; }
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
		__syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/******************/
/* REDUCE1 KERNEL */
/******************/
/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void reduce1(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i		= blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

    // --- Loading data to shared memory
    sdata[tid] = (i < N) ? g_idata[i] : 0;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
		int index = 2 * s * tid;
        // --- Use contiguous threads leading to non-divergent branch
        if (index < blockDim.x) { sdata[index] += sdata[index + s]; /* --- Strided shared memory access */ }
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/******************/
/* REDUCE2 KERNEL */
/******************/
/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i		= blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

    // --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
    sdata[tid] = (i < N) ? g_idata[i] : 0;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/******************/
/* REDUCE3 KERNEL */
/******************/
/*
    This version performs the first level of reduction using registers when reading from global memory.
*/
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int N) {
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory. 
    T mySum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mySum += g_idata[i+blockDim.x];
	sdata[tid] = mySum;
    
    // --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s) { sdata[tid] = mySum = mySum + sdata[tid + s]; }
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/******************/
/* REDUCE4 KERNEL */
/******************/
/*
    This version uses the warp shuffle operation if available to reduce 
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    This kernel assumes that blockSize > 64.
*/
template <class T>
__global__ void reduce4(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory. 
    T mySum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mySum += g_idata[i+blockDim.x];
	sdata[tid] = mySum;

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s) { sdata[tid] = mySum = mySum + sdata[tid + s]; }
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

#if (__CUDA_ARCH__ >= 300 )
	// --- Single warp reduction by shuffle operations
    if ( tid < 32 )
    {
        // --- Last iteration removed from the for loop, but needed for shuffle reduction
        mySum += sdata[tid + 32];
        // --- Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) mySum += __shfl_down(mySum, offset);
		//for (int offset=1; offset < warpSize; offset *= 2) mySum += __shfl_xor(mySum, i);
	}
#else
	// --- Single warp reduction by loop unrolling. Assuming blockDim.x >64
    if (tid < 32) {
		sdata[tid] = mySum = mySum + sdata[tid + 32]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid + 16]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  8]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  4]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  2]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  1]; __syncthreads();
	}
#endif

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/******************/
/* REDUCE5 KERNEL */
/******************/
/*
    This version is completely unrolled, unless warp shuffle is available, then
    shuffle is used within a loop.  It uses a template parameter to achieve
    optimal code for any (power of 2) number of threads. This requires a switch
    statement in the host code to handle all the different thread block sizes at
    compile time. When shuffle is available, it is used to reduce warp synchronization.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void reduce5(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory. 
    T mySum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mySum += g_idata[i+blockDim.x];
	sdata[tid] = mySum;

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

    // --- Reduction in shared memory. Fully unrolled loop.
    if ((blockSize >= 512) && (tid < 256)) sdata[tid] = mySum = mySum + sdata[tid + 256];
    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) sdata[tid] = mySum = mySum + sdata[tid + 128];
     __syncthreads();

    if ((blockSize >= 128) && (tid <  64)) sdata[tid] = mySum = mySum + sdata[tid +  64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	// --- Single warp reduction by shuffle operations
    if ( tid < 32 )
    {
        // --- Last iteration removed from the for loop, but needed for shuffle reduction
        mySum += sdata[tid + 32];
        // --- Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) mySum += __shfl_down(mySum, offset);
		//for (int offset=1; offset < warpSize; offset *= 2) mySum += __shfl_xor(mySum, i);
    }
#else
    // --- Reduction within a single warp. Fully unrolled loop.
    if ((blockSize >=  64) && (tid < 32)) sdata[tid] = mySum = mySum + sdata[tid + 32];
    __syncthreads();

    if ((blockSize >=  32) && (tid < 16)) sdata[tid] = mySum = mySum + sdata[tid + 16];
    __syncthreads();

    if ((blockSize >=  16) && (tid <  8)) sdata[tid] = mySum = mySum + sdata[tid +  8];
    __syncthreads();

    if ((blockSize >=   8) && (tid <  4)) sdata[tid] = mySum = mySum + sdata[tid +  4];
     __syncthreads();

    if ((blockSize >=   4) && (tid <  2)) sdata[tid] = mySum = mySum + sdata[tid +  2];
    __syncthreads();

    if ((blockSize >=   2) && ( tid < 1)) sdata[tid] = mySum = mySum + sdata[tid +  1];
    __syncthreads();
#endif

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

template <class T>
void reduce5_wrapper(T *g_idata, T *g_odata, unsigned int N, int NumBlocks, int NumThreads, int smemSize) {
    switch (NumThreads) {
		case 512: reduce5<T, 512><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case 256: reduce5<T, 256><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case 128: reduce5<T, 128><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case 64:  reduce5<T,  64><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case 32:  reduce5<T,  32><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case 16:  reduce5<T,  16><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case  8:  reduce5<T,   8><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case  4:  reduce5<T,   4><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case  2:  reduce5<T,   2><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		case  1:  reduce5<T,   1><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
}

/******************/
/* REDUCE6 KERNEL */
/******************/
/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension
    unsigned int gridSize = blockSize*2*gridDim.x;

    // --- Performs the first level of reduction in registers when reading from global memory on multiple elements per thread. 
	//     More blocks will result in a larger gridSize and therefore fewer elements per thread
	T mySum = 0;

    while (i < N) {
        mySum += g_idata[i];
        // --- Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < N) mySum += g_idata[i+blockSize];
        i += gridSize; }

    // --- Each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // --- Reduction in shared memory. Fully unrolled loop.
    if ((blockSize >= 512) && (tid < 256)) sdata[tid] = mySum = mySum + sdata[tid + 256];
    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) sdata[tid] = mySum = mySum + sdata[tid + 128];
     __syncthreads();

    if ((blockSize >= 128) && (tid <  64)) sdata[tid] = mySum = mySum + sdata[tid +  64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	// --- Single warp reduction by shuffle operations
    if ( tid < 32 )
    {
        // --- Last iteration removed from the for loop, but needed for shuffle reduction
        mySum += sdata[tid + 32];
        // --- Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) mySum += __shfl_down(mySum, offset);
		//for (int offset=1; offset < warpSize; offset *= 2) mySum += __shfl_xor(mySum, i);
    }
#else
    // --- Reduction within a single warp. Fully unrolled loop.
    if ((blockSize >=  64) && (tid < 32)) sdata[tid] = mySum = mySum + sdata[tid + 32];
    __syncthreads();

    if ((blockSize >=  32) && (tid < 16)) sdata[tid] = mySum = mySum + sdata[tid + 16];
    __syncthreads();

    if ((blockSize >=  16) && (tid <  8)) sdata[tid] = mySum = mySum + sdata[tid +  8];
    __syncthreads();

    if ((blockSize >=   8) && (tid <  4)) sdata[tid] = mySum = mySum + sdata[tid +  4];
     __syncthreads();

    if ((blockSize >=   4) && (tid <  2)) sdata[tid] = mySum = mySum + sdata[tid +  2];
    __syncthreads();

    if ((blockSize >=   2) && ( tid < 1)) sdata[tid] = mySum = mySum + sdata[tid +  1];
    __syncthreads();
#endif

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

template <class T>
void reduce6_wrapper(T *g_idata, T *g_odata, unsigned int N, int NumBlocks, int NumThreads, int smemSize) {
	if (isPow2(N)) {
		switch (NumThreads) {
			case 512: reduce6<T, 512, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 256: reduce6<T, 256, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 128: reduce6<T, 128, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 64:  reduce6<T,  64, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 32:  reduce6<T,  32, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 16:  reduce6<T,  16, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  8:  reduce6<T,   8, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  4:  reduce6<T,   4, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  2:  reduce6<T,   2, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  1:  reduce6<T,   1, true><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
		}
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
    else {
		switch (NumThreads) {
			case 512: reduce6<T, 512, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 256: reduce6<T, 256, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 128: reduce6<T, 128, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 64:  reduce6<T,  64, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 32:  reduce6<T,  32, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case 16:  reduce6<T,  16, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  8:  reduce6<T,   8, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  4:  reduce6<T,   4, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  2:  reduce6<T,   2, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
			case  1:  reduce6<T,   1, false><<< NumBlocks, NumThreads, smemSize >>>(g_idata, g_odata, N); break;
		}
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
}

/***************************************/
/* REDUCE4 KERNEL - NO __syncthreads() */
/***************************************/
/*
    This version demonstrates the need of use of volatile to remove the __syncthreads() call in single warp reduction with loop unrolling.

    This kernel assumes that blockSize > 64.
*/
template <class T>
__device__ void warpReduce(volatile T *sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template <class T>
__global__ void reduce4_no_synchthreads(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory. 
    T mySum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mySum += g_idata[i+blockDim.x];
	sdata[tid] = mySum;

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s) { sdata[tid] = mySum = mySum + sdata[tid + s]; }
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

	// --- Single warp reduction by loop unrolling. Assuming blockDim.x >64
    if (tid < 32) warpReduce(sdata, tid);

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/********************************/
/* REDUCE4 KERNEL - NO DEADLOCK */
/********************************/
template <class T>
__global__ void reduce4_deadlock_test(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory. 
    T mySum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mySum += g_idata[i+blockDim.x];
	sdata[tid] = mySum;

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid >= s) return;
        sdata[tid] = mySum = mySum + sdata[tid + s];
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

	// --- Single warp reduction by loop unrolling. Assuming blockDim.x >64
    if (tid < 32) {
		sdata[tid] = mySum = mySum + sdata[tid + 32]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid + 16]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  8]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  4]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  2]; __syncthreads();
		sdata[tid] = mySum = mySum + sdata[tid +  1]; __syncthreads();
	}

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/******************************************/
/* REDUCE4 KERNEL - ATOMIC WARP REDUCTION */
/******************************************/
/*
    This version serves to analyze the impact of atomic operations.

    This kernel assumes that blockSize > 64.
*/
template <class T>
__device__ void atomicWarpReduce(T *sdata, int tid) {
	atomicAdd(&sdata[tid], sdata[tid + 32]); __syncthreads();
	atomicAdd(&sdata[tid], sdata[tid + 16]); __syncthreads();
	atomicAdd(&sdata[tid], sdata[tid + 8]); __syncthreads();
	atomicAdd(&sdata[tid], sdata[tid + 4]); __syncthreads();
	atomicAdd(&sdata[tid], sdata[tid + 2]); __syncthreads();
	atomicAdd(&sdata[tid], sdata[tid + 1]); __syncthreads();
}

//template <class T>
//__device__ void atomicWarpReduce(T *sdata, int tid) {
//	atomicAdd(&sdata[tid], sdata[tid + 32]); 
//	atomicAdd(&sdata[tid], sdata[tid + 16]);
//	atomicAdd(&sdata[tid], sdata[tid + 8]); 
//	atomicAdd(&sdata[tid], sdata[tid + 4]); 
//	atomicAdd(&sdata[tid], sdata[tid + 2]); 
//	atomicAdd(&sdata[tid], sdata[tid + 1]);
//}

template <class T>
__global__ void reduce4_atomicWarp(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory. 
    T mySum = (i < N) ? g_idata[i] : 0;
	if (i + blockDim.x < N) mySum += g_idata[i+blockDim.x];
	sdata[tid] = mySum;

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s) { sdata[tid] = mySum = mySum + sdata[tid + s]; }
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

	// --- Single warp reduction by loop unrolling. Assuming blockDim.x >64. Atomic additions
    if (tid < 32) atomicWarpReduce(sdata, tid); 
    //if (tid < 32) warpReduce(sdata, tid);

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/***************************/
/* REDUCE0 - STACKOVERFLOW */
/***************************/
template <class T>
__global__ void reduce0_stackoverflow(T *g_idata , T *g_odata, unsigned int N) {
    
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i		= blockIdx.x * blockDim.x + threadIdx.x;	// Global thread index

    // --- Loading data to shared memory
    //sdata[tid] = (i < N) ? g_idata[i] : 0;						// CUDA SDK
	sdata[tid] = 0;
    while(i < N){
        sdata[tid] += g_idata[i];
        i+=gridDim.x * blockDim.x; // to handle array of any size
    }

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory
    //  for (unsigned int s=1; s < blockDim.x; s *= 2)					// CUDA SDK
    //  {
	//     if ((tid % (2*s)) == 0) { sdata[tid] += sdata[tid + s]; }
	//     __syncthreads();
    //  }
	unsigned int s = 1;
    while(s < blockDim.x) 
	{
		// --- Only the threads with index multiple of 2*s perform additions. Furthermore, modulo arithmetic is slow.       
		if ((tid % (2*s)) == 0) { sdata[tid] += sdata[tid + s]; }

		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();

		s*=2;
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

/*************************/
/* THREADFENCE REDUCTION */
/*************************/
template <class T, unsigned int blockSize>
__device__ void
reduceBlock(volatile T *sdata, T mySum, const unsigned int tid)
{
    sdata[tid] = mySum; __syncthreads();

    // --- Reduction in shared memory
    if (blockSize >= 512) { if (tid < 256) sdata[tid] = mySum = mySum + sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] = mySum = mySum + sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] = mySum = mySum + sdata[tid +  64]; __syncthreads(); }
	if (tid < 32)
    {
        if (blockSize >=  64) sdata[tid] = mySum = mySum + sdata[tid + 32];
        if (blockSize >=  32) sdata[tid] = mySum = mySum + sdata[tid + 16];
        if (blockSize >=  16) sdata[tid] = mySum = mySum + sdata[tid +  8];
        if (blockSize >=   8) sdata[tid] = mySum = mySum + sdata[tid +  4];
        if (blockSize >=   4) sdata[tid] = mySum = mySum + sdata[tid +  2];
        if (blockSize >=   2) sdata[tid] = mySum = mySum + sdata[tid +  1];
    }
}

template <class T, unsigned int blockSize, bool nIsPow2>
__device__ void reduceBlocks(const T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid	= threadIdx.x;								// Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;		// Global thread index - Fictitiously double the block dimension
    unsigned int gridSize = blockSize*2*gridDim.x;

    // --- Performs the first level of reduction in registers when reading from global memory on multiple elements per thread. 
	//     More blocks will result in a larger gridSize and therefore fewer elements per thread
	T mySum = 0;

    while (i < N) {
        mySum += g_idata[i];
        // --- Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < N) mySum += g_idata[i+blockSize];
        i += gridSize; }

    // --- Reduction in shared memory. Fully unrolled loop.
    reduceBlock<T, blockSize>(sdata, mySum, tid);

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
	//     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// --- Global variable used by reduceSinglePass to count how many blocks have finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt) { return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice); }

// This reduction kernel reduces an arbitrary sized array in a single kernel invocation.
// It does so by keeping track of how many blocks have finished. After each thread block completes the reduction of its own block of data, 
// it "takes a ticket" by atomically incrementing a global counter. If the ticket value is equal to the number of thread blocks, then the block 
// holding the ticket knows that it is the last block to finish. This last block is responsible for summing the results of all the other blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all of its memory transactions have completed. This is what 
// __threadfence() does -- it blocks until the results of all outstanding memory transactions within the calling thread are visible to all other 
// threads of the entire grid.
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const T *g_idata, T *g_odata, unsigned int N)
{
    // --- PHASE 1: Process all inputs assigned to this block
    reduceBlocks<T, blockSize, nIsPow2>(g_idata, g_odata, N);

    // --- PHASE 2: Last block finished will process all partial sums
    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;
        __shared__ bool amLast;
        extern T __shared__ smem[];

        // --- Wait until all outstanding memory instructions in this thread are finished
        __threadfence();

        // --- Thread 0 takes a ticket
        if (tid==0)
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // --- If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x-1);
        }

        __syncthreads();

        // --- The last block sums the results of all other blocks
        if (amLast)
        {
            int i = tid;
            T mySum = 0;

            while (i < gridDim.x)
            {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<T, blockSize>(smem, mySum, tid);

            if (tid==0)
            {
                g_odata[0] = smem[0];
                // --- Reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}

template <class T>
void reduce_threadfence_wrapper(T *g_idata, T *g_odata, unsigned int N, int NumBlocks, int NumThreads, int smemSize) {
	if (isPow2(N)) {
        switch (NumThreads) {
            case 512: reduceSinglePass<T, 512, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case 256: reduceSinglePass<T, 256, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case 128: reduceSinglePass<T, 128, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case  64: reduceSinglePass<T,  64, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case  32: reduceSinglePass<T,  32, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case  16: reduceSinglePass<T,  16, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   8: reduceSinglePass<T,   8, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   4: reduceSinglePass<T,   4, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   2: reduceSinglePass<T,   2, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   1: reduceSinglePass<T,   1, true><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break; 
		} 
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
    else
    {
        switch (NumThreads) {
            case 512: reduceSinglePass<T, 512, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case 256: reduceSinglePass<T, 256, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case 128: reduceSinglePass<T, 128, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case  64: reduceSinglePass<T,  64, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case  32: reduceSinglePass<T,  32, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case  16: reduceSinglePass<T,  16, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   8: reduceSinglePass<T,   8, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   4: reduceSinglePass<T,   4, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   2: reduceSinglePass<T,   2, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
            case   1: reduceSinglePass<T,   1, false><<<NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N); break;
        }
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
    }
}

/********/
/* MAIN */
/********/
int main()
{
	//const int N = 131072*2*2*2*2*2*2*2*2;
	//const int N =   131072*2*2*2*2*2*2*2*2;
	//const int N = 15336;
	const int N = 30000;

	thrust::device_vector<int> d_vec(N,3);

	int NumThreads	= (N < BLOCKSIZE) ? nextPow2(N) : BLOCKSIZE;
    int NumBlocks	= (N + NumThreads - 1) / NumThreads;

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);

	// --- Creating events for timing
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	thrust::device_vector<int> d_vec_block(NumBlocks);

	/***********/
	/* REDUCE0 */
	/***********/
	cudaEventRecord(start, 0);
	reduce0<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce0 - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	thrust::host_vector<int> h_vec_block(d_vec_block);
	int sum_reduce0 = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce0 = sum_reduce0 + h_vec_block[i];
	printf("Result for reduce0 = %i\n",sum_reduce0);

	/***********/
	/* REDUCE1 */
	/***********/
	cudaEventRecord(start, 0);
	reduce1<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce1 - Elapsed time:  %3.3f ms \n", time);	
	
	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce1 = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce1 = sum_reduce1 + h_vec_block[i];
	printf("Result for reduce1 = %i\n",sum_reduce1);

	/***********/
	/* REDUCE2 */
	/***********/
	cudaEventRecord(start, 0);
	reduce2<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce2 - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce2 = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce2 = sum_reduce2 + h_vec_block[i];
	printf("Result for reduce2 = %i\n",sum_reduce2);

	/***********/
	/* REDUCE3 */
	/***********/
	cudaEventRecord(start, 0);
	reduce3<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce3 - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce3 = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce3 = sum_reduce3 + h_vec_block[i];
	printf("Result for reduce3 = %i\n",sum_reduce3);

	/***********/
	/* REDUCE4 */
	/***********/
	cudaEventRecord(start, 0);
	reduce4<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce4 - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce4 = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce4 = sum_reduce4 + h_vec_block[i];
	printf("Result for reduce4 = %i\n",sum_reduce4);

	/***********/
	/* REDUCE5 */
	/***********/
	cudaEventRecord(start, 0);
	reduce5_wrapper(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N, NumBlocks, NumThreads, smemSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce5 - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce5 = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce5 = sum_reduce5 + h_vec_block[i];
	printf("Result for reduce5 = %i\n",sum_reduce5);

	/***********/
	/* REDUCE6 */
	/***********/
	cudaEventRecord(start, 0);
	reduce6_wrapper(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N, NumBlocks, NumThreads, smemSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce6 - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce6 = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce6 = sum_reduce6 + h_vec_block[i];
	printf("Result for reduce6 = %i\n",sum_reduce6);

	/***************************************/
	/* REDUCE4 KERNEL - NO __syncthreads() */
	/***************************************/
	cudaEventRecord(start, 0);
	reduce4_no_synchthreads<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce4 - no syncthreads() - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce4_no_synchthreads = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce4_no_synchthreads = sum_reduce4_no_synchthreads + h_vec_block[i];
	printf("Result for reduce4_no_synchthreads = %i\n",sum_reduce4_no_synchthreads);

	/********************************/
	/* REDUCE4 KERNEL - NO DEADLOCK */
	/********************************/
	cudaEventRecord(start, 0);
	reduce4_deadlock_test<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce4 - deadlock - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce4_deadlock = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce4_deadlock = sum_reduce4_deadlock + h_vec_block[i];
	printf("Result for reduce4_deadlock = %i\n",sum_reduce4_deadlock);

	/******************************************/
	/* REDUCE4 KERNEL - ATOMIC WARP REDUCTION */
	/******************************************/
	cudaEventRecord(start, 0);
	reduce4_atomicWarp<<<NumBlocks, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce4 - atomic warp reduction - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce4_atomicWarp = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce4_atomicWarp = sum_reduce4_atomicWarp + h_vec_block[i];
	printf("Result for reduce4_atomicWarp = %i\n",sum_reduce4_atomicWarp);

	/**********************************/
	/* REDUCE0 KERNEL - STACKOVERFLOW */
	/**********************************/
	cudaEventRecord(start, 0);
	reduce0_stackoverflow<<<NumBlocks/2, NumThreads, smemSize>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("reduce0 - stackoverflow - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	h_vec_block = d_vec_block;
	int sum_reduce0_stackoverflow = 0;
	for (int i=0; i<NumBlocks; i++) sum_reduce0_stackoverflow = sum_reduce0_stackoverflow + h_vec_block[i];
	printf("Result for reduce0_stackoverflow = %i\n",sum_reduce0_stackoverflow);

	/*************************/
	/* THREADFENCE REDUCTION */
	/*************************/
	cudaEventRecord(start, 0);
	gpuErrchk(setRetirementCount(0));
	reduce_threadfence_wrapper(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_vec_block.data()), N, NumBlocks, NumThreads, smemSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Threadfence reduction - Elapsed time:  %3.3f ms \n", time);	

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	int sum_threadfence = d_vec_block[0];
	printf("Result for threadfence reduction = %i\n",sum_threadfence);

	getchar();
}
