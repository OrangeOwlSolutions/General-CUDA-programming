# CUDA codes

- ```Reductions.cu```: different kinds of reductions, including reduction without```__syncthreads```, reduction with no deadlock, atomic warp reduction and threadfence reduction, see [???](???);
- ```MultipleMovingAverages.cu```: performing several 1d moving averages in parallel, see [Performing several 1D moving averages in parallel using CUDA](http://www.orangeowlsolutions.com/archives/1161);
- ```SurfaceMemory.cu```: simple example on how using CUDA surface memory to write to a texture memory, see [Texture memory with READ and WRITE](http://stackoverflow.com/questions/12509346/texture-memory-with-read-and-write);
- ```MedianFilterPeriodicBoundary.cu```: showing how a median filter can be easily implemented on a signal having periodic boundary using CUDA textures, see [Dealing with boundary conditions in CUDA](http://www.orangeowlsolutions.com/archives/1436);
- ```cudaMallocPitch_and_cudaMemcpy2D.cu```: showing how using ```cudaMallocPitch``` to allocate 2D arrays and how moving 2D data from/to host memory to/from global memory allocated with cudaMallocPitch using ```cudaMemcpy2D```, see [cudaMallocPitch and cudaMemcpy2D](http://www.orangeowlsolutions.com/archives/613);
- ```WriteToCUDATextureAcrossKernels.cu```: writing to a CUDA texture across different kernel launches, see [Writing to a CUDA texture across kernels](http://www.orangeowlsolutions.com/archives/1440);
- ```AddressingModesCUDATextures.cu```: the different types of addressing modes of a CUDA texture, see [The different addressing modes of CUDA textures](http://stackoverflow.com/questions/19020963/the-different-addressing-modes-of-cuda-textures);
- ```Meshgrid.cu```: Emulating Matlab's meshgrid in CUDA, see [Replicate a vector multiple times using CUDA Thrust](http://stackoverflow.com/questions/16900837/replicate-a-vector-multiple-times-using-cuda-thrust/32451396#32451396);
- ```ReverseArray.cu```: Reversing the order of the elements within an array, see [???](???);
