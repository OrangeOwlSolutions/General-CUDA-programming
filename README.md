# CUDA codes

- ```Reductions.cu```: different kinds of reductions, including reduction without```__syncthreads```, reduction with no deadlock, atomic warp reduction and threadfence reduction, see [???](???);
- ```MultipleMovingAverages.cu```: performing several 1d moving averages in parallel, see [Performing several 1D moving averages in parallel using CUDA](http://www.orangeowlsolutions.com/archives/1161);
- ```SurfaceMemory.cu```: simple example on how using CUDA surface memory to write to a texture memory, see [Texture memory with READ and WRITE](http://stackoverflow.com/questions/12509346/texture-memory-with-read-and-write);
- ```MedianFilterPeriodicBoundary.cu```: showing how a median filter can be easily implemented on a signal having periodic boundary using CUDA textures, see [Dealing with Boundary conditions / Halo regions in CUDA](http://stackoverflow.com/questions/5715220/what-is-usually-done-with-the-boundary-pixels-of-the-image/32348533#32348533);
