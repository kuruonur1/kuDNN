#include <cudnn.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        cudaDeviceReset();
        exit(code);
    }
}
#define cudnnErrchk(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line)
{
    if (code != CUDNN_STATUS_SUCCESS) 
    {
        fprintf(stderr,"CUDNNassert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        cudaDeviceReset();
        exit(code);
    }
}
