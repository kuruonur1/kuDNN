#include <stdio.h>
#include <cudnn.h>

#define cat3d(A)    A[0],A[1],A[2]
#define cat5d(A)    A[0],A[1],A[2],A[3],A[4]
#define prod5d(A)   (A[0]*A[1]*A[2]*A[3]*A[4])
#define dims2strides5d(A) A[1]*A[2]*A[3]*A[4],A[2]*A[3]*A[4],A[3]*A[4],A[4],1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort){ cudaDeviceReset(); exit(code);}
    }
}
#define cudnnErrchk(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUDNN_STATUS_SUCCESS) 
    {
        fprintf(stderr,"CUDNNassert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        if (abort){ cudaDeviceReset(); exit(code);}
    }
}

#define ind2d(w,i,j) ((i)*(w)+j)
#define ind3d(h,w,i,j,k) ((i)*(h)*(w)+(j)*(w)+k)
#define ind4d(c,h,w,i,j,k,l) ((i)*(c)*(h)*(w)+(j)*(h)*(w)+(k)*(w)+l)
#define ind5d(c,h,w,d,i,j,k,l,m) ((i)*(c)*(h)*(w)*(d)+(j)*(h)*(w)*(d)+(k)*(w)*(d)+(l)*(d)+m)


cudnnStatus_t CUDNNWINAPI kunetConvolutionForward(        cudnnHandle_t                     handle,
                                                          const void                         *alpha,
                                                          const cudnnTensorDescriptor_t       srcDesc,
                                                          const void                         *srcData,
                                                          const cudnnFilterDescriptor_t       filterDesc,
                                                          const void                         *filterData,
                                                          const cudnnConvolutionDescriptor_t  convDesc,
                                                          cudnnConvolutionFwdAlgo_t           algo,
                                                          void                               *workSpace,
                                                          size_t                              workSpaceSizeInBytes,            
                                                          const void                         *beta,
                                                          const cudnnTensorDescriptor_t       destDesc,
                                                          void                               *destData
                                                 );
cudnnStatus_t CUDNNWINAPI kunetConvolutionBackwardFilter( cudnnHandle_t                       handle,
                                                          const void                         *alpha,
                                                          const cudnnTensorDescriptor_t       srcDesc,
                                                          const void                         *srcData,
                                                          const cudnnTensorDescriptor_t       diffDesc,
                                                          const void                         *diffData,
                                                          const cudnnConvolutionDescriptor_t  convDesc,
                                                          const void                         *beta,
                                                          const cudnnFilterDescriptor_t       gradDesc,
                                                          void                               *gradData
                                                        );
cudnnStatus_t CUDNNWINAPI kunetConvolutionBackwardData(  cudnnHandle_t                       handle,
                                                         const void                         *alpha,
                                                         const cudnnFilterDescriptor_t       filterDesc,
                                                         const void                         *filterData,
                                                         const cudnnTensorDescriptor_t       diffDesc,
                                                         const void                         *diffData,
                                                         const cudnnConvolutionDescriptor_t  convDesc,
                                                         const void                         *beta,
                                                         const cudnnTensorDescriptor_t       gradDesc,
                                                         void                               *gradData
                                                       );
cudnnStatus_t CUDNNWINAPI kunetConvolutionBackwardBias(   cudnnHandle_t                   handle,
                                                          const void                     *alpha,
                                                          const cudnnTensorDescriptor_t   srcDesc,
                                                          const void                      *srcData,
                                                          const void                      *beta,
                                                          const cudnnTensorDescriptor_t   destDesc,
                                                          void                           *destData
                                                      );

cudnnStatus_t CUDNNWINAPI kunetPoolingForward(  cudnnHandle_t handle,
                                                const cudnnPoolingDescriptor_t   poolingDesc,
                                                const void                      *alpha,
                                                const cudnnTensorDescriptor_t    srcDesc,
                                                const void                      *srcData,
                                                const void                      *beta,
                                                const cudnnTensorDescriptor_t    destDesc,
                                                void                            *destData
                                             );

cudnnStatus_t CUDNNWINAPI kunetPoolingBackward( cudnnHandle_t                   handle,
                                                const cudnnPoolingDescriptor_t  poolingDesc,
                                                const void                      *alpha,
                                                const cudnnTensorDescriptor_t   srcDesc,
                                                const void                     *srcData,
                                                const cudnnTensorDescriptor_t   srcDiffDesc,
                                                const void                     *srcDiffData,
                                                const cudnnTensorDescriptor_t   destDesc,
                                                const void                     *destData,
                                                const void                     *beta,
                                                const cudnnTensorDescriptor_t   destDiffDesc,
                                                void                           *destDiffData
                                              );
