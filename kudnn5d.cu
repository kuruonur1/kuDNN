#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <limits.h>


__global__ void krnlXCorr5d(double *src, int N, int C, int H, int W, int D,
                            double *flt, int Kw, int Cw, int Hf, int Wf, int Df,
                            double *dst, int Ny, int K, int Hy, int Wy, int Dy,
                            int hpad, int wpad, int dpad){
    int i = threadIdx.x, k = threadIdx.y; 
    int h = blockIdx.x, w = blockIdx.y, d = blockIdx.z;
    int j,l,m,n;
    int hsrc, wsrc, dsrc;

    double sum=0;
    for(j=0;j<C;j++){ 
        for(l=0; l<Hf;l++){
        for(m=0; m<Wf;m++){
        for(n=0; n<Df;n++){
            hsrc = h+l-hpad; wsrc = w+m-wpad; dsrc = d+n-dpad;
            if(hsrc >= 0 && wsrc >= 0 &&  dsrc >= 0 &&  hsrc < H && wsrc < W && dsrc < D) 
                sum += src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] * flt[ind5d(C,Hf,Wf,Df,k,j,l,m,n)];
        }}}
    }
    dst[ind5d(K,Hy,Wy,Dy,i,k,h,w,d)] = sum;
}


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
                                                 ){
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnDataType_t dataType; // image data type
    cudnnConvolutionMode_t mode;
    int ndimsreq=5; int convndimsreq=3;

    int xndims, xDims[ndimsreq], xStrides[ndimsreq];
    cudnnGetTensorNdDescriptor(srcDesc,  ndimsreq, &dataType, &xndims, xDims, xStrides);
    assert(dataType == CUDNN_DATA_DOUBLE);

    int wndims, wDims[ndimsreq];
    cudnnGetFilterNdDescriptor(filterDesc, ndimsreq, &dataType, &wndims, wDims);
    assert(dataType == CUDNN_DATA_DOUBLE);
    printf("%d %d\n", xndims, wndims);
    assert(xndims == wndims);

    int yndims, yDims[ndimsreq], yStrides[ndimsreq];
    cudnnGetTensorNdDescriptor(destDesc,  ndimsreq, &dataType, &yndims, yDims, yStrides);
    assert(dataType == CUDNN_DATA_DOUBLE);
    assert(xndims == yndims);

    int convndims, convPad[convndimsreq], convStride[convndimsreq], convUpscale[convndimsreq];
    cudnnGetConvolutionNdDescriptor(convDesc, convndimsreq, &convndims, convPad, convStride, convUpscale, &mode);
    assert(convndims==(xndims-2));

    printf("N:%d C:%d H:%d W:%d D:%d\n",        cat5d(xDims));
    printf("K:%d C:%d Hw:%d Ww:%d Dw:%d\n",     cat5d(wDims));
    printf("N:%d K:%d Hy:%d Wy:%d Dy:%d\n",     cat5d(yDims));
    printf("\n");


    dim3 threads(   yDims[0],   yDims[1],   1); // N K
    dim3 grid(      yDims[2],   yDims[3],   yDims[4]); // Hy Wy Dy
    if(mode == CUDNN_CROSS_CORRELATION){
        // xcorr(x,w)
        krnlXCorr5d<<<grid, threads>>>((double *)srcData, cat5d(xDims),
                            (double *)filterData, cat5d(wDims),
                            (double *)destData, cat5d(yDims), convPad[0], convPad[1], convPad[2]);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        // conv(x,w)
        status = CUDNN_STATUS_NOT_SUPPORTED;
        /*krnlConv4d<<<grid,threads>>>((double *)srcData, N, C, H, W,
                                    (double *)filterData, Hf, Wf, K,
                                    (double *)destData, Ho, Wo, pad_h, pad_w);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );*/
        //status = CUDNN_STATUS_NOT_SUPPORTED;
    }else{
        status = CUDNN_STATUS_BAD_PARAM;
    }
    return status;
}

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
                                                        ){
    return CUDNN_STATUS_NOT_SUPPORTED;
}

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
                                                       ){

    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t CUDNNWINAPI kunetConvolutionBackwardBias(   cudnnHandle_t                   handle,
                                                          const void                     *alpha,
                                                          const cudnnTensorDescriptor_t   srcDesc,
                                                          const void                      *srcData,
                                                          const void                      *beta,
                                                          const cudnnTensorDescriptor_t   destDesc,
                                                          void                           *destData
                                                      ){
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t CUDNNWINAPI kunetPoolingForward(  cudnnHandle_t handle,
                                                const cudnnPoolingDescriptor_t   poolingDesc,
                                                const void                      *alpha,
                                                const cudnnTensorDescriptor_t    srcDesc,
                                                const void                      *srcData,
                                                const void                      *beta,
                                                const cudnnTensorDescriptor_t    destDesc,
                                                void                            *destData
                                             ){
    return CUDNN_STATUS_NOT_SUPPORTED;
}

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
                                              ){
    return CUDNN_STATUS_NOT_SUPPORTED;
}
