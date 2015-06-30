#include <stdio.h>
#include <assert.h>
#include "kuassert.h"
#include "kudnn.h"
#include "util.h"
#include "xcorr.cuh"
#include "pool.cuh"

#define BLK 4096
#define THR 256

cudnnStatus_t CUDNNWINAPI kudnnConvolutionForward(        cudnnHandle_t                     handle,
                                                          const void                         *alpha,
                                                          const cudnnTensorDescriptor_t       srcDesc,
                                                          const void                         *src,
                                                          const cudnnFilterDescriptor_t       filterDesc,
                                                          const void                         *flt,
                                                          const cudnnConvolutionDescriptor_t  convDesc,
                                                          cudnnConvolutionFwdAlgo_t           algo,
                                                          void                               *workSpace,
                                                          size_t                              workSpaceSizeInBytes,            
                                                          const void                         *beta,
                                                          const cudnnTensorDescriptor_t       destDesc,
                                                          void                               *dst
                                                 ){
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

    int i;
    int ndimsreq=5; int convndimsreq=3;
    int wndims, wDims[ndimsreq];
    int xndims, xDims[ndimsreq], xStrides[ndimsreq];
    int yndims, yDims[ndimsreq], yStrides[ndimsreq];
    int convndims, convPad[convndimsreq], convStride[convndimsreq], convUpscale[convndimsreq];

    // x
    cudnnGetTensorNdDescriptor(srcDesc,  ndimsreq, &dataType, &xndims, xDims, xStrides);

    // w
    cudnnGetFilterNdDescriptor(filterDesc, ndimsreq, &dataType, &wndims, wDims);
    assert(xndims == wndims);

    // y
    cudnnGetTensorNdDescriptor(destDesc,  ndimsreq, &dataType, &yndims, yDims, yStrides);
    assert(xndims == yndims);

    cudnnGetConvolutionNdDescriptor(convDesc, convndimsreq, &convndims, convPad, convStride, convUpscale, &mode);
    assert(convndims==(xndims-2)); for(i=0; i<convndims; i++) assert(convStride[i]==1); for(i=0; i<convndims; i++) assert(convUpscale[i]==1);

    if(xndims == 4){ // 4-D
        xDims[4] = 1; wDims[4] = 1; yDims[4] = 1; 
        xStrides[4]=1; yStrides[4]=1;
        convPad[2] = 0; convStride[2] = 0; convUpscale[2] = 0;
    }

    if(mode == CUDNN_CROSS_CORRELATION){
        // xcorr(x,w)
        krnlXCorrY5d<<<BLK, THR>>>(
                (double *)src, cat5d(xDims),
                (double *)flt, cat5d(wDims),
                (double *)dst, cat5d(yDims),
                cat5d(yStrides), cat3d(convPad), prod5d(yDims));
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        // conv(x,w)
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }else{
        status = CUDNN_STATUS_BAD_PARAM;
    }
    return status;
}

cudnnStatus_t CUDNNWINAPI kudnnConvolutionBackwardFilter( cudnnHandle_t                       handle,
                                                          const void                         *alpha,
                                                          const cudnnTensorDescriptor_t       srcDesc,
                                                          const void                         *src,
                                                          const cudnnTensorDescriptor_t       diffDesc,
                                                          const void                         *dff,
                                                          const cudnnConvolutionDescriptor_t  convDesc,
                                                          const void                         *beta,
                                                          const cudnnFilterDescriptor_t       gradDesc,
                                                          void                               *grd
                                                        ){
    // dw = xcorr(x,dy)
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

    int i;
    int ndimsreq=5; int convndimsreq=3;
    int dwndims, dwDims[ndimsreq];
    int xndims, xDims[ndimsreq], xStrides[ndimsreq];
    int dyndims, dyDims[ndimsreq], dyStrides[ndimsreq];
    int convndims, convPad[convndimsreq], convStride[convndimsreq], convUpscale[convndimsreq];

    // x
    cudnnGetTensorNdDescriptor(srcDesc,  ndimsreq, &dataType, &xndims, xDims, xStrides);

    // dy
    cudnnGetTensorNdDescriptor(diffDesc,  ndimsreq, &dataType, &dyndims, dyDims, dyStrides);
    assert(xndims == dyndims);

    // dw
    cudnnGetFilterNdDescriptor(gradDesc, ndimsreq, &dataType, &dwndims, dwDims);
    assert(xndims == dwndims);

    cudnnGetConvolutionNdDescriptor(convDesc, convndimsreq, &convndims, convPad, convStride, convUpscale, &mode);
    assert(convndims==(xndims-2)); for(i=0; i<convndims; i++) assert(convStride[i]==1); for(i=0; i<convndims; i++) assert(convUpscale[i]==1);

    if(xndims == 4){ // 4-D
        xDims[4] = 1; dwDims[4] = 1; dyDims[4] = 1; 
        xStrides[4]=1; dyStrides[4]=1;
        convPad[2] = 0; convStride[2] = 0; convUpscale[2] = 0;
    }

    if(mode == CUDNN_CROSS_CORRELATION){
        // dw = xcorr(x,dy)
        krnlXCorrDw5d<<<BLK, THR>>>(
                (double *)src, cat5d(xDims),
                (double *)dff, cat5d(dyDims),
                (double *)grd, cat5d(dwDims),
                dims2strides5d(dwDims), cat3d(convPad), prod5d(dwDims));
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }else{
        status = CUDNN_STATUS_BAD_PARAM;
    }
    return status;
}

cudnnStatus_t CUDNNWINAPI kudnnConvolutionBackwardData(  cudnnHandle_t                       handle,
                                                         const void                         *alpha,
                                                         const cudnnFilterDescriptor_t       filterDesc,
                                                         const void                         *flt,
                                                         const cudnnTensorDescriptor_t       diffDesc,
                                                         const void                         *dff,
                                                         const cudnnConvolutionDescriptor_t  convDesc,
                                                         const void                         *beta,
                                                         const cudnnTensorDescriptor_t       gradDesc,
                                                         void                               *grd
                                                       ){

    // conv(dy,w,'full');
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

    int i;
    int ndimsreq=5; int convndimsreq=3;
    int dyndims, dyDims[ndimsreq], dyStrides[ndimsreq];
    int wndims, wDims[ndimsreq];
    int dxndims, dxDims[ndimsreq], dxStrides[ndimsreq];
    int convndims, convPad[convndimsreq], convStride[convndimsreq], convUpscale[convndimsreq];

    // dy
    cudnnGetTensorNdDescriptor(diffDesc,  ndimsreq, &dataType, &dyndims, dyDims, dyStrides);

    // w
    cudnnGetFilterNdDescriptor(filterDesc, ndimsreq, &dataType, &wndims, wDims);

    // dx
    cudnnGetTensorNdDescriptor(gradDesc,  ndimsreq, &dataType, &dxndims, dxDims, dxStrides);
    assert(dxndims == dyndims); assert(dxndims == wndims);

    cudnnGetConvolutionNdDescriptor(convDesc, convndimsreq, &convndims, convPad, convStride, convUpscale, &mode);
    assert(convndims==(dxndims-2)); for(i=0; i<convndims; i++) assert(convStride[i]==1); for(i=0; i<convndims; i++) assert(convUpscale[i]==1);

    if(dxndims == 4){ // 4-D
        dxDims[4] = 1; wDims[4] = 1; dyDims[4] = 1; 
        dxStrides[4]=1; dyStrides[4]=1;
        convPad[2] = 0; convStride[2] = 0; convUpscale[2] = 0;
    }

    if(mode == CUDNN_CROSS_CORRELATION){
        // conv(dy,w,'full');
        krnlXCorrDx5d<<<BLK, THR>>>(
                (double *)dff, cat5d(dyDims),
                (double *)flt, cat5d(wDims),
                (double *)grd, cat5d(dxDims),
                cat5d(dxStrides), cat3d(convPad), prod5d(dxDims));
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }else{
        status = CUDNN_STATUS_BAD_PARAM;
    }
    return status;
}

cudnnStatus_t CUDNNWINAPI kudnnConvolutionBackwardBias(   cudnnHandle_t                   handle,
                                                          const void                     *alpha,
                                                          const cudnnTensorDescriptor_t   srcDesc,
                                                          const void                      *srcData,
                                                          const void                      *beta,
                                                          const cudnnTensorDescriptor_t   destDesc,
                                                          void                           *destData
                                                      ){
    // dy -> db : N,K,Hy,Wy,Dy -> 1,K,1,1,1
    cudnnDataType_t dataType;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

    int ndimsreq=5;
    int dyndims, dyDims[ndimsreq], dyStrides[ndimsreq];

    // dy
    cudnnGetTensorNdDescriptor(srcDesc,  ndimsreq, &dataType, &dyndims, dyDims, dyStrides);

    if(dyndims == 4){ // 4-D
        dyDims[4] = 1; 
        dyStrides[4]=1;
    }

    dim3 threads(dyDims[1], 1, 1); 
    dim3 grid(1,1,1);
    krnlBackBias5d<<<grid,threads>>>(
            (double *)srcData, 
            cat5d(dyDims),
            (double *)destData
            );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    return status;
}

cudnnStatus_t CUDNNWINAPI kudnnPoolingForward(  cudnnHandle_t handle,
                                                const cudnnPoolingDescriptor_t   poolingDesc,
                                                const void                      *alpha,
                                                const cudnnTensorDescriptor_t    srcDesc,
                                                const void                      *src,
                                                const void                      *beta,
                                                const cudnnTensorDescriptor_t    destDesc,
                                                void                            *dst
                                             ){
    cudnnPoolingMode_t mode;
    cudnnDataType_t dataType;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

    int i;
    int ndimsreq=5, poolndimsreq=3, poolndims;
    int xndims, xDims[ndimsreq], xStrides[ndimsreq];
    int yndims, yDims[ndimsreq], yStrides[ndimsreq];
    int poolDims[poolndimsreq], poolPad[poolndimsreq], poolStride[poolndimsreq];

    // x
    cudnnGetTensorNdDescriptor(srcDesc,  ndimsreq, &dataType, &xndims, xDims, xStrides);

    // y
    cudnnGetTensorNdDescriptor(destDesc,  ndimsreq, &dataType, &yndims, yDims, yStrides);
    assert(xndims == yndims);

    cudnnGetPoolingNdDescriptor(poolingDesc, poolndimsreq, &mode, &poolndims, poolDims, poolPad, poolStride);
    for(i=0;i<poolndims;i++) assert(poolDims[i]>=poolStride[i]);

    if(xndims == 4){ // 4-D
        xDims[4] = 1; yDims[4] = 1; 
        xStrides[4]=1; yStrides[4]=1;
        poolDims[2] = 1; poolPad[2] = 0; poolStride[2] = 0;
    }

    if(mode == CUDNN_POOLING_MAX){
            krnlMaxPoolY5d<<<BLK,THR>>>(  
                    (double *)src,
                    cat5d(xDims),
                    cat3d(poolDims),
                    cat3d(poolStride),
                    (double *)dst,
                    cat5d(yDims),
                    cat5d(yStrides),
                    prod5d(yDims)
                    );
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
    }else{
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }
    return status;
}

cudnnStatus_t CUDNNWINAPI kudnnPoolingBackward( cudnnHandle_t                   handle,
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
    cudnnPoolingMode_t mode;
    cudnnDataType_t dataType;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

    int i;
    int ndimsreq=5, poolndimsreq=3, poolndims;
    int xndims, xDims[ndimsreq], xStrides[ndimsreq];
    int dxndims, dxDims[ndimsreq], dxStrides[ndimsreq];
    int yndims, yDims[ndimsreq], yStrides[ndimsreq];
    int dyndims, dyDims[ndimsreq], dyStrides[ndimsreq];
    int poolDims[poolndimsreq], poolPad[poolndimsreq], poolStride[poolndimsreq];

    // y
    cudnnGetTensorNdDescriptor(srcDesc,  ndimsreq, &dataType, &yndims, yDims, yStrides);

    // dy
    cudnnGetTensorNdDescriptor(srcDiffDesc,  ndimsreq, &dataType, &dyndims, dyDims, dyStrides);

    // x
    cudnnGetTensorNdDescriptor(destDesc,  ndimsreq, &dataType, &xndims, xDims, xStrides);

    // dx
    cudnnGetTensorNdDescriptor(destDiffDesc,  ndimsreq, &dataType, &dxndims, dxDims, dxStrides);

    cudnnGetPoolingNdDescriptor(poolingDesc, poolndimsreq, &mode, &poolndims, poolDims, poolPad, poolStride);
    for(i=0;i<poolndims;i++) assert(poolDims[i]>=poolStride[i]);

    if(xndims == 4){ // 4-D
        xDims[4] = 1; yDims[4] = 1; 
        xStrides[4]=1; yStrides[4]=1;
        poolDims[2] = 1; poolPad[2] = 0; poolStride[2] = 0;
    }

    if(mode == CUDNN_POOLING_MAX){
        krnlMaxPool5dDx<<<BLK,THR>>>( 
                (double *)srcData,
                cat5d(yDims),
                (double *)srcDiffData,
                (double *)destData,
                cat5d(xDims),
                (double *)destDiffData,
                cat3d(poolDims),
                cat3d(poolStride),
                cat5d(yStrides),
                prod5d(yDims)
                );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }else{
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }

    return status;
}
