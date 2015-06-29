#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <float.h>

#define BLK 4096
#define THR 256

// POOLING

__global__ void krnlMaxPoolY5d(  
        double *src,
        int N, int C, int H, int W, int D,
        int Hd, int Wd, int Dd,
        int Hs, int Ws, int Ds,
        double *dst,
        int Ny, int Ky, int Hy, int Wy, int Dy,
        int NySt, int KySt, int HySt, int WySt, int DySt,
        const int lim
        ){
    int i,j,l,m,n, g,cumul, hy,wy,dy,hx,wx,dx,hsrc,wsrc,dsrc;
    double maxm;
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        j=cumul/KySt; cumul -= j*KySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt; cumul -= wy*WySt;
        dy=cumul/DySt;

        hx = hy*Hs; wx = wy*Ws; dx= dy*Ds;
        maxm = DBL_MIN;
        for(l=0; l<Hd;l++){
        for(m=0; m<Wd;m++){
        for(n=0; n<Dd;n++){
            hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
            if(hsrc >= 0 && wsrc >= 0 && dsrc >=0 && hsrc < H && wsrc < W && dsrc < D) 
                if(src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] > maxm)
                    maxm = src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)];
        }}}
        dst[g] = maxm; 
    }
}

__global__ void krnlMaxPool5dDx( 
        double *y, int Ny, int Ky, int Hy, int Wy, int Dy,
        double *diffy,
        double *x,  int N, int C, int H, int W, int D,
        double *diffx,
        int Hd, int Wd, int Dd,
        int Hs, int Ws, int Ds,
        int NySt, int CySt, int HySt, int WySt, int DySt,
        const int lim
        ){
    int i,j,l,m,n, g,cumul, hy,wy,dy,hx,wx,dx, hsrc,wsrc,dsrc, maxmhx,maxmwx,maxmdx;
    double maxm;

    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        j=cumul/CySt; cumul -= j*CySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt; cumul -= wy*WySt;
        dy=cumul/DySt;

        maxmhx = hx = hy*Hs; maxmwx = wx = wy*Ws; maxmdx = dx = dy*Ds;
        maxm = DBL_MIN;

        for(l=0; l<Hd;l++){
        for(m=0; m<Wd;m++){
        for(n=0; n<Dd;n++){
            hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
            if(hsrc >= 0 && wsrc >= 0 && dsrc >=0 && hsrc < H && wsrc < W && dsrc < D){
                if(x[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] > maxm){
                    maxm = x[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)]; // update maxm
                    maxmhx = hsrc; maxmwx = wsrc; maxmdx = dsrc; // update indexes of maxm
                }
            }
        }}}
        diffx[ind5d(C,H,W,D,i,j,maxmhx,maxmwx,maxmdx)] = diffy[ind5d(C,Hy,Wy,Dy,i,j,hy,wy,dy)];
    }
}

// end POOLING


// CROSS CORRELATION

__global__ void krnlXCorrY5d(
        double *src, int N, int C, int H, int W, int D,
        double *flt, int Kw, int Cw, int Hw, int Ww, int Dw,
        double *dst, int Ny, int Ky, int Hy, int Wy, int Dy,
        int NySt, int KySt, int HySt, int WySt, int DySt,
        int hpad, int wpad, int dpad,
        const int lim
        ){ // xcorr(x,w)
    int i,k,hy,wy,dy, j,l,m,n, g,cumul, hsrc,wsrc,dsrc, hx,wx,dx;
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        k=cumul/KySt; cumul -= k*KySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt; cumul -= wy*WySt;
        dy=cumul/DySt;

        hx=hy-hpad; wx=wy-wpad; dx=dy-dpad;

        double sum=0;
        for(j=0;j<C;j++){ 
            for(l=0; l<Hw;l++){
            for(m=0; m<Ww;m++){
            for(n=0; n<Dw;n++){
                hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
                if(hsrc >= 0 && wsrc >= 0 &&  dsrc >= 0 &&  hsrc < H && wsrc < W && dsrc < D) 
                    sum += src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] * flt[ind5d(C,Hw,Ww,Dw,k,j,l,m,n)];
            }}}
        }
        dst[g] = sum;
    }
}

__global__ void krnlXCorrDw5d(
        double* src, int N, int C, int H, int W, int D,
        double* flt, int Ny, int Ky, int Hy, int Wy, int Dy,
        double* dst, int Kw, int Cw, int Hw, int Ww, int Dw,
        int KwSt, int CwSt, int HwSt, int WwSt, int DwSt,
        int hpad, int wpad, int dpad,
        const int lim
        ){
    // dw = xcorr(x,dy)
    int i,j,k, l,m,n, g,cumul, hw,ww,dw, hsrc,wsrc,dsrc, hx,wx,dx; // k j

    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        k=cumul/KwSt; cumul -= k*KwSt;
        j=cumul/CwSt; cumul -= j*CwSt;
        hw=cumul/HwSt; cumul -= hw*HwSt;
        ww=cumul/WwSt; cumul -= ww*WwSt;
        dw=cumul/DwSt;

        hx=hw-hpad; wx=ww-wpad; dx=dw-dpad;

        double sum=0;
        for(i=0;i<N;i++){ 
            for(l=0; l<Hy;l++){
            for(m=0; m<Wy;m++){
            for(n=0; n<Dy;n++){
                hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
                if(hsrc >= 0 && wsrc >= 0 &&  dsrc >= 0 &&  hsrc < H && wsrc < W && dsrc < D) 
                    sum += src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] * flt[ind5d(Ky,Hy,Wy,Dy,i,k,l,m,n)];
            }}}
        }
        dst[g] = sum;
    }
}

__global__ void krnlXCorrDx5d(
        double *src, int Ny, int Ky, int Hy, int Wy, int Dy,
        double *flt, int Kw, int Cw, int Hw, int Ww, int Dw,
        double *dst, int N, int C, int H, int W, int D,
        int NSt, int CSt, int HSt, int WSt, int DSt,
        int hpad, int wpad, int dpad,
        const int lim
        ){
    // dx=conv(dy,w,'full')
    int i,j,k, l,m,n, g,cumul, h,w,d, hsrc,wsrc,dsrc, hy,wy,dy; // i j

    hpad = Hw-1-hpad; wpad = Ww-1-wpad; dpad = Dw-1-dpad; // full convolution
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NSt; cumul -= i*NSt;
        j=cumul/CSt; cumul -= j*CSt;
        h=cumul/HSt; cumul -= h*HSt;
        w=cumul/WSt; cumul -= w*WSt;
        d=cumul/DSt;

        hy=h-hpad; wy=w-wpad; dy=d-dpad;

        double sum=0;
        for(k=0;k<Ky;k++){
            for(l=Hw-1; l>=0;l--){
            for(m=Ww-1; m>=0;m--){
            for(n=Dw-1; n>=0;n--){
                hsrc = hy+Hw-1-l; //int hsrc = h+Hf-1-l;
                wsrc = wy+Ww-1-m;
                dsrc = dy+Dw-1-n;
                if(hsrc >= 0 && wsrc >= 0 && dsrc >= 0 && hsrc < Hy && wsrc < Wy && dsrc < Dy) 
                    sum += src[ind5d(Ky,Hy,Wy,Dy,i,k,hsrc,wsrc,dsrc)] * flt[ind5d(Cw,Hw,Ww,Dw,k,j,l,m,n)];
            }}}
        }
        dst[g] = sum;
    }
}

// end CROSS CORRELATION

__global__ void krnlBackBias5d(
        double *src, int N, int C, int H, int W, int D, double *dst
        ){
    int j = threadIdx.x;
    int i,k,l,m;
    double sum=0;
    for(i=0;i<N;i++) for(k=0;k<H;k++) for(l=0;l<W;l++) for(m=0;m<D;m++) sum += src[ind5d(C,H,W,D,i,j,k,l,m)];
    dst[j] = sum;
}


cudnnStatus_t CUDNNWINAPI kunetConvolutionForward(        cudnnHandle_t                     handle,
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

cudnnStatus_t CUDNNWINAPI kunetConvolutionBackwardFilter( cudnnHandle_t                       handle,
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

cudnnStatus_t CUDNNWINAPI kunetConvolutionBackwardData(  cudnnHandle_t                       handle,
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

cudnnStatus_t CUDNNWINAPI kunetConvolutionBackwardBias(   cudnnHandle_t                   handle,
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

cudnnStatus_t CUDNNWINAPI kunetPoolingForward(  cudnnHandle_t handle,
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
