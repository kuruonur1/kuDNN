#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <float.h>

#define BLK 4096
#define THR 256



// POOLING
__global__ void krnlMaxPool4d(  double *src, int N, int C, int H, int W,
                                int Hd, int Wd, int Hs, int Ws,
                                double *dst, int Hy, int Wy,
                                int NySt, int CySt, int HySt, int WySt, const int lim){
    int i,j,hy,wy,hx,wx, l,m, g,cumul, hsrc, wsrc;
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        j=cumul/CySt; cumul -= j*CySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt;

        hx = hy*Hs; wx = wy*Ws;
        double maxm = DBL_MIN; // maxm=src[ind4d(C,H,W,i,j,hx,wx)]; // will cause a problem when pad != 0!!!
        for(l=0; l<Hd;l++){ for(m=0; m<Wd;m++){
            hsrc = hx+l; wsrc = wx+m;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                if(src[ind4d(C,H,W,i,j,hsrc,wsrc)] > maxm)
                    maxm = src[ind4d(C,H,W,i,j,hsrc,wsrc)];
        }}
        dst[g] = maxm; 
    }
}
__global__ void krnlMaxPool4dDx( double *y, int N, int C, int Hy, int Wy,
                                double *dy,
                                double *x,  int H, int W,
                                double *dx,
                            int Hd, int Wd, int Hs, int Ws,
                            int NySt, int CySt, int HySt, int WySt, int lim){
    int i,j,hy,wy,hx,wx, l,m, g,cumul, hsrc, wsrc, maxmhx, maxmwx;
    double maxm;

    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        j=cumul/CySt; cumul -= j*CySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt;

        maxmhx = hx = hy*Hs; maxmwx = wx = wy*Ws;
        maxm = DBL_MIN;

        for(l=0; l<Hd;l++){
        for(m=0; m<Wd;m++){
            hsrc = hx+l; wsrc = wx+m;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W){
                if(x[ind4d(C,H,W,i,j,hsrc,wsrc)] > maxm){
                    maxm = x[ind4d(C,H,W,i,j,hsrc,wsrc)];
                    maxmhx = hsrc; maxmwx = wsrc;
                }
                // dx[ind4d(C,H,W,i,j,hsrc,wsrc)] = 0; // writing to global mem decreases prfrmce drastically
            }
        }}
        dx[ind4d(C,H,W,i,j,maxmhx,maxmwx)] = dy[ind4d(C,Hy,Wy,i,j,hy,wy)];
    }
}
// END POOLING


// CROSS CORRELATION
__global__ void krnlXCorrY4d(   double *src, int N, int C, int H, int W,
                                double *flt, int K, int Cw, int Hw, int Ww, 
                                double *dst, int Ny, int Ky, int Hy, int Wy,
                                int NySt, int KySt, int HySt, int WySt,
                                int hpad, int wpad, const int lim){
    // mode:xcorr y=xcorr(x,w)
    // output N K Hy Wy
    int i,k,hy,wy, j,l,m, g,cumul, hsrc,wsrc,hx,wx;
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        k=cumul/KySt; cumul -= k*KySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt;

        hx=hy-hpad; wx=wy-wpad;
        double sum=0;
        for(j=0;j<C;j++){ 
        for(l=0; l<Hw;l++){
        for(m=0; m<Ww;m++){
            hsrc = hx+l;
            wsrc = wx+m;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(C,Hw,Ww,k,j,l,m)];
        }}}
        dst[g] = sum; // dst[ind4d(K,Hy,Wy,i,k,h,w)] = sum;
    }
}

__global__ void krnlXCorrDw4d(  double* src, int N, int C, int H, int W,
                                double* flt, int Ny, int Ky, int Hy, int Wy,
                                double* dst, int K, int Cw, int Hw, int Ww, 
                                int KSt, int CSt, int HwSt, int WwSt,
                                int hpad, int wpad, const int lim){
    // mode:xcorr dw = xcorr(x,dy)
    int i,k,hw,ww, j,l,m, g,cumul, hsrc,wsrc,hx,wx; // k j

    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        k=cumul/KSt; cumul -= k*KSt;
        j=cumul/CSt; cumul -= j*CSt;
        hw=cumul/HwSt; cumul -= hw*HwSt;
        ww=cumul/WwSt;

        hx=hw-hpad; wx=ww-wpad;
        double sum=0;
        for(i=0;i<N;i++){ 
        for(l=0; l<Hy;l++){
        for(m=0; m<Wy;m++){
            hsrc = hx+l;
            wsrc = wx+m;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(K,Hy,Wy,i,k,l,m)];
        }}}
        dst[g] = sum; // dst[ind4d(C,Hw,Ww,k,j,h,w)] = sum;
    }
}

__global__ void krnlXCorrDx4d(  double *src, int Ny, int Ky, int Hy, int Wy,
                                double *flt, int K, int Cw, int Hw, int Ww, 
                                double *dst, int N, int C, int H, int W,
                                int NSt, int CSt, int HSt, int WSt,
                                int hpad, int wpad, const int lim){
    // mode:xcorr dx=conv(dy,w,'full')
    int i,j,h,w, k,l,m, g,cumul, hsrc, wsrc; // i j
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NSt; cumul -= i*NSt;
        j=cumul/CSt; cumul -= j*CSt;
        h=cumul/HSt; cumul -= h*HSt;
        w=cumul/WSt;

        double sum=0;
        for(k=0;k<K;k++){
        for(l=Hw-1; l>=0;l--){
        for(m=Ww-1; m>=0;m--){
            hsrc = h+Hw-1-l-hpad; //int hsrc = h+Hf-1-l;
            wsrc = w+Ww-1-m-wpad;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < Hy && wsrc < Wy) 
                sum += src[ind4d(K,Hy,Wy,i,k,hsrc,wsrc)] * flt[ind4d(C,Hw,Ww,k,j,l,m)];
        }}}
        dst[g] = sum; // dst[ind4d(C,Hy,Wy,i,j,h,w)] = sum;
    }

}
// END CROSS CORRELATION

__global__ void krnlBackBias4d( double *src, int N, int C, int H, int W,
                            double *dst){
    int j = threadIdx.x;
    int i,k,l;
    double sum=0;
    for(i=0;i<N;i++) 
        for(k=0;k<H;k++) 
            for(l=0;l<W;l++) 
                sum += src[ind4d(C,H,W,i,j,k,l)];
    dst[j] = sum;
}

// CONVOLUTION
/*
__global__ void krnlConv4d( double *src, int N, int C, int H, int W,
                            double *flt, int Hf, int Wf, int K,
                            double *dst, int Ho, int Wo, int hpad, int wpad){
    // mode:conv y=conv(x,w)
    int h = blockIdx.x; int w = blockIdx.y; int i = threadIdx.x; int k = threadIdx.y; 
    int j,l,m;
    int hsrc, wsrc;

    double sum=0;
    for(j=0;j<C;j++){ 
    for(l=Hf-1; l>=0;l--){
    for(m=Wf-1; m>=0;m--){
        hsrc = h+Hf-1-l-hpad; //int hsrc = h+Hf-1-l;
        wsrc = w+Wf-1-m-wpad;
        if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
            sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(C,Hf,Wf,k,j,l,m)];
    }}}
    dst[ind4d(K,Ho,Wo,i,k,h,w)] = sum;
}



__global__ void krnlXCorr4dDx( double *src, int N, int K, int Hy, int Wy,
                            double *flt, int Hw, int Ww, int C,
                            double *dst, int H, int W, int hpad, int wpad){
    // mode:conv dx=xcorr(dy,w,'full')
    int i = blockIdx.x; int j = blockIdx.y; 
    int h = threadIdx.x; int w = threadIdx.y; 
    int k,l,m;
    double sum=0;
    int hsrc, wsrc;
    for(k=0;k<K;k++){
    for(l=0; l<Hw;l++){
    for(m=0; m<Ww;m++){
        hsrc = h+l-hpad; //int hsrc = h+Hf-1-l;
        wsrc = w+m-wpad;
        if(hsrc >= 0 && wsrc >= 0 && hsrc < Hy && wsrc < Wy) 
            sum += src[ind4d(K,Hy,Wy,i,k,hsrc,wsrc)] * flt[ind4d(C,Hw,Ww,k,j,l,m)];
    }}}
    dst[ind4d(C,H,W,i,j,h,w)] = sum;
}


__global__ void krnlXCorr4dDwRot180( double *src, int N, int C, int H, int W,
                            double *flt, int Hy, int Wy, int K,
                            double *dst, int Hw, int Ww, int hpad, int wpad){
    // mode:conv dw=rot180(xcorr(x,dy))
    int k = blockIdx.x; int j = blockIdx.y; 
    int h = threadIdx.x; int w = threadIdx.y; 
    int i,l,m;
    double sum=0;
    int hsrc, wsrc;
    for(i=0;i<N;i++){ 
    for(l=0; l<Hy;l++){
    for(m=0; m<Wy;m++){
        hsrc = h+l-hpad; //int hsrc = h+Hf-1-l;
        wsrc = w+m-wpad;
        if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
            sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(K,Hy,Wy,i,k,l,m)];
    }}}
    dst[ind4d(C,Hw,Ww,k,j,Hw-h-1,Ww-w-1)] = sum;
}
*/
// END CONVOLUTION

// CUDNN LIKE API

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
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnDataType_t dataType;
    
    cudnnConvolutionMode_t mode;
    int convHPad, convWPad, convHSt, convWSt, convUpX, convUpY;
    cudnnGetConvolution2dDescriptor(convDesc, &convHPad, &convWPad, &convHSt, &convWSt, &convUpX, &convUpY, &mode);
    assert(convUpX==1);assert(convUpY==1); // other values are not supported yet.
    assert(convHSt==1);assert(convWSt==1); // other values are not supported yet.

    int N,C,H,W;
    int NSt, CSt, HSt, WSt; // not used
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &C, &H, &W, &NSt, &CSt, &HSt, &WSt);
    assert(dataType == CUDNN_DATA_DOUBLE);

    int Kw,Cw,Hw,Ww;
    cudnnGetFilter4dDescriptor(filterDesc, &dataType, &Kw, &Cw, &Hw, &Ww);
    assert(dataType == CUDNN_DATA_DOUBLE);

    int Ny, Ky, Hy, Wy;
    int NySt, KySt, HySt, WySt;
    cudnnGetTensor4dDescriptor(destDesc, &dataType, &Ny, &Ky, &Hy, &Wy, &NySt, &KySt, &HySt, &WySt);
    assert(dataType == CUDNN_DATA_DOUBLE);


    if(mode == CUDNN_CROSS_CORRELATION){
        // xcorr(x,w)
        krnlXCorrY4d<<<BLK,THR>>>(  (double *)src, N, C, H, W,
                                    (double *)flt, Kw, Cw, Hw, Ww, 
                                    (double *)dst, Ny, Ky, Hy, Wy,
                                    NySt, KySt, HySt, WySt,
                                    convHPad, convWPad, Ny*Ky*Hy*Wy);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        // conv(x,w)
        /*dim3 grid(Ho, Wo, 1); 
        dim3 threads(N,K,1);
        krnlConv4d<<<grid,threads>>>((double *)src, N, C, H, W,
                                    (double *)flt, Hf, Wf, K,
                                    (double *)dst, Ho, Wo, convHPad, convWPad);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );*/
        status = CUDNN_STATUS_NOT_SUPPORTED;
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

    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnDataType_t dataType;

    cudnnConvolutionMode_t mode;
    int convHPad, convWPad, convHSt, convWSt, convUpX, convUpY;
    cudnnGetConvolution2dDescriptor(convDesc, &convHPad, &convWPad, &convHSt, &convWSt, &convUpX, &convUpY, &mode);
    assert(convUpX==1);assert(convUpY==1); // other values are not supported yet.
    assert(convHSt==1);assert(convWSt==1); // other values are not supported yet.

    int N,C,H,W;
    int NSt, CSt, HSt, WSt;
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &C, &H, &W, &NSt, &CSt, &HSt, &WSt);
    assert(dataType == CUDNN_DATA_DOUBLE);

    int Ny,Ky,Hy,Wy;
    int NySt, KySt, HySt, WySt;
    cudnnGetTensor4dDescriptor(diffDesc, &dataType, &Ny, &Ky, &Hy, &Wy, &NySt, &KySt, &HySt, &WySt);
    assert(Ny==N);

    int Kw,Cw,Hw,Ww;
    cudnnGetFilter4dDescriptor(gradDesc, &dataType, &Kw, &Cw, &Hw, &Ww);
    assert(Ky==Kw); assert(Cw==C);


    if(mode == CUDNN_CROSS_CORRELATION){
        // xcorr(x,dy);
        krnlXCorrDw4d<<<BLK,THR>>>( (double *)srcData, N, C, H, W,
                                    (double *)diffData, Ny, Ky, Hy, Wy,
                                    (double *)gradData, Kw, Cw, Hw, Ww, 
                                    C*Hw*Ww, Hw*Ww, Ww, 1,
                                    convHPad, convWPad, Kw*Cw*Hw*Ww);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        // rot180(xcorr(x,dy));
        /*
        krnlXCorr4dDwRot180<<<grid,threads>>>(    (double *)srcData, N, C, H, W,
                                            (double *)diffData, Hy, Wy, K,
                                            (double *)gradData, Hw, Ww, 0, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
                                            */
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }else{
        status = CUDNN_STATUS_BAD_PARAM;
    }
    return status;
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
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnDataType_t dataType; // image data type

    cudnnConvolutionMode_t mode;
    int convHPad, convWPad, convHSt, convWSt, convUpX, convUpY;
    cudnnGetConvolution2dDescriptor(convDesc, &convHPad, &convWPad, &convHSt, &convWSt, &convUpX, &convUpY, &mode);
    assert(convUpX==1);assert(convUpY==1); // other values are not supported yet.
    assert(convHSt==1);assert(convWSt==1); // other values are not supported yet.

    int Kw,Cw,Hw,Ww;
    cudnnGetFilter4dDescriptor(filterDesc, &dataType, &Kw, &Cw, &Hw, &Ww);

    int Ny,Ky,Hy,Wy;
    int NySt, KySt, HySt, WySt;
    cudnnGetTensor4dDescriptor(diffDesc, &dataType, &Ny, &Ky, &Hy, &Wy, &NySt, &KySt, &HySt, &WySt);
    assert(Ky==Kw); 

    int N,C,H,W;
    int NSt, CSt, HSt, WSt;
    cudnnGetTensor4dDescriptor(gradDesc, &dataType, &N, &C, &H, &W, &NSt, &CSt, &HSt, &WSt); 
    assert(Ny==N);assert(C==Cw);

    if(mode == CUDNN_CROSS_CORRELATION){
        // conv(dy,w,'full');
        krnlXCorrDx4d<<<BLK,THR>>>( (double *)diffData, Ny, Ky, Hy, Wy,
                                    (double *)filterData, Kw, Cw, Hw, Ww, 
                                    (double *)gradData, N, C, H, W,
                                    C*H*W, H*W, W, 1,
                                    Hw-1-convHPad, Ww-1-convWPad, N*C*H*W);
        /*krnlXcorr4dDx<<<BLK,threads>>>(    (double *)diffData, N, K, Hy, Wy,
                                            (double *)filterData, Hw, Ww, C,
                                            (double *)gradData, H, W, Hw-1, Ww-1);*/
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        // xcorr(dy,w,'full')
        /*
        krnlXCorr4dDx<<<grid,threads>>>(    (double *)diffData, N, K, Hy, Wy,
                                            (double *)filterData, Hw, Ww, C,
                                            (double *)gradData, H, W, Hw-1, Ww-1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        */
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
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnDataType_t dataType; // image data type
    int N,K,H,W;
    int nStride, cStride, hStride, wStride;
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &K, &H, &W, &nStride, &cStride, &hStride, &wStride);
    dim3 threads(K, 1, 1); 
    dim3 grid(1,1,1);
    krnlBackBias4d<<<grid,threads>>>((double *)srcData, N, K, H, W, (double *)destData);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    return status;
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
    //y=1+ceil((x+2p-d)/s)
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnDataType_t dataType; // image data type
    cudnnPoolingMode_t mode;

    int strides[4];

    int N,C,H,W;
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &C, &H, &W, strides, strides+1, strides+2, strides+3);

    int Hd, Wd, Hp, Wp, Hs, Ws;
    cudnnGetPooling2dDescriptor(poolingDesc, &mode, &Hd, &Wd, &Hp, &Wp, &Hs, &Ws);
    assert(Hp==0); assert(Wp==0);

    int No,K,Hy,Wy;
    cudnnGetTensor4dDescriptor(destDesc, &dataType, &No, &K, &Hy, &Wy, strides, strides+1, strides+2, strides+3);
    assert(N==No); assert(C==K);

    /*
    printf("N:%d C:%d H:%d W:%d\n",N,C,H,W);
    printf("Hd:%d Wd:%d Hs:%d Ws:%d Hp:%d Wp:%d\n",Hd,Wd,Hs,Ws,Hp,Wp);
    printf("N:%d K:%d Hy:%d Wy:%d\n",N,C,Hy,Wy);
    */

    if(mode == CUDNN_POOLING_MAX){
        krnlMaxPool4d<<<BLK,THR>>>(     (double *)srcData, N, C, H, W,
                                        Hd, Wd, Hs, Ws,
                                        (double *)destData, Hy, Wy,
                                        C*Hy*Wy, Hy*Wy, Wy, 1, N*C*Hy*Wy);
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
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnDataType_t dataType; // image data type
    cudnnPoolingMode_t mode;
    int strides[4];

    int Hd, Wd, Hp, Wp, Hs, Ws;
    cudnnGetPooling2dDescriptor(poolingDesc, &mode, &Hd, &Wd, &Hp, &Wp, &Hs, &Ws);
    assert(Hp==0); assert(Wp==0);

    int N,K,Hy,Wy;
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &K, &Hy, &Wy, strides, strides+1, strides+2, strides+3);
    int Ndy,Kdy,Hdy,Wdy;
    cudnnGetTensor4dDescriptor(srcDiffDesc, &dataType, &Ndy, &Kdy, &Hdy, &Wdy, strides, strides+1, strides+2, strides+3);

    int Nx,C,H,W;
    cudnnGetTensor4dDescriptor(destDesc, &dataType, &Nx, &C, &H, &W, strides, strides+1, strides+2, strides+3);
    int Ndx,Cdx,Hdx,Wdx;
    cudnnGetTensor4dDescriptor(destDiffDesc, &dataType, &Ndx, &Cdx, &Hdx, &Wdx, strides, strides+1, strides+2, strides+3);
    dim3 grid(N,K,1);
    dim3 threads(Hy, Wy, 1); 
    if(mode == CUDNN_POOLING_MAX){
    //krnlMaxPool4dDx<<<grid,threads>>>((double *)srcData, N, C, Hy, Wy,
    krnlMaxPool4dDx<<<BLK,THR>>>((double *)srcData, N, C, Hy, Wy,
                                (double *)srcDiffData,
                                (double *)destData, H, W,
                                (double *)destDiffData,
                                    Hd, Wd, Hs, Ws,
                                    C*Hy*Wy, Hy*Wy, Wy, 1, N*C*Hy*Wy);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }else{
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }
    return status;
}
