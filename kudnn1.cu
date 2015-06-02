#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <limits.h>



/**
  cudnnPoolingForward(pd::PoolingDescriptor, src::Tensor, dest::Tensor) Performs the pooling operation specified by pd on src, writes the result to dest and returns dest. The C and N dimensions of src and dest should match. If a src dimension (other than C,N) is x, and the corresponding pooling area dimension is d, padding is p, stride is s, then the corresponding dest dimension should be y=1+ceil((x+2p-d)/s).**/

__global__ void krnlMaxPool4d( double *src, int N, int C, int H, int W,
                            int Hp, int Wp,
                            double *dst, int Ho, int Wo, int hStride, int wStride){
    int h = threadIdx.x * hStride; // stride
    int w = threadIdx.y * wStride; // stride
    int i,j,l,m;
    int hsrc, wsrc;
    double maxm;
    for(i=0;i<N;i++){ for(j=0;j<C;j++){
        maxm=src[ind4d(C,H,W,i,j,h,w)];
        //maxm=INT_MIN;
        for(l=0; l<Hp;l++){
        for(m=0; m<Wp;m++){
            hsrc = h+l; wsrc = w+m;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                if(src[ind4d(C,H,W,i,j,hsrc,wsrc)] > maxm)
                    maxm = src[ind4d(C,H,W,i,j,hsrc,wsrc)];
        }}
        dst[ind4d(C,Ho,Wo,i,j,h,w)] = maxm; 
    }}
}

__global__ void krnlBackBias4d( double *src, int N, int C, int H, int W,
                            double *dst){
    int j = threadIdx.x;
    int i,k,l;
    double sum=0;
    for(i=0;i<N;i++){ 
        for(k=0;k<H;k++){ 
        for(l=0;l<W;l++){ 
        sum += src[ind4d(C,H,W,i,j,k,l)];
        }}
    }
    dst[ind4d(C,H,W,1,j,1,1)] = sum;
}
__global__ void krnlXCorr4dDw( double *src, int N, int C, int H, int W,
                            double *flt, int Hy, int Wy, int K,
                            double *dst, int Hw, int Ww, int hpad, int wpad){
    /* 
        src: (W,H,C,N)          (N,C,H,W)
        flt: (X,Y,C,K)          (K,C,Hf,Wf)
        dst: (W-X+1,H-Y+1,K,N)  (N,K,H-Hf+1,W-Wf+1)
    */
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j,k,l,m;
    double sum=0;
    int hsrc, wsrc;
    for(k=0;k<K;k++){ for(j=0;j<C;j++){
        sum=0;
        for(i=0;i<N;i++){ 
        for(l=0; l<Hy;l++){
        for(m=0; m<Wy;m++){
            hsrc = h+l-hpad; //int hsrc = h+Hf-1-l;
            wsrc = w+m-wpad;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(K,Hy,Wy,i,k,l,m)];
        }}}
        dst[ind4d(C,Hw,Ww,k,j,h,w)] = sum;
    } }
}

__global__ void krnlXCorr4dDwRot180( double *src, int N, int C, int H, int W,
                            double *flt, int Hy, int Wy, int K,
                            double *dst, int Hw, int Ww, int hpad, int wpad){
    /* 
        src: (W,H,C,N)          (N,C,H,W)
        flt: (X,Y,C,K)          (K,C,Hf,Wf)
        dst: (W-X+1,H-Y+1,K,N)  (N,K,H-Hf+1,W-Wf+1)
    */
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j,k,l,m;
    double sum=0;
    int hsrc, wsrc;
    for(k=0;k<K;k++){ for(j=0;j<C;j++){
        sum=0;
        for(i=0;i<N;i++){ 
        for(l=0; l<Hy;l++){
        for(m=0; m<Wy;m++){
            hsrc = h+l-hpad; //int hsrc = h+Hf-1-l;
            wsrc = w+m-wpad;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(K,Hy,Wy,i,k,l,m)];
        }}}
        dst[ind4d(C,Hw,Ww,k,j,Hw-h-1,Ww-w-1)] = sum;
    } }
}

__global__ void krnlXCorr4d( double *src, int N, int C, int H, int W,
                            double *flt, int Hf, int Wf, int K,
                            double *dst, int Ho, int Wo, int hpad, int wpad){
    /* 
        src: (W,H,C,N)          (N,C,H,W)
        flt: (X,Y,C,K)          (K,C,Hf,Wf)
        dst: (W-X+1,H-Y+1,K,N)  (N,K,H-Hf+1,W-Wf+1)
    */
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j,k,l,m;
    double sum=0;
    int hsrc, wsrc;
    for(i=0;i<N;i++){ for(k=0;k<K;k++){ 
        sum=0;
        for(j=0;j<C;j++){ 
        for(l=0; l<Hf;l++){
        for(m=0; m<Wf;m++){
            hsrc = h+l-hpad; //int hsrc = h+Hf-1-l;
            wsrc = w+m-wpad;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(C,Hf,Wf,k,j,l,m)];
        }}}
        dst[ind4d(K,Ho,Wo,i,k,h,w)] = sum;
    } }
}

__global__ void krnlConv4dDx( double *src, int N, int K, int H, int W,
                            double *flt, int Hf, int Wf, int C,
                            double *dst, int Ho, int Wo, int hpad, int wpad){
    /* 
        src: (W,H,C,N)          (N,C,H,W)
        flt: (X,Y,C,K)          (K,C,Hf,Wf)
        dst: (W-X+1,H-Y+1,K,N)  (N,K,H-Hf+1,W-Wf+1)
    */
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j,k,l,m;
    int hsrc, wsrc;
    double sum=0;
    for(i=0;i<N;i++){ for(j=0;j<C;j++){  
        sum=0;
        for(k=0;k<K;k++){
        for(l=Hf-1; l>=0;l--){
        for(m=Wf-1; m>=0;m--){
            hsrc = h+Hf-1-l-hpad; //int hsrc = h+Hf-1-l;
            wsrc = w+Wf-1-m-wpad;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                sum += src[ind4d(K,H,W,i,k,hsrc,wsrc)] * flt[ind4d(C,Hf,Wf,k,j,l,m)];
        }}}
        dst[ind4d(C,Ho,Wo,i,j,h,w)] = sum;
    }} 
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
    int pad_h, pad_w;
    int u,v; // strides
    int upscalex, upscaley; // upscale the input in x-direction/y-direction
    cudnnConvolutionMode_t mode;
    cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &u, &v, &upscalex, &upscaley, &mode);
    assert(u==1);assert(v==1);assert(upscalex==1);assert(upscaley==1); // other values are not supported yet.

    int N,C,H,W;
    int nStride, cStride, hStride, wStride; // not used
    cudnnDataType_t dataType; // image data type
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &C, &H, &W,
                            &nStride, &cStride, &hStride, &wStride);
    assert(dataType == CUDNN_DATA_DOUBLE);
    /*printf("pad %d %d\n", pad_h, pad_w);
    printf("src strides %d %d %d %d\n", nStride, cStride, hStride, wStride);
    printf("src: N C H W %d %d %d %d\n", N, C, H, W);*/


    int K,Cf,Hf,Wf;
    cudnnGetFilter4dDescriptor(filterDesc, &dataType, &K, &Cf, &Hf, &Wf);
    assert(dataType == CUDNN_DATA_DOUBLE);
    //printf("flt: K C H W %d %d %d %d\n", K, Cf, Hf, Wf);

    int No, Co, Ho, Wo;
    cudnnGetTensor4dDescriptor(destDesc, &dataType, &No, &Co, &Ho, &Wo,
                            &nStride, &cStride, &hStride, &wStride);
    assert(dataType == CUDNN_DATA_DOUBLE);
    //printf("dst: N C H W %d %d %d %d\n", No, Co, Ho, Wo);

    dim3 threads(Ho, Wo, 1); 
    dim3 grid(1,1,1);

    if(mode == CUDNN_CROSS_CORRELATION){
        krnlXCorr4d<<<grid,threads>>>(    (double *)srcData, N, C, H, W,
                                            (double *)filterData, Hf, Wf, K,
                                            (double *)destData, Ho, Wo, pad_h, pad_w);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        /*krnlConv4d<<<grid,threads>>>(    (double *)srcData, N, C, H, W,
                                            (double *)filterData, Hf, Wf, K,
                                            (double *)destData, Ho, Wo, pad_h, pad_w);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );*/
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
    int pad_h, pad_w;
    int u,v; // strides
    int upscalex, upscaley; // upscale the input in x-direction/y-direction
    cudnnConvolutionMode_t mode;
    cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &u, &v, &upscalex, &upscaley, &mode);
    assert(u==1);assert(v==1);assert(upscalex==1);assert(upscaley==1); // other values are not supported yet.

    int N,C,H,W;
    int nStride, cStride, hStride, wStride; // not used
    cudnnDataType_t dataType; // image data type
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &C, &H, &W, &nStride, &cStride, &hStride, &wStride);
    assert(dataType == CUDNN_DATA_DOUBLE);

    int Ny,Cy,Hy,Wy;
    cudnnGetTensor4dDescriptor(diffDesc, &dataType, &Ny, &Cy, &Hy, &Wy, &nStride, &cStride, &hStride, &wStride);
    assert(Ny==N);

    int K,Cw,Hw,Ww;
    cudnnGetFilter4dDescriptor(gradDesc, &dataType, &K, &Cw, &Hw, &Ww);
    assert(Cy==K); assert(Cw==C);


    dim3 threads(Hw, Ww, 1); 
    dim3 grid(1,1,1);

    if(mode == CUDNN_CROSS_CORRELATION){
        // xcorr(x,dy);
        krnlXCorr4dDw<<<grid,threads>>>(    (double *)srcData, N, C, H, W,
                                            (double *)diffData, Hy, Wy, K,
                                            (double *)gradData, Hw, Ww, 0, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        // rot180(xcorr(x,dy));
        //status = CUDNN_STATUS_NOT_SUPPORTED;
        krnlXCorr4dDwRot180<<<grid,threads>>>(    (double *)srcData, N, C, H, W,
                                            (double *)diffData, Hy, Wy, K,
                                            (double *)gradData, Hw, Ww, 0, 0);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
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
    int pad_h, pad_w;
    int u,v; // strides
    int nStride, cStride, hStride, wStride; // not used
    int upscalex, upscaley; // upscale the input in x-direction/y-direction
    cudnnConvolutionMode_t mode;
    cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &u, &v, &upscalex, &upscaley, &mode);
    assert(u==1);assert(v==1);assert(upscalex==1);assert(upscaley==1); // other values are not supported yet.

    int K,C,Hw,Ww;
    cudnnGetFilter4dDescriptor(filterDesc, &dataType, &K, &C, &Hw, &Ww);

    int N,Cy,Hy,Wy;
    cudnnGetTensor4dDescriptor(diffDesc, &dataType, &N, &Cy, &Hy, &Wy,
                            &nStride, &cStride, &hStride, &wStride);
    assert(Cy==K); 

    int Nx,Cx,H,W;
    cudnnGetTensor4dDescriptor(gradDesc, &dataType, &Nx, &Cx, &H, &W,
                            &nStride, &cStride, &hStride, &wStride); 
    assert(Nx==N);assert(Cx==C);

    dim3 threads(H, W, 1); 
    dim3 grid(1,1,1);

    if(mode == CUDNN_CROSS_CORRELATION){
        // conv(dy,w,'full');
        krnlConv4dDx<<<grid,threads>>>(    (double *)diffData, N, K, Hy, Wy,
                                            (double *)filterData, Hw, Ww, C,
                                            (double *)gradData, H, W, Hw-1, Ww-1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }else if(mode == CUDNN_CONVOLUTION){
        // xcorr(dy,w,'full')
        /*krnlXCorr4d<<<grid,threads>>>(    (double *)diffData, N, K, Hy, Wy,
                                            (double *)filterData, Hw, Ww, C,
                                            (double *)gradData, H, W, Hy-1, Wy-1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );*/
    }else{
        status = CUDNN_STATUS_BAD_PARAM;
    }
    return status;
}

/*
   cudnnConvolutionBackwardBias(src::Tensor, [dest::Tensor]) Given src=dJ/dy this function computes and returns dest=dJ/db. It is assumed that there is a single scalar bias for each channel, i.e. the same number is added to every pixel of every image for that channel after the convolution. Thus dJ/db is simply the sum of dJ/dy across each channel, i.e. dest=sum(src,(1,2,4)). For 2-D images if src has size (W,H,C,N), dest will have size (1,1,C,1). If dest is not specified it will be allocated.*/

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
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &K, &H, &W,
                            &nStride, &cStride, &hStride, &wStride);
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
    int N,K,Hy,Wy;
    int strides[4];
    cudnnGetTensor4dDescriptor(srcDesc, &dataType, &N, &K, &Hy, &Wy,
                            strides, strides+1, strides+2, strides+3);

    int Hp, Wp, hPad, wPad, hStride, wStride;
    cudnnGetPooling2dDescriptor(poolingDesc, &mode, &Hp, &Wp, &hPad, &wPad, &hStride, &wStride);
    printf("Hp:%d Wp:%d hStride:%d wStride:%d\n", Hp, Wp, hStride, wStride);

    int No,Ko,Hyp,Wyp;
    cudnnGetTensor4dDescriptor(destDesc, &dataType, &No, &Ko, &Hyp, &Wyp,
                            strides, strides+1, strides+2, strides+3);
    assert(N==No); assert(K==Ko);
    printf("Hy:%d Wy:%d\n", Hy, Wy);
    printf("Hyp:%d Wyp:%d\n", Hyp, Wyp);

    dim3 threads(Hyp, Wyp, 1); 
    dim3 grid(1,1,1);
    if(mode == CUDNN_POOLING_MAX){
        krnlMaxPool4d<<<grid,threads>>>((double *)srcData, N, K, Hy, Wy,
                Hp, Wp,
                (double *)destData, Hyp, Wyp, hStride, wStride);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }else{
        status = CUDNN_STATUS_NOT_SUPPORTED;
    }
    return status;
}
