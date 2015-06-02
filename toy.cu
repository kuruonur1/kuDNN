#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"

__global__ void krnlCorr2dPadding(   double *src, int H, int W,
                            double *flt, int Hf, int Wf,
                            double *dst, int Ho, int Wo, int pad){
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j;
    double sum=0;
    for(i=0; i<Hf;i++){
        for(j=0; j<Wf;j++){
           //sum += src[srcW*(h+i)+(w+j)] * flt[i*fltW+j];
           sum += src[ind2d(W,h+i,w+j)] * flt[ind2d(Wf,i,j)];
        }
    }
    dst[ind2d(Wo,h,w)] = sum;
}

__global__ void krnlCorr2d(   double *src, int srcH, int srcW,
                            double *flt, int fltH, int fltW,
                            double *dst, int dstH, int dstW){
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j;
    double sum=0;
    for(i=0; i<fltH;i++){
        for(j=0; j<fltW;j++){
           sum += src[srcW*(h+i)+(w+j)] * flt[i*fltW+j];
        }
    }
    dst[dstW*h+w] = sum;
}

__global__ void krnlCorr5d( double *src, int N, int C, int H, int W, int D,
                            double *flt, int Hf, int Wf, int Df, int K,
                            double *dst, int Ho, int Wo, int Do){
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int d = threadIdx.z; 

    int i,j,k,l,m,n;
    double sum=0;
    for(i=0;i<N;i++){ for(j=0;j<C;j++){ for(k=0;k<K;k++){
        sum=0;
        for(l=0; l<Hf;l++){
        for(m=0; m<Wf;m++){
        for(n=0; n<Df;n++){
            sum += src[ind5d(C,H,W,D,i,j,h+l,w+m,d+n)] * flt[ind5d(C,Hf,Wf,Df,k,j,l,m,n)];
           // sum += src[srcW*(h+i)+(w+j)] * flt[i*fltW+j];
        }}}
        //dst[dstW*h+w] = sum;
        dst[ind5d(K,Ho,Wo,Do,i,k,h,w,d)] = sum;
    } } }
}

__global__ void krnlConv5d( double *src, int N, int C, int H, int W, int D,
                            double *flt, int Hf, int Wf, int Df, int K,
                            double *dst, int Ho, int Wo, int Do){
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int d = threadIdx.z; 

    int i,j,k,l,m,n;
    double sum=0;
    for(i=0;i<N;i++){ for(j=0;j<C;j++){ for(k=0;k<K;k++){
        sum=0;
        for(l=Hf-1; l>=0;l--){
        for(m=Wf-1; m>=0;m--){
        for(n=Df-1; n>=0;n--){
            sum += src[ind5d(C,H,W,D,i,j,h+Hf-1-l,w+Wf-1-m,d+Df-1-n)] * flt[ind5d(C,Hf,Wf,Df,k,j,l,m,n)];
        }}}
        dst[ind5d(K,Ho,Wo,Do,i,k,h,w,d)] = sum;
    } } }
}

__global__ void krnlCorr4d( double *src, int N, int C, int H, int W,
                            double *flt, int Hf, int Wf, int K,
                            double *dst, int Ho, int Wo){
    /* 
        src: (W,H,C,N)          (N,C,H,W)
        flt: (X,Y,C,K)          (K,C,Hf,Wf)
        dst: (W-X+1,H-Y+1,K,N)  (N,K,H-Hf+1,W-Wf+1)
    */
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j,k,l,m;
    double sum=0;
    for(i=0;i<N;i++){ for(j=0;j<C;j++){ for(k=0;k<K;k++){
        sum=0;
        for(l=0; l<Hf;l++){
        for(m=0; m<Wf;m++){
            sum += src[ind4d(C,H,W,i,j,h+l,w+m)] * flt[ind4d(C,Hf,Wf,k,j,l,m)];
           // sum += src[srcW*(h+i)+(w+j)] * flt[i*fltW+j];
        }}
        //dst[dstW*h+w] = sum;
        dst[ind4d(K,Ho,Wo,i,k,h,w)] = sum;
    } } }
}

__global__ void krnlConv4d( double *src, int N, int C, int H, int W,
                            double *flt, int Hf, int Wf, int K,
                            double *dst, int Ho, int Wo){
    /* 
        src: (W,H,C,N)          (N,C,H,W)
        flt: (X,Y,C,K)          (K,C,Hf,Wf)
        dst: (W-X+1,H-Y+1,K,N)  (N,K,H-Hf+1,W-Wf+1)
    */
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j,k,l,m;
    double sum=0;
    for(i=0;i<N;i++){ for(j=0;j<C;j++){ for(k=0;k<K;k++){
        sum=0;
        for(l=Hf-1; l>=0;l--){
        for(m=Wf-1; m>=0;m--){
            sum += src[ind4d(C,H,W,i,j,h+Hf-1-l,w+Wf-1-m)] * flt[ind4d(C,Hf,Wf,k,j,l,m)];
            //sum += src[srcW*(h+fltH-1-i)+(w+fltW-1-j)] * flt[i*fltW+j];
        }}
        dst[ind4d(K,Ho,Wo,i,k,h,w)] = sum;
        //dst[dstW*h+w] = sum;
    } } }
}

__global__ void krnlConv4dPad( double *src, int N, int C, int H, int W,
                            double *flt, int Hf, int Wf, int K,
                            double *dst, int Ho, int Wo, int pad){
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
    for(i=0;i<N;i++){ for(j=0;j<C;j++){ for(k=0;k<K;k++){
        sum=0;
        for(l=Hf-1; l>=0;l--){
        for(m=Wf-1; m>=0;m--){
            hsrc = h+Hf-1-l-pad; //int hsrc = h+Hf-1-l;
            wsrc = w+Wf-1-m-pad;
            if(hsrc >= 0 && wsrc >= 0 && hsrc < H && wsrc < W) 
                sum += src[ind4d(C,H,W,i,j,hsrc,wsrc)] * flt[ind4d(C,Hf,Wf,k,j,l,m)];
        }}
        dst[ind4d(K,Ho,Wo,i,k,h,w)] = sum;
        //dst[dstW*h+w] = sum;
    } } }
}

__global__ void krnlConv2d(   double *src, int srcH, int srcW,
                            double *flt, int fltH, int fltW,
                            double *dst, int dstH, int dstW){
    int h = threadIdx.x; 
    int w = threadIdx.y; 
    int i,j;
    double sum=0;
    for(i=fltH-1; i>=0;i--){
        for(j=fltW-1; j>=0;j--){
            sum += src[srcW*(h+fltH-1-i)+(w+fltW-1-j)] * flt[i*fltW+j];
        }
    }
    dst[dstW*h+w] = sum;
}
//int fgetc(FILE *stream);

//
/*
   cudnnStatus_t cudnnCreate(cudnnHandle_t *handle)
   cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *fltDesc)
   CUDNN_DATA_DOUBLE
   cudnnStatus_t
   cudnnSetConvolution2dDescriptor
   cudnnGetConvolutionForwardAlgorithm(
    cudnnGetConvolutionForwardWorkspaceSize
*/
double* readImages(){
    FILE *fp;
    fp=fopen("data0", "rb");
    //size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
    int N=1000*28*28;
    unsigned char images[N];
    double *E = (double*)malloc(sizeof(double)*N);
    fread(images, sizeof(unsigned char), N, fp);
    int i;
    for(i=0;i<N;i++)
        E[i] = images[i]/255.0;
    fclose(fp);
    return E;
}

int main3(){
    double* images = readImages();
    int i,j;
    for(i=0;i<28;i++){
        for(j=0;j<28;j++){
            if(images[ind2d(28,i,j)]>0)
                printf("%d\t%d\n",i,j);
            //printf("%3.1f ", images[sub2ind(28,i,j)]);
        }
        //printf("\n");
    }
    printf("%.4f\n", images[ind2d(28,20,18)]);
    printf("%.4f\n", images[ind3d(28,28,0,20,18)]);
    printf("%.4f\n", images[ind4d(1,28,28,0,0,20,18)]);
    return 0;
}

double *cudnnConv(  double *src, int N, int C, int H, int W,
                    double *flt, int Hf, int Wf, int K){
    cudnnHandle_t                   handle = NULL;
    cudnnTensorDescriptor_t         srcDesc = NULL;
    cudnnTensorDescriptor_t         dstDesc = NULL;
    cudnnFilterDescriptor_t         fltDesc = NULL;
    cudnnConvolutionDescriptor_t    convDesc = NULL;


    //creation
    cudnnErrchk( cudnnCreate(&handle) );
    cudnnErrchk( cudnnCreateTensorDescriptor(&srcDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(&dstDesc) );
    cudnnErrchk( cudnnCreateFilterDescriptor(&fltDesc) );
    cudnnErrchk( cudnnCreateConvolutionDescriptor(&convDesc) );

    //set
    cudnnErrchk( cudnnSetTensor4dDescriptor(srcDesc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
            N, C, H, W) );
    cudnnErrchk( cudnnSetFilter4dDescriptor(fltDesc,
            CUDNN_DATA_DOUBLE, K, C, Hf, Wf) );
    cudnnErrchk( cudnnSetConvolution2dDescriptor(convDesc,
            0,0,1,1,1,1, CUDNN_CROSS_CORRELATION) );
    //output tensor desc
    int No,Co,Ho,Wo;
    cudnnErrchk( cudnnGetConvolution2dForwardOutputDim(convDesc, srcDesc, fltDesc,
            &No, &Co, &Ho, &Wo) );
    //printf("%d %d %d %d\n", No, Co, Ho, Wo);
    assert(No == N); assert(Co == K);
    cudnnErrchk( cudnnSetTensor4dDescriptor(dstDesc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
            No, Co, Ho, Wo) );
    double *dst_d;
    gpuErrchk( cudaMalloc(&dst_d, sizeof(double)*No*Co*Ho*Wo) );
    double *dst_h = (double*)malloc(sizeof(double)*No*Co*Ho*Wo);

    //forward algo & workspace
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionFwdPreference_t convFwdPref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    void *workSpace = NULL;
    size_t workSpaceSize = 0, memLimit=0;

    cudnnGetConvolutionForwardAlgorithm(handle, srcDesc, fltDesc, convDesc,
            dstDesc, convFwdPref, memLimit, &convFwdAlgo);
    cudnnGetConvolutionForwardWorkspaceSize(handle, srcDesc, fltDesc, 
            convDesc, dstDesc, convFwdAlgo, &workSpaceSize);

    double alpha=1, beta=1; //scaling params for input and output
    cudnnConvolutionForward(handle, &alpha, srcDesc, src, fltDesc, flt,
            convDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, dstDesc,
            dst_d);

    cudaMemcpy(dst_h, dst_d, sizeof(double)*No*Co*Ho*Wo, cudaMemcpyDeviceToHost);

    //destroy
    if (srcDesc != NULL) cudnnDestroyTensorDescriptor(srcDesc);
    if (dstDesc != NULL) cudnnDestroyTensorDescriptor(dstDesc);
    if (fltDesc != NULL) cudnnDestroyFilterDescriptor(fltDesc);
    if (convDesc != NULL) cudnnDestroyConvolutionDescriptor(convDesc);
    if (handle != NULL) cudnnDestroy(handle);

    //free
    cudaFree(dst_d);

    return dst_h;
}
double *myConv(  double *src_d, int N, int C, int H, int W,
                    double *flt_d, int Hf, int Wf, int K)
{
    int Ho=H-Hf+1, Wo=W-Wf+1;
    double *dst2_d;
    gpuErrchk( cudaMalloc(&dst2_d, sizeof(double)*N*C*Ho*Wo) );
    double *dst2_h = (double*)malloc(sizeof(double)*N*C*Ho*Wo);

    dim3 threads(Ho, Wo, 1); 
    dim3 grid(1,1,1);
    krnlConv4d<<<grid,threads>>>(   src_d, N, C, H, W,
                                    flt_d, Hf, Wf, K,
                                    dst2_d, Ho, Wo);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(dst2_h, dst2_d, sizeof(double)*N*C*Ho*Wo, cudaMemcpyDeviceToHost);

    cudaFree(dst2_d);
    return dst2_h;
}
double *myConvPad(  double *src_d, int N, int C, int H, int W,
                    double *flt_d, int Hf, int Wf, int K,
                    int *pHo, int *pWo, int pad)
{
    int Ho=H+(2*pad)-Hf+1, Wo=W+(2*pad)-Wf+1; //int Ho=H-Hf+1, Wo=W-Wf+1;
    *pHo = Ho; *pWo = Wo;
    double *dst2_d;
    gpuErrchk( cudaMalloc(&dst2_d, sizeof(double)*N*C*Ho*Wo) );
    double *dst2_h = (double*)malloc(sizeof(double)*N*C*Ho*Wo);

    dim3 threads(Ho, Wo, 1); 
    dim3 grid(1,1,1);
    krnlConv4dPad<<<grid,threads>>>(   src_d, N, C, H, W,
                                    flt_d, Hf, Wf, K,
                                    dst2_d, Ho, Wo, pad);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(dst2_h, dst2_d, sizeof(double)*N*C*Ho*Wo, cudaMemcpyDeviceToHost);

    cudaFree(dst2_d);
    return dst2_h;
}

int main4(){
    double srcData[] =     {1.0, 6.0, 11.0,
                            2.0, 7.0, 12.0};
    double destData[6];
    int i,j;
    for(i=0;i<2;i++){
        for(j=0;j<3;j++){
            destData[ind2d(2,j,2-1-i)] = srcData[ind2d(3,i,j)];
            //destData[ind2d(2,j,i)] = srcData[ind2d(3,i,j)];
        }
    }
    print2Dd(srcData,2,3);
    print2Dd(destData,3,2);

    return 0;
}

int main(){
    /*double srcData[] =     {1.0, 6.0, 11.0, 16.0,
                            2.0, 7.0, 12.0, 17.0,
                            3.0, 8.0, 13.0, 18.0,
                            4.0, 9.0, 14.0, 19.0,
                            5.0, 10.0, 15.0, 20.0,
                            4.0, 9.0, 14.0, 19.0,
                            };*/
    double srcData[] = {    1.0, 4.0, 7.0,
                            2.0, 5.0, 8.0,
                            3.0, 6.0, 9.0};
    double fltData[] = {1.0, 3.0, 2.0, 4.0};

    int N=1, C=1, H=3, W=3;
    int K=1, Hf=2, Wf=2;
    double *flt_h = &fltData[0];
    double *src_h = &srcData[0];
    //double *src_h = readImages();
    double *src_d, *flt_d;

    gpuErrchk( cudaMalloc(&src_d, sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMemcpy(src_d, src_h, sizeof(double)*N*C*H*W, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&flt_d, sizeof(double)*K*C*Hf*Wf) );
    gpuErrchk( cudaMemcpy(flt_d, flt_h, sizeof(double)*K*C*Hf*Wf, cudaMemcpyHostToDevice) );

    double *dst_h = cudnnConv(src_d, N, C, H, W,
                    flt_d, Hf, Wf, K);

    int pad=1,Ho,Wo;
    double *dst2_h = myConvPad(src_d, N, C, H, W,
                    flt_d, Hf, Wf, K, &Ho, &Wo, pad);
    printf("%d %d\n", Ho, Wo);

    print2Dd(dst_h, Ho, Wo); printf("\n");
    print2Dd(dst2_h, Ho, Wo);

    /*int i;
    double err=0;
    for(i=0;i<(N*K*Ho*Wo);i++)
        err += abs(dst_h[i]-dst2_h[i]);
    printf("%.4f\n", err);*/

    cudaFree(src_d);
    cudaFree(flt_d);

    return 0;
}

