#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <time.h>

void  readImages(double *E, int N){
    FILE *fp;
    fp=fopen("data0", "rb");
    //size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
    unsigned char images[N];
    //double *E = (double*)malloc(sizeof(double)*N);
    fread(images, sizeof(unsigned char), N, fp);
    int i;
    for(i=0;i<N;i++)
        E[i] = images[i]/255.0;
    fclose(fp);
}

void fillRandom(double *E, int N){
    int i;
    for(i=0; i<N; i++)
        E[i] = rand() % 10 + 1;
}

double eqseq(double *A, double *B, int N){
    int i;
    double err=0;
    for(i=0;i<N;i++)
        err += abs(A[i]-B[i]);
    return err;
}

double xtoyData[] = 
{   1.0, 6.0, 11.0, 16.0,
    2.0, 7.0, 12.0, 17.0,
    3.0, 8.0, 13.0, 18.0,
    4.0, 9.0, 14.0, 19.0,
    5.0, 10.0, 15.0, 20.0,
    4.0, 9.0, 14.0, 19.0 };
double wtoyData[] = 
{   1.0, 3.0, 1.0, 3.0,
    2.0, 4.0, 2.0, 4.0};

int main(){
    int VERBOSE=0;
    int CONV=1;
    srand(time(NULL));
    const int N=100, C=3, H=28, W=28; // src
    const int K=10, Hw=7, Ww=7; // flt
    assert(H>=Hw); assert(W>=Ww);
    const int Hy=H-Hw+1, Wy=W-Ww+1; // dst 
    double xData[N*C*H*W]; fillRandom(xData,N*C*H*W);
    double wData[K*C*Hw*Hw]; fillRandom(wData, K*C*Hw*Ww);
    double dyData[N*K*Hy*Wy]; fillRandom(dyData, N*K*Hy*Wy);

    printf("N:%d C:%d H:%d W:%d\n",N,C,H,W);
    printf("K:%d C:%d Hw:%d Ww:%d\n",K,C,Hw,Ww);
    printf("N:%d K:%d Hy:%d Wy:%d\n",N,K,Hy,Wy);
    printf("\n");

    if(VERBOSE){
        printf("x:\n");
        print2Dd(xData, H, W);
        printf("w:\n");
        print2Dd(wData, Hw, Ww);
        printf("dy:\n");
        print2Dd(dyData, Hy, Wy);
        printf("\n");
    }

    

    double *x_h = &xData[0], *w_h = &wData[0], *dy_h=&dyData[0]; // given
    double dx_h[N*C*H*W], dw_h[C*K*Hw*Ww], y_h[N*K*Hy*Wy], db_h[1*K*1*1]; // compute cudnn
    double dx1_h[N*C*H*W], dw1_h[K*C*Hw*Ww], y1_h[N*K*Hy*Wy], db1_h[1*K*1*1]; // compute kunet
    double *x_d, *dx_d, *w_d, *dw_d, *y_d, *dy_d, *db_d; // gpu pointers

    gpuErrchk( cudaMalloc(&x_d, sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&dx_d, sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&w_d, sizeof(double)*K*C*Hw*Ww) );
    gpuErrchk( cudaMalloc(&dw_d, sizeof(double)*K*C*Hw*Ww) );
    gpuErrchk( cudaMalloc(&y_d, sizeof(double)*N*K*Hy*Wy) );
    gpuErrchk( cudaMalloc(&dy_d, sizeof(double)*N*K*Hy*Wy) );
    gpuErrchk( cudaMalloc(&db_d, sizeof(double)*1*K*1*1) );

    // send x, w, dy to GPU
    gpuErrchk( cudaMemcpy(x_d, x_h, sizeof(double)*N*C*H*W, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(w_d, w_h, sizeof(double)*K*C*Hw*Ww, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dy_d, dy_h, sizeof(double)*N*K*Hy*Wy, cudaMemcpyHostToDevice) );
    // end send x, w, dy to GPU
    
    /**
      CUDNN KUNET COMPARISON TESTS
    **/
    cudnnHandle_t                   handle = NULL;
    cudnnTensorDescriptor_t         xDesc = NULL;
    cudnnTensorDescriptor_t         dxDesc = NULL;
    cudnnTensorDescriptor_t         yDesc = NULL;
    cudnnTensorDescriptor_t         dyDesc = NULL;
    cudnnTensorDescriptor_t         dbDesc = NULL;
    cudnnFilterDescriptor_t         wDesc = NULL;
    cudnnFilterDescriptor_t         dwDesc = NULL;
    cudnnConvolutionDescriptor_t    xcorr00Desc = NULL;
    cudnnConvolutionDescriptor_t    conv00Desc = NULL;


    // creation
    cudnnErrchk( cudnnCreate(                       &handle) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &xDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dxDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &yDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dyDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dbDesc) );
    cudnnErrchk( cudnnCreateFilterDescriptor(       &wDesc) );
    cudnnErrchk( cudnnCreateFilterDescriptor(       &dwDesc) );
    cudnnErrchk( cudnnCreateConvolutionDescriptor(  &xcorr00Desc) );
    cudnnErrchk( cudnnCreateConvolutionDescriptor(  &conv00Desc) );
    // end creation

    // set
    cudnnErrchk( cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W) );
    cudnnErrchk( cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_DOUBLE, K, C, Hw, Ww) );
    cudnnErrchk( cudnnSetFilter4dDescriptor(dwDesc, CUDNN_DATA_DOUBLE, K, C, Hw, Ww) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, K, Hy, Wy) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, K, Hy, Wy) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, 1, 1) );
    cudnnErrchk( cudnnSetConvolution2dDescriptor(xcorr00Desc, 0,0,1,1,1,1, CUDNN_CROSS_CORRELATION) );
    cudnnErrchk( cudnnSetConvolution2dDescriptor(conv00Desc, 0,0,1,1,1,1, CUDNN_CONVOLUTION) );
    // end set input and conf

    // set conv mode
    cudnnConvolutionDescriptor_t    tconvDesc = NULL;
    if(CONV){
        tconvDesc = conv00Desc;
        printf("mode: conv00\n");
    }else{
        tconvDesc = xcorr00Desc;
        printf("mode: xcorr00\n");
    }
    // end set conv mode

    // forward algo conf & workspace
    double alpha=1, beta=1; //scaling params for input and output
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionFwdPreference_t convFwdPref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    void *workSpace = NULL; size_t workSpaceSize = 0, memLimit=0;
    cudnnErrchk( cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, tconvDesc, yDesc, convFwdPref, memLimit, &convFwdAlgo) );
    cudnnErrchk( cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, tconvDesc, yDesc, convFwdAlgo, &workSpaceSize) );
    printf("workspace size: %d\n", workSpaceSize);
    // end forward algo conf & workspace

    // forward test
    printf("\ny:\n");
    cudnnErrchk( cudnnConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    gpuErrchk( cudaMemcpy(y_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    gpuErrchk( cudaMemcpy(y1_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );
    if(VERBOSE){ print2Dd(y_h, Hy, Wy); printf("\n"); print2Dd(y1_h, Hy, Wy);}
    assert(eqseq(y_h,y1_h,N*K*Hy*Wy) < 1.0E-4);
    printf("y: ok.\n");
    // end forward test

    // backward filter test
    printf("\ndw:\n");
    cudnnErrchk( cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    gpuErrchk( cudaMemcpy(dw_h, dw_d, sizeof(double)*K*C*Hw*Ww, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    gpuErrchk( cudaMemcpy(dw1_h, dw_d, sizeof(double)*K*C*Hw*Ww, cudaMemcpyDeviceToHost) );
    if(VERBOSE){ print2Dd(dw_h, Hw, Ww); printf("\n"); print2Dd(dw1_h, Hw, Ww);}
    assert(eqseq(dw_h,dw1_h,K*C*Hw*Ww) < 1.0E-4);
    printf("dw: ok.\n");
    //print2Dd(dw_h, Hw, Ww); printf("\n");
    // end backward filter test

    // backward data test
    printf("\ndx:\n");
    cudnnErrchk( cudnnConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaMemcpy(dx_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaMemcpy(dx1_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );
    if(VERBOSE){print2Dd(dx_h, H, W); printf("\n");print2Dd(dx1_h, H, W);}
    assert(eqseq(dx_h,dx1_h,N*C*H*W) < 1.0E-4);
    printf("dx: ok.\n");
    // end backward data test

    // backward bias test
    printf("\ndb:\n");
    cudnnErrchk( cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    gpuErrchk( cudaMemcpy(db_h, db_d, sizeof(double)*1*K*1*1, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    gpuErrchk( cudaMemcpy(db1_h, db_d, sizeof(double)*1*K*1*1, cudaMemcpyDeviceToHost) );
    if(VERBOSE){print2Dd(db_h, 1, K); printf("\n");print2Dd(db1_h, 1, K);}
    assert(eqseq(db_h,db1_h,1*K*1*1) < 1.0E-4);
    printf("db: ok.\n\n");
    // end backward bias test

    printf("ok.\n");

    // destroy
    if (xDesc != NULL) cudnnDestroyTensorDescriptor(xDesc);
    if (dxDesc != NULL) cudnnDestroyTensorDescriptor(dxDesc);
    if (wDesc != NULL) cudnnDestroyFilterDescriptor(wDesc);
    if (dwDesc != NULL) cudnnDestroyFilterDescriptor(dwDesc);
    if (yDesc != NULL) cudnnDestroyTensorDescriptor(yDesc);
    if (dyDesc != NULL) cudnnDestroyTensorDescriptor(dyDesc);
    if (dbDesc != NULL) cudnnDestroyTensorDescriptor(dbDesc);
    if (xcorr00Desc != NULL) cudnnDestroyConvolutionDescriptor(xcorr00Desc);
    if (conv00Desc != NULL) cudnnDestroyConvolutionDescriptor(conv00Desc);
    if (handle != NULL) cudnnDestroy(handle);

    // free
    cudaFree(x_d); cudaFree(dx_d); cudaFree(w_d); cudaFree(dw_d); cudaFree(y_d); cudaFree(dy_d); cudaFree(db_d);
    // END TESTS
    return 0;
}

