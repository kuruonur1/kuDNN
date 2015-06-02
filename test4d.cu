#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <time.h>

//#define MODE_CONV
//#define SIMPLE 1

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

void fillRandomData(){
}

int main(){
    srand(time(NULL));

#ifdef SIMPLE
    int TOY=1;
    printf("toy\n");
    double xData[] = 
    {   1.0, 6.0, 11.0, 16.0,
        2.0, 7.0, 12.0, 17.0,
        3.0, 8.0, 13.0, 18.0,
        4.0, 9.0, 14.0, 19.0,
        5.0, 10.0, 15.0, 20.0,
        4.0, 9.0, 14.0, 19.0 };
    int N=1, C=1, H=6, W=4; // src
    double wData[] = 
    {   1.0, 3.0, 1.0, 3.0,
        2.0, 4.0, 2.0, 4.0};
    const int K=1, Hw=2, Ww=2; // flt
    assert(H>=Hw); assert(W>=Ww);
    int Hy=H-Hw+1, Wy=W-Ww+1; // dst 
    int Hp=2, Wp=2; // pooling
    double dyData[N*K*Hy*Wy]; fillRandom(dyData, N*K*Hy*Wy);
    printf("x:\n");
    print2Dd(xData, H, W);
    printf("w:\n");
    print2Dd(wData, Hw, Ww);
    printf("dy:\n");
    print2Dd(dyData, Hy, Wy);
    printf("\n");
#else
    int TOY=0;
    printf("random data:\n");
    int N=3, C=3, H=28, W=28; // src
    int K=2, Hw=8, Ww=8; // flt
    int Hy=H-Hw+1, Wy=W-Ww+1; // dst
    int Hp=7, Wp=7; // pooling
    double xData[N*C*H*W]; fillRandom(xData,N*C*H*W);
    double dyData[N*K*Hy*Wy]; fillRandom(dyData, N*K*Hy*Wy);
    double wData[K*C*Hw*Hw]; fillRandom(wData, K*C*Hw*Ww);
#endif
    int hStride=Hp, wStride=Wp;
    int Hyp=1+ceil((Hy+2*0-Hp)/(double)hStride), Wyp=1+ceil((Wy+2*0-Wp)/(double)wStride); // y=1+ceil((x+2p-d)/s)
    printf("\n");
    printf("N:%d C:%d H:%d W:%d\n",N,C,H,W);
    printf("K:%d C:%d Hw:%d Ww:%d\n",K,C,Hw,Ww);
    printf("N:%d K:%d Hy:%d Wy:%d\n",N,K,Hy,Wy);
    printf("Hp:%d Wp:%d\n", Hp, Wp);
    printf("N:%d K:%d Hyp:%d Wyp:%d\n",N,K,Hyp,Wyp);
    printf("\n");
    

    double *x_h = &xData[0], *w_h = &wData[0], *dy_h=&dyData[0]; // given
    double dx_h[N*C*H*W], dw_h[C*K*Hw*Ww], y_h[N*K*Hy*Wy], db_h[1*K*1*1], yp_h[N*K*Hyp*Wyp]; // compute cudnn
    double dx1_h[N*C*H*W], dw1_h[K*C*Hw*Ww], y1_h[N*K*Hy*Wy], db1_h[1*K*1*1], yp1_h[N*K*Hyp*Wyp]; // compute kunet
    double *x_d, *dx_d, *w_d, *dw_d, *y_d, *dy_d, *db_d, *yp_d; // gpu pointers

    gpuErrchk( cudaMalloc(&x_d, sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&dx_d, sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&w_d, sizeof(double)*K*C*Hw*Ww) );
    gpuErrchk( cudaMalloc(&dw_d, sizeof(double)*K*C*Hw*Ww) );
    gpuErrchk( cudaMalloc(&y_d, sizeof(double)*N*K*Hy*Wy) );
    gpuErrchk( cudaMalloc(&dy_d, sizeof(double)*N*K*Hy*Wy) );
    gpuErrchk( cudaMalloc(&db_d, sizeof(double)*1*K*1*1) );
    gpuErrchk( cudaMalloc(&yp_d, sizeof(double)*N*K*Hyp*Wyp) );

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
    cudnnTensorDescriptor_t         ypDesc = NULL;
    cudnnTensorDescriptor_t         dyDesc = NULL;
    cudnnTensorDescriptor_t         dbDesc = NULL;
    cudnnFilterDescriptor_t         wDesc = NULL;
    cudnnFilterDescriptor_t         dwDesc = NULL;
    cudnnConvolutionDescriptor_t    xcorr00Desc = NULL;
    cudnnConvolutionDescriptor_t    conv00Desc = NULL;
    cudnnPoolingDescriptor_t        maxPoolDesc = NULL;


    // creation
    cudnnErrchk( cudnnCreate(                       &handle) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &xDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dxDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &yDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &ypDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dyDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dbDesc) );
    cudnnErrchk( cudnnCreateFilterDescriptor(       &wDesc) );
    cudnnErrchk( cudnnCreateFilterDescriptor(       &dwDesc) );
    cudnnErrchk( cudnnCreateConvolutionDescriptor(  &xcorr00Desc) );
    cudnnErrchk( cudnnCreateConvolutionDescriptor(  &conv00Desc) );
    cudnnErrchk( cudnnCreatePoolingDescriptor(      &maxPoolDesc) );
    // end creation

    // set
    cudnnErrchk( cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W) );
    cudnnErrchk( cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_DOUBLE, K, C, Hw, Ww) );
    cudnnErrchk( cudnnSetFilter4dDescriptor(dwDesc, CUDNN_DATA_DOUBLE, K, C, Hw, Ww) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, K, Hy, Wy) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(ypDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, K, Hyp, Wyp) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, K, Hy, Wy) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, 1, 1) );
    cudnnErrchk( cudnnSetConvolution2dDescriptor(xcorr00Desc, 0,0,1,1,1,1, CUDNN_CROSS_CORRELATION) );
    cudnnErrchk( cudnnSetConvolution2dDescriptor(conv00Desc, 0,0,1,1,1,1, CUDNN_CONVOLUTION) );
    cudnnErrchk( cudnnSetPooling2dDescriptor(maxPoolDesc, CUDNN_POOLING_MAX, Hp,Wp,0,0,hStride,wStride) );
    // end set input and conf

    // set conv mode
    cudnnConvolutionDescriptor_t    tconvDesc = NULL;
#ifdef MODE_CONV
    tconvDesc = conv00Desc;
    printf("mode: conv00\n");
#else
    tconvDesc = xcorr00Desc;
    printf("mode: xcorr00\n");
#endif
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
    printf("testing y:\n");
    cudnnErrchk( cudnnConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    gpuErrchk( cudaMemcpy(y_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );
    kunetConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d);
    gpuErrchk( cudaMemcpy(y1_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );
    if(TOY){ print2Dd(y_h, Hy, Wy); printf("\n"); print2Dd(y1_h, Hy, Wy); printf("\n");}
    assert(eqseq(y_h,y1_h,N*K*Hy*Wy) < 1.0E-4);
    printf("y: ok.\n\n");
    // end forward test

    // backward filter test
    printf("dw:\n");
    cudnnErrchk( cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    gpuErrchk( cudaMemcpy(dw_h, dw_d, sizeof(double)*K*C*Hw*Ww, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    gpuErrchk( cudaMemcpy(dw1_h, dw_d, sizeof(double)*K*C*Hw*Ww, cudaMemcpyDeviceToHost) );
    if(TOY){ print2Dd(dw_h, Hw, Ww); printf("\n"); print2Dd(dw1_h, Hw, Ww); printf("\n");}
    assert(eqseq(dw_h,dw1_h,K*C*Hw*Ww) < 1.0E-4);
    printf("dw: ok.\n\n");
    //print2Dd(dw_h, Hw, Ww); printf("\n");
    // end backward filter test

    // backward data test
    printf("dx:\n");
    cudnnErrchk( cudnnConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaMemcpy(dx_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaMemcpy(dx1_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );
    if(TOY){print2Dd(dx1_h, H, W); printf("\n");}
    assert(eqseq(dx_h,dx1_h,N*C*H*W) < 1.0E-4);
    printf("dx: ok.\n");
    // end backward data test

    // backward bias test
    printf("db:\n");
    cudnnErrchk( cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    gpuErrchk( cudaMemcpy(db_h, db_d, sizeof(double)*1*K*1*1, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    gpuErrchk( cudaMemcpy(db1_h, db_d, sizeof(double)*1*K*1*1, cudaMemcpyDeviceToHost) );
    if(TOY){print2Dd(db1_h, 1, K); printf("\n");}
    assert(eqseq(db_h,db1_h,1*K*1*1) < 1.0E-4);
    printf("db: ok.\n\n");
    // end backward bias test

    // pooling test
    printf("yp:\n");
    cudnnErrchk( cudnnPoolingForward(handle, maxPoolDesc, &alpha, yDesc, y_d, &beta, ypDesc, yp_d) );
    gpuErrchk( cudaMemcpy(yp_h, yp_d, sizeof(double)*N*K*Hyp*Wyp, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetPoolingForward(handle, maxPoolDesc, &alpha, yDesc, y_d, &beta, ypDesc, yp_d) );
    gpuErrchk( cudaMemcpy(yp1_h, yp_d, sizeof(double)*N*K*Hyp*Wyp, cudaMemcpyDeviceToHost) );
    if(TOY){print2Dd(yp_h, Hyp, Wyp); printf("\n"); print2Dd(yp1_h, Hyp, Wyp); printf("\n");}
    print2Dd(y_h, Hy, Wy); printf("\n");
    print2Dd(yp_h, Hyp, Wyp); printf("\n"); print2Dd(yp1_h, Hyp, Wyp); printf("\n");
    printf("%.f\n",eqseq(yp_h,yp1_h,N*K*Hyp*Wyp));
    assert(eqseq(yp_h,yp1_h,N*K*Hyp*Wyp) < 1.0E-4);
    printf("yp: ok.\n\n");
    // end pooling test
    
    printf("ok.\n");

    // destroy
    if (xDesc != NULL) cudnnDestroyTensorDescriptor(xDesc);
    if (dxDesc != NULL) cudnnDestroyTensorDescriptor(dxDesc);
    if (wDesc != NULL) cudnnDestroyFilterDescriptor(wDesc);
    if (dwDesc != NULL) cudnnDestroyFilterDescriptor(dwDesc);
    if (yDesc != NULL) cudnnDestroyTensorDescriptor(yDesc);
    if (ypDesc != NULL) cudnnDestroyTensorDescriptor(ypDesc);
    if (dyDesc != NULL) cudnnDestroyTensorDescriptor(dyDesc);
    if (dbDesc != NULL) cudnnDestroyTensorDescriptor(dbDesc);
    if (xcorr00Desc != NULL) cudnnDestroyConvolutionDescriptor(xcorr00Desc);
    if (conv00Desc != NULL) cudnnDestroyConvolutionDescriptor(conv00Desc);
    if (handle != NULL) cudnnDestroy(handle);

    // free
    cudaFree(x_d); cudaFree(dx_d); cudaFree(w_d); cudaFree(dw_d); cudaFree(y_d); cudaFree(dy_d); cudaFree(db_d); cudaFree(yp_d);
    // END TESTS
    return 0;
}

