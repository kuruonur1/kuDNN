#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <time.h>
#include "test.h"
#include <string.h>
#include <sys/time.h>

static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
        //cerr << "ERROR: Bad call to gettimeofday" << endl;
        printf("ERROR: Bad call to gettimeofday\n");
        return(-1);
    }

    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

int main(int argc, char *argv[]){
    int CONV, VERBOSE;
    if(argc==3){
        if(!strcmp(argv[1], "conv"))
            CONV=1;
        else if(!strcmp(argv[1],"xcorr"))
            CONV=0;
        else{
            printf("usage: ./testConv4d xcorr/conv v[01]\n"); exit(-1);
        }

        if(!strcmp(argv[2], "v0"))
            VERBOSE=0;
        else if(!strcmp(argv[2],"v1"))
            VERBOSE=1;
        else{
            printf("usage: ./testConv4d xcorr/conv v[01]\n"); exit(-1);
        }
    }else{
        printf("usage: ./testConv4d xcorr/conv v[01]\n"); exit(-1);
    }

    int N, C, H, W; // src
    int K, Hw, Ww; // flt
    if(VERBOSE){
        N=1; C=1; H=5; W=4; // src
        K=1; Hw=2; Ww=2; // flt
    }else{
        /*
        N=28; C=3; H=40; W=80; // src
        K=5; Hw=8; Ww=7; // flt
        */
        N=128; C=3; H=40; W=80; // src
        K=5; Hw=18; Ww=17; // flt
    }

    int convHpad=0, convWpad=0, convHSt=1, convWSt=1;

    assert(H>=Hw); assert(W>=Ww);
    srand(time(NULL));
    const int Hy=1+(H+2*convHpad-Hw)/convHSt, Wy=1+(W+2*convWpad-Ww)/convWSt; // dst 
    double xData[N*C*H*W]; fillRandom(xData,N*C*H*W);
    double wData[K*C*Hw*Hw]; fillRandom(wData, K*C*Hw*Ww);
    double dyData[N*K*Hy*Wy]; fillRandom(dyData, N*K*Hy*Wy);

    double t0, time_elapsed;

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
    double *x_d=NULL, *dx_d, *w_d, *dw_d, *y_d, *dy_d, *db_d; // gpu pointers

    gpuErrchk( cudaMalloc(&x_d,     sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&dx_d,    sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&w_d,     sizeof(double)*K*C*Hw*Ww) );
    gpuErrchk( cudaMalloc(&dw_d,    sizeof(double)*K*C*Hw*Ww) );
    gpuErrchk( cudaMalloc(&y_d,     sizeof(double)*N*K*Hy*Wy) );
    gpuErrchk( cudaMalloc(&dy_d,    sizeof(double)*N*K*Hy*Wy) );
    gpuErrchk( cudaMalloc(&db_d,    sizeof(double)*1*K*1*1) );


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
    cudnnConvolutionDescriptor_t    xcorrppDesc = NULL;


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
    cudnnErrchk( cudnnCreateConvolutionDescriptor(  &xcorrppDesc) );
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
    cudnnErrchk( cudnnSetConvolution2dDescriptor(xcorrppDesc, convHpad,convWpad,convHSt,convWSt,1,1, CUDNN_CROSS_CORRELATION) );
    cudnnErrchk( cudnnSetConvolution2dDescriptor(conv00Desc, 0,0,1,1,1,1, CUDNN_CONVOLUTION) );
    // end set input and conf

    // set conv mode
    cudnnConvolutionDescriptor_t    tconvDesc = NULL;
    if(CONV){
        tconvDesc = conv00Desc;
        printf("mode: conv00\n");
    }else{
        // tconvDesc = xcorr00Desc;
        tconvDesc = xcorrppDesc;
        printf("mode: xcorrpp\n");
    }
    // end set conv mode

    // forward algo conf & workspace
    double alpha=1, beta=1; //scaling params for input and output
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionFwdPreference_t convFwdPref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    void *workSpace = NULL; size_t workSpaceSize = 0, memLimit=0;
    cudnnErrchk( cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, tconvDesc, yDesc, convFwdPref, memLimit, &convFwdAlgo) );
    cudnnErrchk( cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, tconvDesc, yDesc, convFwdAlgo, &workSpaceSize) );
    //printf("workspace size: %d\n", workSpaceSize);
    // end forward algo conf & workspace

    // forward test
    printf("\ny:\n");
    t0 = getTime();
    cudnnErrchk( cudnnConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    gpuErrchk( cudaPeekAtLastError() ); gpuErrchk( cudaDeviceSynchronize() );
    time_elapsed = getTime() - t0; printf("cudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(y_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );

    t0 = getTime();
    cudnnErrchk( kunetConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    time_elapsed = getTime() - t0; printf("kudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(y1_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );
    if(VERBOSE){ print2Dd(y_h, Hy, Wy); printf("\n"); print2Dd(y1_h, Hy, Wy);}
    assert(eqseq(y_h,y1_h,N*K*Hy*Wy) < 1.0E-4);
    printf("y: ok.\n");
    // end forward test

    // backward filter test
    printf("\ndw:\n");
    t0 = getTime();
    cudnnErrchk( cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    gpuErrchk( cudaPeekAtLastError() ); gpuErrchk( cudaDeviceSynchronize() );
    time_elapsed = getTime() - t0; printf("cudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(dw_h, dw_d, sizeof(double)*K*C*Hw*Ww, cudaMemcpyDeviceToHost) );

    t0 = getTime();
    cudnnErrchk( kunetConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    time_elapsed = getTime() - t0; printf("kudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(dw1_h, dw_d, sizeof(double)*K*C*Hw*Ww, cudaMemcpyDeviceToHost) );
    if(VERBOSE){ print2Dd(dw_h, Hw, Ww); printf("\n"); print2Dd(dw1_h, Hw, Ww);}
    assert(eqseq(dw_h,dw1_h,K*C*Hw*Ww) < 1.0E-4);
    printf("dw: ok.\n");
    //print2Dd(dw_h, Hw, Ww); printf("\n");
    // end backward filter test

    // backward data test
    printf("\ndx:\n");
    t0 = getTime();
    cudnnErrchk( cudnnConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaPeekAtLastError() ); gpuErrchk( cudaDeviceSynchronize() );
    time_elapsed = getTime() - t0; printf("cudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(dx_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );

    t0 = getTime();
    cudnnErrchk( kunetConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    time_elapsed = getTime() - t0; printf("kudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(dx1_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );
    if(VERBOSE){print2Dd(dx_h, H, W); printf("\n");print2Dd(dx1_h, H, W);}
    assert(eqseq(dx_h,dx1_h,N*C*H*W) < 1.0E-4);
    printf("dx: ok.\n");
    // end backward data test

    // backward bias test
    printf("\ndb:\n");
    t0 = getTime();
    cudnnErrchk( cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    gpuErrchk( cudaPeekAtLastError() ); gpuErrchk( cudaDeviceSynchronize() );
    time_elapsed = getTime() - t0; printf("cudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(db_h, db_d, sizeof(double)*1*K*1*1, cudaMemcpyDeviceToHost) );

    t0 = getTime();
    cudnnErrchk( kunetConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    time_elapsed = getTime() - t0; printf("kudnn: %.4f\n",time_elapsed);
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
    if (xcorrppDesc != NULL) cudnnDestroyConvolutionDescriptor(xcorrppDesc);
    if (conv00Desc != NULL) cudnnDestroyConvolutionDescriptor(conv00Desc);
    if (handle != NULL) cudnnDestroy(handle);

    // free
    cudaFree(x_d); cudaFree(dx_d); cudaFree(w_d); cudaFree(dw_d); cudaFree(y_d); cudaFree(dy_d); cudaFree(db_d);
    // END TESTS
    return 0;
}

