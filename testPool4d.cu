#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <time.h>
#include "test.h"
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


int main(){
    int VERBOSE=0;
    //int PMODE=1; // max
    srand(time(NULL));
    const int N=28, C=3, H=40, W=80; // src
    const int K=C, Hd=8, Wd=8; // window
    const int Hs=Hd, Ws=Hd; // stride
    const int Hp=0, Wp=0; // padding
    assert(H>=Hd); assert(W>=Wd);
    const int Hy=1+ceil((H+2*Hp-Hd)/(double)Hs), Wy=1+ceil((W+2*Wp-Wd)/(double)Ws); // dst 

    printf("N:%d C:%d H:%d W:%d\n",N,C,H,W);
    printf("Hd:%d Wd:%d Hs:%d Ws:%d Hp:%d Wp:%d\n",Hd,Wd,Hs,Ws,Hp,Wp);
    printf("N:%d K:%d Hy:%d Wy:%d\n",N,C,Hy,Wy);
    printf("\n");

    double xData[N*C*H*W]; fillRandom(xData,N*C*H*W);
    double dyData[N*K*Hy*Wy]; fillRandom(dyData,N*K*Hy*Wy);

    if(VERBOSE){
        printf("x:\n");
        print2Dd(xData, H, W);
        printf("dy:\n");
        print2Dd(dyData, Hy, Wy);
        printf("\n");
    }

    double t0, time_elapsed;

    double *x_h = &xData[0], *dy_h = &dyData[0]; // given
    double y_h[N*C*Hy*Wy], dx_h[N*C*H*W]; // compute cudnn
    double y1_h[N*C*H*W], dx1_h[N*C*H*W]; // compute kunet
    double *x_d, *y_d, *dx_d, *dy_d; // gpu pointers

    gpuErrchk( cudaMalloc(&x_d, sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&dx_d, sizeof(double)*N*C*H*W) );
    gpuErrchk( cudaMalloc(&y_d, sizeof(double)*N*K*Hy*Wy) );
    gpuErrchk( cudaMalloc(&dy_d, sizeof(double)*N*K*Hy*Wy) );

    // send x, dy to GPU
    gpuErrchk( cudaMemcpy(x_d, x_h, sizeof(double)*N*C*H*W, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dy_d, dy_h, sizeof(double)*N*K*Hy*Wy, cudaMemcpyHostToDevice) );
    // end send x, dy to GPU
    
    /**
      CUDNN KUNET COMPARISON TESTS
    **/
    cudnnHandle_t                   handle = NULL;
    cudnnTensorDescriptor_t         xDesc = NULL;
    cudnnTensorDescriptor_t         dxDesc = NULL;
    cudnnTensorDescriptor_t         yDesc = NULL;
    cudnnTensorDescriptor_t         dyDesc = NULL;
    cudnnPoolingDescriptor_t        maxPool00Desc = NULL;


    // creation
    cudnnErrchk( cudnnCreate(                       &handle) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &xDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dxDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &yDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dyDesc) );
    cudnnErrchk( cudnnCreatePoolingDescriptor(      &maxPool00Desc) );
    // end creation

    // set
    cudnnErrchk( cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, K, Hy, Wy) );
    cudnnErrchk( cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, K, Hy, Wy) );
    cudnnErrchk( cudnnSetPooling2dDescriptor(maxPool00Desc, CUDNN_POOLING_MAX, Hd,Wd,0,0,Hs,Ws) );
    // end set input and conf

    // set pool mode
    cudnnPoolingDescriptor_t    tpoolDesc = NULL;
    tpoolDesc = maxPool00Desc;
    printf("mode: maxPool00\n");
    // end set pool mode

    double alpha=1, beta=1;

    // forward test
    printf("y:\n");
    t0 = getTime();
    cudnnErrchk( cudnnPoolingForward(handle, tpoolDesc, &alpha, xDesc, x_d, &beta, yDesc, y_d) );
    gpuErrchk( cudaPeekAtLastError() ); gpuErrchk( cudaDeviceSynchronize() );
    time_elapsed = getTime() - t0; printf("cudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(y_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );

    t0 = getTime();
    cudnnErrchk( kunetPoolingForward(handle, tpoolDesc, &alpha, xDesc, x_d, &beta, yDesc, y_d) );
    time_elapsed = getTime() - t0; printf("kudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(y1_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );
    if(VERBOSE){print2Dd(y_h, Hy, Wy); printf("\n"); print2Dd(y1_h, Hy, Wy);}
    assert(eqseq(y_h,y1_h,N*K*Hy*Wy) < 1.0E-4);
    printf("y: ok.\n\n");
    // end forward test

    // backward test
    printf("dx:\n");
    t0 = getTime();
    cudnnErrchk( cudnnPoolingBackward(handle, tpoolDesc, &alpha, yDesc, y_d, dyDesc, dy_d, xDesc, x_d, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaPeekAtLastError() ); gpuErrchk( cudaDeviceSynchronize() );
    time_elapsed = getTime() - t0; printf("cudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(dx_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );

    t0 = getTime();
    cudnnErrchk( kunetPoolingBackward(handle, tpoolDesc, &alpha, yDesc, y_d, dyDesc, dy_d, xDesc, x_d, &beta, dxDesc, dx_d) );
    time_elapsed = getTime() - t0; printf("kudnn: %.4f\n",time_elapsed);
    gpuErrchk( cudaMemcpy(dx1_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );
    if(VERBOSE){print2Dd(dx_h, H, W); printf("\n");print2Dd(dx1_h, H, W);}
    assert(eqseq(dx_h,dx1_h,N*C*H*W) < 1.0E-4);
    printf("dx:ok\n");
    // end backward test
    
    printf("ok.\n");

    // destroy
    if (xDesc != NULL) cudnnDestroyTensorDescriptor(xDesc);
    if (dxDesc != NULL) cudnnDestroyTensorDescriptor(dxDesc);
    if (dyDesc != NULL) cudnnDestroyTensorDescriptor(dyDesc);
    if (maxPool00Desc != NULL) cudnnDestroyPoolingDescriptor(maxPool00Desc);
    if (handle != NULL) cudnnDestroy(handle);

    // free
    cudaFree(x_d); cudaFree(y_d);
    cudaFree(dx_d); cudaFree(dy_d);
    // END TESTS
    return 0;
}

