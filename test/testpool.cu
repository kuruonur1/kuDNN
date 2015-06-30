#include <stdlib.h>
#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include "../src/kudnn.h"
#include <time.h>
#include <string.h>
#include "testutil.h"
#include "../src/kuassert.h"
#include "../src/kudnn.h"



void testPooling(
        int tdims, int xDims[],
        int pdims, int poolDims[], int poolPad[], int poolStride[],
        int verbose, int compare
    ){
    int i;
    int yDims[tdims], xStrides[tdims], yStrides[tdims];

    cudnnHandle_t                   handle = NULL;
    cudnnTensorDescriptor_t         xDesc = NULL;
    cudnnTensorDescriptor_t         dxDesc = NULL;
    cudnnTensorDescriptor_t         yDesc = NULL;
    cudnnTensorDescriptor_t         dyDesc = NULL;
    cudnnPoolingDescriptor_t        poolDesc = NULL;

    // create
    cudnnErrchk( cudnnCreate(                       &handle) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &xDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dxDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &yDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dyDesc) );
    cudnnErrchk( cudnnCreatePoolingDescriptor(      &poolDesc) );
    // end create

    // set
    cudnnErrchk( cudnnSetPoolingNdDescriptor(poolDesc, CUDNN_POOLING_MAX, pdims, poolDims, poolPad, poolStride) );

    dims2strides(xDims, tdims, xStrides);
    cudnnErrchk( cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_DOUBLE, tdims, xDims, xStrides) );
    cudnnErrchk( cudnnSetTensorNdDescriptor(dxDesc, CUDNN_DATA_DOUBLE, tdims, xDims, xStrides) );

    getPoolingNdForwardOutputDim(xDims, pdims, poolDims, poolPad, poolStride, yDims);
    dims2strides(yDims, tdims, yStrides);
    cudnnErrchk( cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_DOUBLE, tdims, yDims, yStrides) );
    cudnnErrchk( cudnnSetTensorNdDescriptor(dyDesc, CUDNN_DATA_DOUBLE, tdims, yDims, yStrides) );
    // end set

    printf("x:\t"); for(i=0;i<tdims;i++) printf("%d\t", xDims[i]); printf("\n");
    printf("w:\t"); printf("\t\t"); for(i=0;i<pdims;i++) printf("%d\t", poolDims[i]); printf("\tpool window\n");
    printf("p:\t"); printf("\t\t"); for(i=0;i<pdims;i++) printf("%d\t", poolPad[i]); printf("\tpadding\n");
    printf("s:\t"); printf("\t\t"); for(i=0;i<pdims;i++) printf("%d\t", poolStride[i]); printf("\tstride\n");
    printf("y:\t"); for(i=0;i<tdims;i++) printf("%d\t", yDims[i]); printf("\n");

    // random data
    double xData[prod(xDims,tdims)];    fillRandom(xData,prod(xDims,tdims));
    double dyData[prod(yDims,tdims)];   fillRandom(dyData,prod(yDims,tdims));
    // end random data

    double *x_h = &xData[0], *dy_h = &dyData[0]; // given
    double y_h[prod(yDims,tdims)], dx_h[prod(xDims,tdims)]; // compute kudnn
    double y1_h[prod(yDims,tdims)], dx1_h[prod(xDims,tdims)]; // compute cudnn
    // gpu pointers
    double *x_d, *dy_d;
    double *y_d, *dx_d; 
    double *y1_d, *dx1_d; 

    // send x, dy to GPU
    gpuErrchk( cudaMalloc(&x_d,     sizeof(double)*prod(xDims,tdims)) );
    gpuErrchk( cudaMalloc(&dy_d,    sizeof(double)*prod(yDims,tdims)) );

    gpuErrchk( cudaMemcpy(x_d, x_h, sizeof(double)*prod(xDims,tdims), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dy_d, dy_h, sizeof(double)*prod(yDims,tdims), cudaMemcpyHostToDevice) );
    // end send x, dy to GPU

    // y, dx
    gpuErrchk( cudaMalloc(&dx_d,    sizeof(double)*prod(xDims,tdims)) );
    gpuErrchk( cudaMalloc(&dx1_d,   sizeof(double)*prod(xDims,tdims)) );
    gpuErrchk( cudaMalloc(&y_d,     sizeof(double)*prod(yDims,tdims)) );
    gpuErrchk( cudaMalloc(&y1_d,    sizeof(double)*prod(yDims,tdims)) );

    // memset!
    gpuErrchk( cudaMemset(y_d, 0,       sizeof(double)*prod(yDims,tdims)) );
    gpuErrchk( cudaMemset(dx_d, 0,      sizeof(double)*prod(xDims,tdims)) );
    gpuErrchk( cudaMemset(y1_d, 0,      sizeof(double)*prod(yDims,tdims)) );
    gpuErrchk( cudaMemset(dx1_d, 0,     sizeof(double)*prod(xDims,tdims)) );


    // forward test
    double alpha=1, beta=1;
    printf("y:\n");
    cudnnErrchk( kudnnPoolingForward(handle, poolDesc, &alpha, xDesc, x_d, &beta, yDesc, y_d) );
    gpuErrchk( cudaMemcpy(y_h, y_d, sizeof(double)*prod(yDims,tdims), cudaMemcpyDeviceToHost) );
    if(verbose){ print2Dd(y_h, yDims[2], yDims[3]); printf("\n");} 

    if(compare){
        cudnnErrchk( cudnnPoolingForward(handle, poolDesc, &alpha, xDesc, x_d, &beta, yDesc, y1_d) );
        gpuErrchk( cudaMemcpy(y1_h, y1_d, sizeof(double)*prod(yDims,tdims), cudaMemcpyDeviceToHost) );
        if(verbose){ print2Dd(y1_h, yDims[2], yDims[3]); printf("\n");} 
        assert(eqseq(y_h,y1_h,prod(yDims,tdims)) < 1.0E-4);
    }
    printf("y: ok.\n\n");
    // end forward test 

    // backward test
    printf("dx:\n");
    cudnnErrchk( kudnnPoolingBackward(handle, poolDesc, &alpha, yDesc, y_d, dyDesc, dy_d, xDesc, x_d, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaMemcpy(dx_h, dx_d, sizeof(double)*prod(xDims,tdims), cudaMemcpyDeviceToHost) );
    if(verbose){ print2Dd(dx_h, xDims[2], xDims[3]); printf("\n");} 

    if(compare){
        cudnnErrchk( cudnnPoolingBackward(handle, poolDesc, &alpha, yDesc, y_d, dyDesc, dy_d, xDesc, x_d, &beta, dxDesc, dx1_d) );
        gpuErrchk( cudaMemcpy(dx1_h, dx1_d, sizeof(double)*prod(xDims,tdims), cudaMemcpyDeviceToHost) );
        if(verbose){ print2Dd(dx1_h, xDims[2], xDims[3]); printf("\n");} 
        assert(eqseq(dx_h,dx1_h,prod(xDims,tdims)) < 1.0E-4);
    }
    printf("dx:ok\n");
    // end backward test

    // destroy
    if (xDesc != NULL) cudnnDestroyTensorDescriptor(xDesc);
    if (dxDesc != NULL) cudnnDestroyTensorDescriptor(dxDesc);
    if (dyDesc != NULL) cudnnDestroyTensorDescriptor(dyDesc);
    if (poolDesc != NULL) cudnnDestroyPoolingDescriptor(poolDesc);
    if (handle != NULL) cudnnDestroy(handle);
    // end destroy

    // free
    cudaFree(x_d); cudaFree(dy_d);
    cudaFree(y_d); cudaFree(dx_d); 
    cudaFree(y1_d); cudaFree(dx1_d); 
    // end free
}

