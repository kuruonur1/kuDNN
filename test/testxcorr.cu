#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cudnn.h>
#include "../src/kudnn.h"
#include <time.h>
#include "testutil.h"
#include "../src/kuassert.h"
#include "../src/kudnn.h"



void testXcorr(
        int tdims, int xDims[], int wDims[],
        int cdims, int convPad[], int convStride[], int convUpscale[], int verbose, int compare
        ){

    int i; 
    for(i=2;i<tdims;i++) assert(xDims[i]>=wDims[i]);
    for(i=0;i<cdims;i++) {assert(convUpscale[i]==1); assert(convStride[i]==1);}

    int K=wDims[0];
    int yDims[tdims], xStrides[tdims], yStrides[tdims], dbStrides[tdims]; 
    int dbDims[tdims]; for(i=0;i<tdims;i++){ dbDims[i]=1; } dbDims[1]=K;

    cudnnHandle_t                   handle = NULL;
    cudnnTensorDescriptor_t         xDesc = NULL;
    cudnnTensorDescriptor_t         dxDesc = NULL;
    cudnnTensorDescriptor_t         yDesc = NULL;
    cudnnTensorDescriptor_t         dyDesc = NULL;
    cudnnTensorDescriptor_t         dbDesc = NULL;
    cudnnFilterDescriptor_t         wDesc = NULL;
    cudnnFilterDescriptor_t         dwDesc = NULL;
    cudnnConvolutionDescriptor_t    convDesc = NULL;


    // creation
    cudnnErrchk( cudnnCreate(                       &handle) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &xDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dxDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &yDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dyDesc) );
    cudnnErrchk( cudnnCreateTensorDescriptor(       &dbDesc) );
    cudnnErrchk( cudnnCreateFilterDescriptor(       &wDesc) );
    cudnnErrchk( cudnnCreateFilterDescriptor(       &dwDesc) );
    cudnnErrchk( cudnnCreateConvolutionDescriptor(  &convDesc) );
    // end creation

    // set
    // x, dx
    dims2strides(xDims,tdims,xStrides);
    cudnnErrchk( cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_DOUBLE, tdims, xDims, xStrides) );
    cudnnErrchk( cudnnSetTensorNdDescriptor(dxDesc, CUDNN_DATA_DOUBLE, tdims, xDims, xStrides) );
    printf("x:\t"); for(i=0;i<tdims;i++) printf("%d\t", xDims[i]); printf("\n");

    // w, dw
    cudnnErrchk( cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_DOUBLE, tdims, wDims) );
    cudnnErrchk( cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_DOUBLE, tdims, wDims) );
    printf("w:\t"); for(i=0;i<tdims;i++) printf("%d\t", wDims[i]); printf("\n");

    // conv
    cudnnErrchk( cudnnSetConvolutionNdDescriptor(convDesc, cdims, convPad, convStride, convUpscale, CUDNN_CROSS_CORRELATION) );
    printf("p:\t"); printf("\t\t"); for(i=0;i<cdims;i++) printf("%d\t", convPad[i]); printf("\tpadding\n");
    printf("s:\t"); printf("\t\t"); for(i=0;i<cdims;i++) printf("%d\t", convStride[i]); printf("\tstride\n");

    // y, dy
    cudnnErrchk( cudnnGetConvolutionNdForwardOutputDim(convDesc, xDesc, wDesc, tdims, yDims) );
    printf("y:\t"); for(i=0;i<tdims;i++) printf("%d\t", yDims[i]); printf("\n");
    dims2strides(yDims,tdims,yStrides);
    cudnnErrchk( cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_DOUBLE, tdims, yDims, yStrides) );
    cudnnErrchk( cudnnSetTensorNdDescriptor(dyDesc, CUDNN_DATA_DOUBLE, tdims, yDims, yStrides) );

    // db
    dims2strides(dbDims,tdims,dbStrides);
    cudnnErrchk( cudnnSetTensorNdDescriptor(dbDesc, CUDNN_DATA_DOUBLE, tdims, dbDims, dbStrides) );
    // end set input and conf

    srand(time(NULL));
    double xData[prod(xDims,tdims)];    fillRandom(xData,   prod(xDims,tdims));
    double wData[prod(wDims,tdims)];    fillRandom(wData,   prod(wDims,tdims));
    double dyData[prod(yDims,tdims)];   fillRandom(dyData,  prod(yDims,tdims));

    double *x_h = &xData[0],            *w_h = &wData[0],           *dy_h=&dyData[0];                       // given
    double dx_h[prod(xDims,tdims)],     dw_h[prod(wDims,tdims)],    y_h[prod(yDims,tdims)],     db_h[K];    // compute kudnn
    double dx1_h[prod(xDims,tdims)],    dw1_h[prod(wDims,tdims)],   y1_h[prod(yDims,tdims)],    db1_h[K];   // compute cudnn
    double *x_d, *w_d, *dy_d; // gpu pointers
    double *y_d,    *dw_d,  *dx_d,  *db_d;       // compute kudnn
    double *y1_d,   *dw1_d, *dx1_d, *db1_d;      // compute cudnn

    if(verbose){ print2Dd(x_h, xDims[2], xDims[3]); printf("\n");} 
    if(verbose){ print2Dd(w_h, wDims[2], wDims[3]); printf("\n");} 

    gpuErrchk( cudaMalloc(&x_d,     sizeof(double)*prod(xDims,tdims)) );
    gpuErrchk( cudaMalloc(&dx_d,    sizeof(double)*prod(xDims,tdims)) );
    gpuErrchk( cudaMalloc(&dx1_d,    sizeof(double)*prod(xDims,tdims)) );

    gpuErrchk( cudaMalloc(&w_d,     sizeof(double)*prod(wDims,tdims)) );
    gpuErrchk( cudaMalloc(&dw_d,    sizeof(double)*prod(wDims,tdims)) );
    gpuErrchk( cudaMalloc(&dw1_d,    sizeof(double)*prod(wDims,tdims)) );

    gpuErrchk( cudaMalloc(&y_d,     sizeof(double)*prod(yDims,tdims)) );
    gpuErrchk( cudaMalloc(&y1_d,     sizeof(double)*prod(yDims,tdims)) );
    gpuErrchk( cudaMalloc(&dy_d,    sizeof(double)*prod(yDims,tdims)) );
    gpuErrchk( cudaMalloc(&db_d,    sizeof(double)*K) );
    gpuErrchk( cudaMalloc(&db1_d,    sizeof(double)*K) );

    // send x, w, dy to GPU
    gpuErrchk( cudaMemcpy(x_d,      x_h,    sizeof(double)*prod(xDims,tdims), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(w_d,      w_h,    sizeof(double)*prod(wDims,tdims), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dy_d,     dy_h,   sizeof(double)*prod(yDims,tdims), cudaMemcpyHostToDevice) );
    // end send x, w, dy to GPU

    // memset!
    gpuErrchk( cudaMemset(y1_d, 0,     sizeof(double)*prod(yDims,tdims)) );
    gpuErrchk( cudaMemset(dx1_d, 0,     sizeof(double)*prod(xDims,tdims)) );
    gpuErrchk( cudaMemset(dw1_d, 0,     sizeof(double)*prod(wDims,tdims)) );
    gpuErrchk( cudaMemset(db1_d, 0,     sizeof(double)*prod(dbDims,tdims)) );
    
    // forward test

    // forward algo conf & workspace
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionFwdPreference_t convFwdPref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    void *workSpace = NULL; size_t workSpaceSize = 0, memLimit=0;
    cudnnErrchk( cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, convFwdPref, memLimit, &convFwdAlgo) );
    cudnnErrchk( cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, convFwdAlgo, &workSpaceSize) );
    // end forward algo conf & workspace

    printf("\ny:\n");
    double alpha=1, beta=1; //scaling params for input and output
    cudnnErrchk( kudnnConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, convDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    gpuErrchk( cudaMemcpy(y_h, y_d, sizeof(double)*prod(yDims,tdims), cudaMemcpyDeviceToHost) );
    if(verbose){ print2Dd(y_h, yDims[2], yDims[3]); printf("\n");} 

    if(compare){
        cudnnErrchk( cudnnConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, convDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y1_d) );
        gpuErrchk( cudaMemcpy(y1_h, y1_d, sizeof(double)*prod(yDims,tdims), cudaMemcpyDeviceToHost) );
        if(verbose){ print2Dd(y1_h, yDims[2], yDims[3]); printf("\n");} 
        assert(eqseq(y_h,y1_h,prod(yDims,tdims)) < 1.0E-4);
    }
    printf("y: ok.\n");
    // end forward test

    // backward filter test
    printf("\ndw:\n");
    cudnnErrchk( kudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, convDesc, &beta, dwDesc, dw_d) );
    gpuErrchk( cudaMemcpy(dw_h, dw_d, sizeof(double)*prod(wDims,tdims), cudaMemcpyDeviceToHost) );
    if(verbose){ print2Dd(dw_h, wDims[2], wDims[3]); printf("\n"); }

    if(compare){
        cudnnErrchk( cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, convDesc, &beta, dwDesc, dw1_d) );
        gpuErrchk( cudaMemcpy(dw1_h, dw1_d, sizeof(double)*prod(wDims,tdims), cudaMemcpyDeviceToHost) );
        if(verbose){ print2Dd(dw1_h, wDims[2], wDims[3]); printf("\n"); }
        assert(eqseq(dw_h,dw1_h,prod(wDims,tdims)) < 1.0E-4);
    }
    printf("dw: ok.\n");
    // end backward filter test

    // backward data test
    printf("\ndx:\n");
    cudnnErrchk( kudnnConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, convDesc, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaMemcpy(dx_h, dx_d, sizeof(double)*prod(xDims,tdims), cudaMemcpyDeviceToHost) );
    if(verbose){ print2Dd(dx_h, xDims[2], xDims[3]); printf("\n"); }

    if(compare){
        cudnnErrchk( cudnnConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, convDesc, &beta, dxDesc, dx1_d) );
        gpuErrchk( cudaMemcpy(dx1_h, dx1_d, sizeof(double)*prod(xDims,tdims), cudaMemcpyDeviceToHost) );
        if(verbose){ print2Dd(dx1_h, xDims[2], xDims[3]); printf("\n"); }
        assert(eqseq(dx_h,dx1_h,prod(xDims,tdims)) < 1.0E-4);
    }
    printf("dx: ok.\n");
    // end backward data test

    // backward bias test
    printf("\ndb:\n");
    cudnnErrchk( kudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    gpuErrchk( cudaMemcpy(db_h, db_d, sizeof(double)*K, cudaMemcpyDeviceToHost) );
    if(verbose){ print2Dd(db_h, 1, K); printf("\n"); }

    if(compare){
        cudnnErrchk( cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db1_d) );
        gpuErrchk( cudaMemcpy(db1_h, db1_d, sizeof(double)*K, cudaMemcpyDeviceToHost) );
        if(verbose){ print2Dd(db1_h, 1, K); printf("\n"); }
        assert(eqseq(db_h,db1_h,K) < 1.0E-4);
    }
    printf("db: ok.\n\n");
    // end backward bias test

    printf("ok.\n");
    /*
    */

    // destroy
    if (xDesc != NULL) cudnnDestroyTensorDescriptor(xDesc);
    if (dxDesc != NULL) cudnnDestroyTensorDescriptor(dxDesc);
    if (wDesc != NULL) cudnnDestroyFilterDescriptor(wDesc);
    if (dwDesc != NULL) cudnnDestroyFilterDescriptor(dwDesc);
    if (yDesc != NULL) cudnnDestroyTensorDescriptor(yDesc);
    if (dyDesc != NULL) cudnnDestroyTensorDescriptor(dyDesc);
    if (dbDesc != NULL) cudnnDestroyTensorDescriptor(dbDesc);
    if (convDesc != NULL) cudnnDestroyConvolutionDescriptor(convDesc);
    if (handle != NULL) cudnnDestroy(handle);

    // free
    cudaFree(x_d); cudaFree(w_d); cudaFree(dy_d); 
    cudaFree(dx_d); cudaFree(dw_d); cudaFree(y_d); cudaFree(db_d);
    if(compare){
        cudaFree(dx1_d); cudaFree(dw1_d); cudaFree(y1_d); cudaFree(db1_d);
    }
    // END TESTS
}
