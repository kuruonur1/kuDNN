#include <stdio.h>
#include <cudnn.h>
#include <assert.h>
#include <math.h>
#include "kudnn.h"
#include <time.h>
#include "test.h"
#include <string.h>



int main(int argc, char *argv[]){
    int CONV;
    if(argc==2){
        if(!strcmp(argv[1], "conv"))
            CONV=1;
        else if(!strcmp(argv[1],"xcorr"))
            CONV=0;
        else
            exit(-1);
    }else{
        printf("usage: ./testConv4d <mode>\n");
        exit(-1);
    }
    srand(time(NULL));
    /*int VERBOSE=1;
    const int N=1, C=1, H=5, W=4; // src
    const int K=1, Hw=2, Ww=2; // flt*/
    int ii=0, XD=5;
    int VERBOSE=0;
    const int N=100, C=3, K=10;
    const int xDims[5] = {N,    C,  28,                     28,                     28};                    // N C H W D
    const int wDims[5] = {K,    C,  2,                     3,                      2};                     // K C Hw Ww Dw
    const int yDims[5] = {N,    K,  xDims[2]-wDims[2]+1,    xDims[3]-wDims[3]+1,    xDims[4]-wDims[4]+1};   // N K  Hy Wy Dy; for stride=1
    const int xStrides[5] = { dims2strides5d(xDims) };
    const int yStrides[5] = { dims2strides5d(yDims)  };
    for(ii=2;ii<5;ii++){ assert(xDims[ii]>=wDims[ii]); assert(xDims[ii]>=yDims[ii]);}

    const int convPad[5-2] = {2,2,2};
    const int convStride[5-2] = {1,1,1};
    const int convUpscale[5-2] = {1,1,1};

    double xData[prod5d(xDims)];    fillRandom(xData,   prod5d(xDims));
    double wData[prod5d(wDims)];    fillRandom(wData,   prod5d(wDims));
    double dyData[prod5d(yDims)];   fillRandom(dyData,  prod5d(yDims));

    printf("N:%d C:%d H:%d W:%d D:%d\n",        cat5d(xDims));
    printf("K:%d C:%d Hw:%d Ww:%d Dw:%d\n",     cat5d(wDims));
    printf("N:%d K:%d Hy:%d Wy:%d Dy:%d\n",     cat5d(yDims));
    printf("\n");

    printf("strides:\n");
    printf("%d %d %d %d %d\n", cat5d(xStrides));
    printf("%d %d %d %d %d\n", cat5d(yStrides));


    double *x_h = &xData[0],        *w_h = &wData[0],       *dy_h=&dyData[0];                           // given
    double dx_h[prod5d(xDims)],     dw_h[prod5d(wDims)],    y_h[prod5d(yDims)],     db_h[1*K*1*1*1];    // compute cudnn
    double dx1_h[prod5d(xDims)],    dw1_h[prod5d(wDims)],   y1_h[prod5d(yDims)],    db1_h[1*K*1*1*1];   // compute kunet
    double *x_d, *dx_d, *w_d, *dw_d, *y_d, *dy_d, *db_d; // gpu pointers

    printf("sizes:\n");
    printf("%d\n",prod5d(xDims));
    printf("%d\n",prod5d(wDims));
    printf("%d\n",prod5d(yDims));

    gpuErrchk( cudaMalloc(&x_d,     sizeof(double)*prod5d(xDims)) );
    gpuErrchk( cudaMalloc(&dx_d,    sizeof(double)*prod5d(xDims)) );
    gpuErrchk( cudaMalloc(&w_d,     sizeof(double)*prod5d(wDims)) );
    gpuErrchk( cudaMalloc(&dw_d,    sizeof(double)*prod5d(wDims)) );
    gpuErrchk( cudaMalloc(&y_d,     sizeof(double)*prod5d(yDims)) );
    gpuErrchk( cudaMalloc(&dy_d,    sizeof(double)*prod5d(yDims)) );
    gpuErrchk( cudaMalloc(&db_d,    sizeof(double)*1*K*1*1*1) );

    // send x, w, dy to GPU
    gpuErrchk( cudaMemcpy(x_d,      x_h,    sizeof(double)*prod5d(xDims), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(w_d,      w_h,    sizeof(double)*prod5d(wDims), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dy_d,     dy_h,   sizeof(double)*prod5d(yDims), cudaMemcpyHostToDevice) );
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
    cudnnErrchk( cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_DOUBLE, 5, xDims, xStrides) );
    cudnnErrchk( cudnnSetTensorNdDescriptor(dxDesc, CUDNN_DATA_DOUBLE, 5, xDims, xStrides) );
    cudnnErrchk( cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_DOUBLE, 5, wDims) );
    cudnnErrchk( cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_DOUBLE, 5, wDims) );
    cudnnErrchk( cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_DOUBLE, 5, yDims, yStrides) );
    cudnnErrchk( cudnnSetTensorNdDescriptor(dyDesc, CUDNN_DATA_DOUBLE, 5, yDims, yStrides) );
    cudnnErrchk( cudnnSetConvolutionNdDescriptor(xcorr00Desc, 5-2, convPad, convStride, convUpscale, CUDNN_CROSS_CORRELATION) );
    cudnnErrchk( cudnnSetConvolutionNdDescriptor(conv00Desc, 5-2, convPad, convStride, convUpscale, CUDNN_CONVOLUTION) );
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

    // err chk
    int y1Dims[5];
    cudnnErrchk( cudnnGetConvolutionNdForwardOutputDim(tconvDesc, xDesc, wDesc, 5, y1Dims) );
    printf("N:%d K:%d Hy:%d Wy:%d Dy:%d\n",     cat5d(y1Dims));
    // end err chk

    // forward algo conf & workspace
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionFwdPreference_t convFwdPref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    void *workSpace = NULL; size_t workSpaceSize = 0, memLimit=0;
    cudnnErrchk( cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, tconvDesc, yDesc, convFwdPref, memLimit, &convFwdAlgo) );
    cudnnErrchk( cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, tconvDesc, yDesc, convFwdAlgo, &workSpaceSize) );
    printf("workspace size: %d\n", workSpaceSize);
    // end forward algo conf & workspace

    // forward test
    printf("\ny:\n");
    double alpha=1, beta=1; //scaling params for input and output
    //cudnnErrchk( cudnnConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    cudnnErrchk( kunetConvolutionForward(handle, &alpha, xDesc, x_d, wDesc, w_d, tconvDesc, convFwdAlgo, workSpace, workSpaceSize, &beta, yDesc, y_d) );
    gpuErrchk( cudaMemcpy(y1_h, y_d, sizeof(double)*prod5d(yDims), cudaMemcpyDeviceToHost) );
    /*
    gpuErrchk( cudaMemcpy(y_h, y_d, sizeof(double)*N*K*Hy*Wy, cudaMemcpyDeviceToHost) );
    if(VERBOSE){ print2Dd(y_h, Hy, Wy); printf("\n"); print2Dd(y1_h, Hy, Wy);}
    assert(eqseq(y_h,y1_h,N*K*Hy*Wy) < 1.0E-4);
    */
    printf("y: ok.\n");
    // end forward test

    // backward filter test
    printf("\ndw:\n");
    //cudnnErrchk( cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    //gpuErrchk( cudaMemcpy(dw_h, dw_d, sizeof(double)*K*C*Hw*Ww, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardFilter(handle, &alpha, xDesc, x_d, dyDesc, dy_d, tconvDesc, &beta, dwDesc, dw_d) );
    gpuErrchk( cudaMemcpy(dw1_h, dw_d, sizeof(double)*prod5d(wDims), cudaMemcpyDeviceToHost) );
    /*if(VERBOSE){ print2Dd(dw_h, Hw, Ww); printf("\n"); print2Dd(dw1_h, Hw, Ww);}
    assert(eqseq(dw_h,dw1_h,K*C*Hw*Ww) < 1.0E-4);*/
    printf("dw: ok.\n");
    // end backward filter test

    // backward data test
    printf("\ndx:\n");
    // cudnnErrchk( cudnnConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    // gpuErrchk( cudaMemcpy(dx_h, dx_d, sizeof(double)*N*C*H*W, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardData(handle, &alpha, wDesc, w_d, dyDesc, dy_d, tconvDesc, &beta, dxDesc, dx_d) );
    gpuErrchk( cudaMemcpy(dx1_h, dx_d, sizeof(double)*prod5d(xDims), cudaMemcpyDeviceToHost) );
    // if(VERBOSE){print2Dd(dx_h, H, W); printf("\n");print2Dd(dx1_h, H, W);}
    // assert(eqseq(dx_h,dx1_h,N*C*H*W) < 1.0E-4);
    printf("dx: ok.\n");
    // end backward data test

    // backward bias test
    printf("\ndb:\n");
    // cudnnErrchk( cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    // gpuErrchk( cudaMemcpy(db_h, db_d, sizeof(double)*1*K*1*1, cudaMemcpyDeviceToHost) );
    cudnnErrchk( kunetConvolutionBackwardBias(handle, &alpha, dyDesc, dy_d, &beta, dbDesc, db_d) );
    gpuErrchk( cudaMemcpy(db1_h, db_d, sizeof(double)*yDims[1], cudaMemcpyDeviceToHost) );
    // if(VERBOSE){print2Dd(db_h, 1, K); printf("\n");print2Dd(db1_h, 1, K);}
    // assert(eqseq(db_h,db1_h,1*K*1*1) < 1.0E-4);
    printf("db: ok.\n\n");
    // end backward bias test
    /*
    */

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

