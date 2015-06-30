#include <stdio.h>
#include <math.h>
#include <assert.h>

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

int prod(int a[], int n){
    int res = 1;
    int i;
    for(i=0;i<n;i++)
        res *= a[i];
    return res;
}

void dims2strides(int dims[], int n, int strides[]){
    // #define dims2strides5d(A) A[1]*A[2]*A[3]*A[4],A[2]*A[3]*A[4],A[3]*A[4],A[4],1
    int i,j,z;
    for(i=0;i<n-1;i++){
        z=1;
        for(j=i+1;j<n;j++){
            z*=dims[j];
        }
        strides[i] = z;
    }
    strides[n-1]=1;
}

void getPoolingNdForwardOutputDim(
        int xDims[], int pdims, int poolDims[], int poolPad[], int poolStride[], int yDims[]
        ){

    int i;
    for(i=0;i<pdims;i++) assert(poolDims[i]>=poolStride[i]);

    yDims[0] = xDims[0]; yDims[1] = xDims[1]; // N K (C)
    for(i=0;i<pdims;i++){
        yDims[i+2] = 1+ceil((xDims[i+2]+2*poolPad[i]-poolDims[i])/(double)poolStride[i]);
    }
}
