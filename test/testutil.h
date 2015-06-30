
void fillRandom(double *E, int N);
double eqseq(double *A, double *B, int N);
int prod(int a[], int n);
void dims2strides(int dims[], int n, int strides[]);
void getPoolingNdForwardOutputDim( int xDims[], int pdims, int poolDims[], int poolPad[], int poolStride[], int yDims[]);

inline void print2Dd(double *E, int h, int w){
    int i,j;
    for(i=0;i<h;i++){
        for(j=0;j<w;j++){
            printf("%.4f\t", E[i*w+j]);
        }
        printf("\n");
    }
}
