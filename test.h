

inline void print2Dd(double *E, int h, int w){
    int i,j;
    for(i=0;i<h;i++){
        for(j=0;j<w;j++){
            printf("%.4f\t", E[i*w+j]);
        }
        printf("\n");
    }
}

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
