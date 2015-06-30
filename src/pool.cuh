
__global__ void krnlMaxPoolY5d(  
        double *src,
        int N, int C, int H, int W, int D,
        int Hd, int Wd, int Dd,
        int Hs, int Ws, int Ds,
        double *dst,
        int Ny, int Ky, int Hy, int Wy, int Dy,
        int NySt, int KySt, int HySt, int WySt, int DySt,
        const int lim
        );

__global__ void krnlMaxPool5dDx( 
        double *y, int Ny, int Ky, int Hy, int Wy, int Dy,
        double *diffy,
        double *x,  int N, int C, int H, int W, int D,
        double *diffx,
        int Hd, int Wd, int Dd,
        int Hs, int Ws, int Ds,
        int NySt, int CySt, int HySt, int WySt, int DySt,
        const int lim
        );
