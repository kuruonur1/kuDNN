#include <float.h>
#include "util.h"

// POOLING

__global__ void krnlMaxPoolY5d(  
        double *src,
        int N, int C, int H, int W, int D,
        int Hd, int Wd, int Dd,
        int Hs, int Ws, int Ds,
        double *dst,
        int Ny, int Ky, int Hy, int Wy, int Dy,
        int NySt, int KySt, int HySt, int WySt, int DySt,
        const int lim
        ){
    int i,j,l,m,n, g,cumul, hy,wy,dy,hx,wx,dx,hsrc,wsrc,dsrc;
    double maxm;
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        j=cumul/KySt; cumul -= j*KySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt; cumul -= wy*WySt;
        dy=cumul/DySt;

        hx = hy*Hs; wx = wy*Ws; dx= dy*Ds;
        maxm = DBL_MIN;
        for(l=0; l<Hd;l++){
        for(m=0; m<Wd;m++){
        for(n=0; n<Dd;n++){
            hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
            if(hsrc >= 0 && wsrc >= 0 && dsrc >=0 && hsrc < H && wsrc < W && dsrc < D) 
                if(src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] > maxm)
                    maxm = src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)];
        }}}
        dst[g] = maxm; 
    }
}

__global__ void krnlMaxPool5dDx( 
        double *y, int Ny, int Ky, int Hy, int Wy, int Dy,
        double *diffy,
        double *x,  int N, int C, int H, int W, int D,
        double *diffx,
        int Hd, int Wd, int Dd,
        int Hs, int Ws, int Ds,
        int NySt, int CySt, int HySt, int WySt, int DySt,
        const int lim
        ){
    int i,j,l,m,n, g,cumul, hy,wy,dy,hx,wx,dx, hsrc,wsrc,dsrc, maxmhx,maxmwx,maxmdx;
    double maxm;

    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        j=cumul/CySt; cumul -= j*CySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt; cumul -= wy*WySt;
        dy=cumul/DySt;

        maxmhx = hx = hy*Hs; maxmwx = wx = wy*Ws; maxmdx = dx = dy*Ds;
        maxm = DBL_MIN;

        for(l=0; l<Hd;l++){
        for(m=0; m<Wd;m++){
        for(n=0; n<Dd;n++){
            hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
            if(hsrc >= 0 && wsrc >= 0 && dsrc >=0 && hsrc < H && wsrc < W && dsrc < D){
                if(x[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] > maxm){
                    maxm = x[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)]; // update maxm
                    maxmhx = hsrc; maxmwx = wsrc; maxmdx = dsrc; // update indexes of maxm
                }
            }
        }}}
        diffx[ind5d(C,H,W,D,i,j,maxmhx,maxmwx,maxmdx)] = diffy[ind5d(C,Hy,Wy,Dy,i,j,hy,wy,dy)];
    }
}

// end POOLING
