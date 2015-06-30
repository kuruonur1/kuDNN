#include "util.h"


// CROSS CORRELATION

__global__ void krnlXCorrY5d(
        double *src, int N, int C, int H, int W, int D,
        double *flt, int Kw, int Cw, int Hw, int Ww, int Dw,
        double *dst, int Ny, int Ky, int Hy, int Wy, int Dy,
        int NySt, int KySt, int HySt, int WySt, int DySt,
        int hpad, int wpad, int dpad,
        const int lim
        ){ // xcorr(x,w)
    int i,k,hy,wy,dy, j,l,m,n, g,cumul, hsrc,wsrc,dsrc, hx,wx,dx;
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NySt; cumul -= i*NySt;
        k=cumul/KySt; cumul -= k*KySt;
        hy=cumul/HySt; cumul -= hy*HySt;
        wy=cumul/WySt; cumul -= wy*WySt;
        dy=cumul/DySt;

        hx=hy-hpad; wx=wy-wpad; dx=dy-dpad;

        double sum=0;
        for(j=0;j<C;j++){ 
            for(l=0; l<Hw;l++){
            for(m=0; m<Ww;m++){
            for(n=0; n<Dw;n++){
                hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
                if(hsrc >= 0 && wsrc >= 0 &&  dsrc >= 0 &&  hsrc < H && wsrc < W && dsrc < D) 
                    sum += src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] * flt[ind5d(C,Hw,Ww,Dw,k,j,l,m,n)];
            }}}
        }
        dst[g] = sum;
    }
}

__global__ void krnlXCorrDw5d(
        double* src, int N, int C, int H, int W, int D,
        double* flt, int Ny, int Ky, int Hy, int Wy, int Dy,
        double* dst, int Kw, int Cw, int Hw, int Ww, int Dw,
        int KwSt, int CwSt, int HwSt, int WwSt, int DwSt,
        int hpad, int wpad, int dpad,
        const int lim
        ){
    // dw = xcorr(x,dy)
    int i,j,k, l,m,n, g,cumul, hw,ww,dw, hsrc,wsrc,dsrc, hx,wx,dx; // k j

    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        k=cumul/KwSt; cumul -= k*KwSt;
        j=cumul/CwSt; cumul -= j*CwSt;
        hw=cumul/HwSt; cumul -= hw*HwSt;
        ww=cumul/WwSt; cumul -= ww*WwSt;
        dw=cumul/DwSt;

        hx=hw-hpad; wx=ww-wpad; dx=dw-dpad;

        double sum=0;
        for(i=0;i<N;i++){ 
            for(l=0; l<Hy;l++){
            for(m=0; m<Wy;m++){
            for(n=0; n<Dy;n++){
                hsrc = hx+l; wsrc = wx+m; dsrc = dx+n;
                if(hsrc >= 0 && wsrc >= 0 &&  dsrc >= 0 &&  hsrc < H && wsrc < W && dsrc < D) 
                    sum += src[ind5d(C,H,W,D,i,j,hsrc,wsrc,dsrc)] * flt[ind5d(Ky,Hy,Wy,Dy,i,k,l,m,n)];
            }}}
        }
        dst[g] = sum;
    }
}

__global__ void krnlXCorrDx5d(
        double *src, int Ny, int Ky, int Hy, int Wy, int Dy,
        double *flt, int Kw, int Cw, int Hw, int Ww, int Dw,
        double *dst, int N, int C, int H, int W, int D,
        int NSt, int CSt, int HSt, int WSt, int DSt,
        int hpad, int wpad, int dpad,
        const int lim
        ){
    // dx=conv(dy,w,'full')
    int i,j,k, l,m,n, g,cumul, h,w,d, hsrc,wsrc,dsrc, hy,wy,dy; // i j

    hpad = Hw-1-hpad; wpad = Ww-1-wpad; dpad = Dw-1-dpad; // full convolution
    
    for(g = threadIdx.x + blockIdx.x * blockDim.x; g < lim; g += blockDim.x * gridDim.x){
        cumul=g;
        i=cumul/NSt; cumul -= i*NSt;
        j=cumul/CSt; cumul -= j*CSt;
        h=cumul/HSt; cumul -= h*HSt;
        w=cumul/WSt; cumul -= w*WSt;
        d=cumul/DSt;

        hy=h-hpad; wy=w-wpad; dy=d-dpad;

        double sum=0;
        for(k=0;k<Ky;k++){
            for(l=Hw-1; l>=0;l--){
            for(m=Ww-1; m>=0;m--){
            for(n=Dw-1; n>=0;n--){
                hsrc = hy+Hw-1-l; //int hsrc = h+Hf-1-l;
                wsrc = wy+Ww-1-m;
                dsrc = dy+Dw-1-n;
                if(hsrc >= 0 && wsrc >= 0 && dsrc >= 0 && hsrc < Hy && wsrc < Wy && dsrc < Dy) 
                    sum += src[ind5d(Ky,Hy,Wy,Dy,i,k,hsrc,wsrc,dsrc)] * flt[ind5d(Cw,Hw,Ww,Dw,k,j,l,m,n)];
            }}}
        }
        dst[g] = sum;
    }
}

// end CROSS CORRELATION

__global__ void krnlBackBias5d(
        double *src, int N, int C, int H, int W, int D, double *dst
        ){
    int j = threadIdx.x;
    int i,k,l,m;
    double sum=0;
    for(i=0;i<N;i++) for(k=0;k<H;k++) for(l=0;l<W;l++) for(m=0;m<D;m++) sum += src[ind5d(C,H,W,D,i,j,k,l,m)];
    dst[j] = sum;
}
