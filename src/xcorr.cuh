__global__ void krnlXCorrY5d(
        double *src, int N, int C, int H, int W, int D,
        double *flt, int Kw, int Cw, int Hw, int Ww, int Dw,
        double *dst, int Ny, int Ky, int Hy, int Wy, int Dy,
        int NySt, int KySt, int HySt, int WySt, int DySt,
        int hpad, int wpad, int dpad,
        const int lim
        );

__global__ void krnlXCorrDw5d(
        double* src, int N, int C, int H, int W, int D,
        double* flt, int Ny, int Ky, int Hy, int Wy, int Dy,
        double* dst, int Kw, int Cw, int Hw, int Ww, int Dw,
        int KwSt, int CwSt, int HwSt, int WwSt, int DwSt,
        int hpad, int wpad, int dpad,
        const int lim
        );

__global__ void krnlXCorrDx5d(
        double *src, int Ny, int Ky, int Hy, int Wy, int Dy,
        double *flt, int Kw, int Cw, int Hw, int Ww, int Dw,
        double *dst, int N, int C, int H, int W, int D,
        int NSt, int CSt, int HSt, int WSt, int DSt,
        int hpad, int wpad, int dpad,
        const int lim
        );

__global__ void krnlBackBias5d(
        double *src, int N, int C, int H, int W, int D, double *dst
        );
