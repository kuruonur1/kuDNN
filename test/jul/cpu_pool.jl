using Base.Test
using CUDArt
using CUDNN

include("pool.jl")

using CUDNN: PD, cudnnPoolingForward, CUDNN_POOLING_MAX
psize = 2
pd = PD(2, psize, 0, 1, CUDNN_POOLING_MAX)
x = reshape(Float32[1:20;], 5, 4, 1, 1); tx = CudaArray(x); @show x
y = zeros(Float32, cudnnGetPoolingNdForwardOutputDim(pd, tx)); ty = CudaArray(y); @show y

# y = to_host(cudnnPoolingForward(tx, ty; pd=pd)); @show y
y = to_host(cudnnPoolingForward(tx, ty; pd=pd)); @show y
y2 = kudnnPoolingForward(x,y; window=psize)
@test_approx_eq y y2

using CUDNN: cudnnPoolingBackward
dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
dy = rand(Float32, size(y)); tdy = CudaArray(dy); @show dy
dx = to_host(cudnnPoolingBackward(ty, tdy, tx, tdx; pd=pd)); @show dx
dx2 = kudnnPoolingBackward(y, dy, x, dx; window=psize); @show dx2
@test_approx_eq dx dx2


psize = 8
pd = PD(2, psize, 0, 1, CUDNN_POOLING_MAX)
x = rand(Float32,64,64,3,128); tx = CudaArray(x)
y = zeros(Float32, cudnnGetPoolingNdForwardOutputDim(pd, tx)); ty = CudaArray(y);
y = to_host(cudnnPoolingForward(tx, ty; pd=pd));
y2 = kudnnPoolingForward(x,y; window=psize)
@test_approx_eq y y2

dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
dy = rand(Float32, size(y)); tdy = CudaArray(dy);
dx = to_host(cudnnPoolingBackward(ty, tdy, tx, tdx; pd=pd));
dx2 = kudnnPoolingBackward(y, dy, x, dx; window=psize);
@test_approx_eq dx dx2
