using Base.Test
using CUDArt
using CUDNN
@show CUDNN_VERSION

import CUDNN.cudnnConvolutionForward
import CUDNN.cudnnConvolutionBackwardFilter
import CUDNN.cudnnConvolutionBackwardData
import CUDNN.cudnnGetConvolutionNdForwardOutputDim
import CUDNN.cudnnPoolingForward
import CUDNN.cudnnPoolingBackward
include("conv_pool_cpu.jl")

padding=0
stride=1
# cd =  CD(; ndims=2, padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION, eltype=Float32)
x = reshape(Float32[1:16;], 4, 4, 1, 1); tx = CudaArray(x); @show x
w = reshape(Float32[1:4;], 2, 2, 1, 1); tw = CudaArray(w); @show w
y = zeros(Float32,3,3,1,1); ty = CudaArray(y); @show y
cudnnConvolutionForward(tx,tw,ty; padding=padding, stride=stride); y = to_host(ty); @show y
y2 = zeros(Float32,3,3,1,1); @show y2
cudnnConvolutionForward(x,w,y2; padding=padding, stride=stride); @show y2
@test_approx_eq y y2

dy = rand(Float32, size(y)); tdy = CudaArray(dy); @show dy
dw = zeros(Float32, size(w)); tdw = CudaArray(dw); @show dw
cudnnConvolutionBackwardFilter(tx,tdy,tdw); dw = to_host(tdw); @show dw
dw2 = zeros(Float32, size(w));  @show dw2
cudnnConvolutionBackwardFilter(x,dy,dw2); @show dw2
@test_approx_eq dw dw2

dx = zeros(Float32, size(x)); tdx = CudaArray(dx); @show dx
cudnnConvolutionBackwardData(tw, tdy, tdx); dx = to_host(tdx); @show dx
dx2 = zeros(Float32, size(x));  @show dx2
cudnnConvolutionBackwardData(w, dy, dx2); @show dx2
@test_approx_eq dx dx2

#=
x = rand(Float32,64,64,3,128); tx = CudaArray(x)
w = rand(Float32,8,8,3,10); tw = CudaArray(w)

dw = zeros(Float32, size(w)); tdw = CudaArray(dw);
dx = zeros(Float32, size(x)); tdx = CudaArray(dx);

dw2 = zeros(Float32, size(w))
dx2 = zeros(Float32, size(x));

y = zeros(Float32, cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride)); ty = CudaArray(y)
y2 = zeros(Float32, size(y))
dy = rand(Float32,size(y)); tdy = CudaArray(dy)


cudnnConvolutionForward(tx,tw,ty; padding=padding, stride=stride); y = to_host(ty)
cudnnConvolutionForward(x,w,y2; padding=padding, stride=stride);
@test_approx_eq y y2

cudnnConvolutionBackwardFilter(tx,tdy,tdw); dw = to_host(tdw)
cudnnConvolutionBackwardFilter(x,dy,dw2)
@test_approx_eq dw dw2

cudnnConvolutionBackwardData(tw, tdy, tdx); dx = to_host(tdx)
cudnnConvolutionBackwardData(w, dy, dx2);
@test_approx_eq dx dx2
=#

psize = 2
padding=0
stride=1
x = rand(Float32,4,4,1,1); tx = CudaArray(x); @show x
@show GetPoolingNdForwardOutputDim(x, window=psize, padding=padding, stride=stride, mode=0)
y = zeros(Float32, GetPoolingNdForwardOutputDim(x, window=psize, padding=padding, stride=stride, mode=0)); ty = CudaArray(y);
y2 = zeros(y)
cudnnPoolingForward(tx, ty; window=psize, padding=padding, stride=stride); y = to_host(ty); @show y
cudnnPoolingForward(x, y2; window=psize, padding=padding, stride=stride); @show y2
@test_approx_eq y y2

dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
dx2 = zeros(Float32, size(x));
dy = rand(Float32, size(y)); tdy = CudaArray(dy); @show dy
cudnnPoolingBackward(ty, tdy, tx, tdx; window=psize, padding=padding, stride=stride, mode=0); dx = to_host(tdx); @show dx
cudnnPoolingBackward(y, dy, x, dx2; window=psize, padding=padding, stride=stride, mode=0); @show dx2
@test_approx_eq dx dx2
