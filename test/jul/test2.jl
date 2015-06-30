using CUDNN: cudnnConvolutionBackwardBias, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData, cudnnConvolutionForward
using CUDNN: ConvolutionDescriptor, CUDNN_CROSS_CORRELATION, CUDNN_CONVOLUTION
using CUDArt


# x = reshape(Float64[1:9],3,3,1,1); tx = CudaArray(x)
# w = reshape(Float64[1:4],2,2,1,1); tw = CudaArray(w)
pads = (0,0); strides = (2,2)
corrd = ConvolutionDescriptor(padding=pads, stride=strides, mode=CUDNN_CROSS_CORRELATION)
x = float64(rand(1:10,5,5,1,1)); tx = CudaArray(x)
w = float64(rand(1:10,2,2,1,1)); tw = CudaArray(w)
@show size(x)
@show size(w)
@show pads
@show strides
ty = cudnnConvolutionForward(tx, tw;convDesc=corrd); y = to_host(ty)
@show size(y)
dy = float64(rand(1:10,size(ty))); tdy = CudaArray(dy)
# dy = reshape(Float64[5 6 7 8],2,2,1,1); tdy = CudaArray(dy) #2x2
println(x)
println(w)
println("dy:")
println(dy)
dx = zeros(x); tdx = CudaArray(dx)
dw = zeros(w); tdw = CudaArray(dw)

println("corr dw:")
cudnnConvolutionBackwardFilter(tx, tdy, tdw; convDesc=corrd)
println(to_host(tdw))
println("corr dx:")
cudnnConvolutionBackwardData(tw, tdy, tdx; convDesc=corrd)
println(to_host(tdx))

println("testing pooling..")
using CUDNN: cudnnPoolingForward, cudnnPoolingBackward, cudnnGetPoolingNdForwardOutputDim, PoolingDescriptor, CUDNN_POOLING_MAX
# x = reshape(Float64[5 2 3 4 5 6 7 7 3 1 4 5], 3, 4, 1, 1); tx = CudaArray(x)
x = float64(rand(1:10,10,10,1,1)); tx = CudaArray(x)
println("x:"); println(x)
pd2 = PoolingDescriptor((3,3); padding=(0,0), stride=(5,5), mode=CUDNN_POOLING_MAX)
ydims = cudnnGetPoolingNdForwardOutputDim(pd2, tx)
@show ydims
