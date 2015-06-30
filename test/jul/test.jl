using CUDNN: cudnnConvolutionBackwardBias, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData, cudnnConvolutionForward
using CUDNN: ConvolutionDescriptor, CUDNN_CROSS_CORRELATION, CUDNN_CONVOLUTION
using CUDArt

corrd = ConvolutionDescriptor(padding=(1,1), mode=CUDNN_CROSS_CORRELATION)
convd = ConvolutionDescriptor(mode=CUDNN_CONVOLUTION)

x = reshape(Float64[1:9],3,3,1,1,1); tx = CudaArray(x)
w = reshape(Float64[1:4],2,2,1,1,1); tw = CudaArray(w)
@show size(x) #3x3
@show size(w) #2x2
println("x:")
println(x)
println("w:")
println(w)
println("corr y:")
println(to_host(cudnnConvolutionForward(tx, tw;convDesc=corrd)))
println("conv y:")
println(to_host(cudnnConvolutionForward(tx, tw;convDesc=convd)))
println()

#dy = rand(size(ty)); tdy = CudaArray(dy) #2x2
dy = reshape(Float64[5 6 7 8],2,2,1,1); tdy = CudaArray(dy) #2x2
@show size(dy)
println("dy:")
println(dy)
dx = zeros(x); tdx = CudaArray(dx) #3x3
dw = zeros(w); tdw = CudaArray(dw) #2x2

println("conv dw:")
cudnnConvolutionBackwardFilter(tx, tdy, tdw; convDesc=convd)
println(to_host(tdw))
println("corr dw:")
cudnnConvolutionBackwardFilter(tx, tdy, tdw; convDesc=corrd)
println(to_host(tdw))
println("conv dx:")
cudnnConvolutionBackwardData(tw, tdy, tdx; convDesc=convd)
println(to_host(tdx))
println("corr dx:")
cudnnConvolutionBackwardData(tw, tdy, tdx; convDesc=corrd)
println(to_host(tdx))

println("testing pooling..")
using CUDNN: cudnnPoolingForward, cudnnPoolingBackward, cudnnGetPoolingNdForwardOutputDim, PoolingDescriptor, CUDNN_POOLING_MAX
x = reshape(Float64[5 2 3 4 5 6 7 7 3 1 4 5], 3, 4, 1, 1); tx = CudaArray(x)
println("x:"); println(x)
dims = (2,2)
@show dims
pd1 = PoolingDescriptor(dims; padding=map(x->0,dims), stride=dims, mode=CUDNN_POOLING_MAX)
ydims = cudnnGetPoolingNdForwardOutputDim(pd1, tx)
y = zeros(ydims); ty = CudaArray(y)
cudnnPoolingForward(pd1, tx, ty)
println(to_host(ty))
dy = reshape(Float64[5 3 2 6],2,2,1,1); tdy = CudaArray(dy)
dx = zeros(x); tdx = CudaArray(dx)
cudnnPoolingBackward(pd1,ty,tdy,tx,tdx)
println(to_host(tdx))
println(to_host(ty))
