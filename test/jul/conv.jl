using CUDNN: CD, cudnnConvolutionForward, CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION, cudnnGetConvolutionNdDescriptor

function cudnnConvolutionForward(x,w; cd=nothing)
    nd, pads, strides, upscale, mode, dtype = cd != nothing ? cudnnGetConvolutionNdDescriptor(cd) : (2, (0,0), (1,1), (1,1), CUDNN_CONVOLUTION, Float32)
    pad, stride = pads[1], strides[1]
end


# outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride
###
# cudnnConvolutionForward(x, w, y;
#    padding=c.padding, stride=c.stride, upscale=c.upscale, mode=c.mode,
#                            algorithm=c.algorithm, workSpace=c.workSpace, workSpaceSizeInBytes=c.workSpaceSizeInBytes,
#                                                        alpha=c.alpha, beta=c.beta)
###
function kudnnConvolutionForward(x,w; cd=nothing)
    # x: (W,H,C,N)
    # w: (W,H,C,K) 
    # y: (W,H,K,N) 
    nd, pads, strides, upscale, mode, dtype = cd != nothing ? cudnnGetConvolutionNdDescriptor(cd) : (2, (0,0), (1,1), (1,1), CUDNN_CONVOLUTION, Float32)
    pad, stride = pads[1], strides[1]
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw
    @assert mode == CUDNN_CONVOLUTION
    Wy,Hy = floor(Int, 1 + (Int[Wx,Hx] + 2*pad - Int[Ww,Hw]) / stride)
    # Wy,Hy = 1 + div(Int[Wx,Hx] + 2*pad - Int[Ww,Hw], stride)
    y = zeros(dtype, (Wy,Hy,K,N))
    for n in 1:N, k in 1:K, c in 1:Cx
        y[:,:,k,n] += _conv2(x[:,:,c,n], w[:,:,c,k]; pad=pad, stride=stride, xcorr=CUDNN_CONVOLUTION!=mode)
    end
    return y
end

# dw = rot180(xcorr(x,dy))
function kudnnConvolutionBackwardFilter(x,dy,dw; cd=nothing)
    # x:    (Wx,Hx,Cx,N)
    # dy:   (Wy,Hy,K,N) 
    # dw:    (Ww,Hw,Cw,K) 
    nd, pads, strides, upscale, mode, dtype = cd != nothing ? cudnnGetConvolutionNdDescriptor(cd) : (2, (0,0), (1,1), (1,1), CUDNN_CONVOLUTION, Float32)
    pad, stride = pads[1], strides[1]
    @assert pad==0 && stride==1
    @assert mode == CUDNN_CONVOLUTION
    Wx,Hx,C,Nx = size(x)
    Wy,Hy,K,Ny = size(dy)
    dw2 = zeros(dw)
    for c in 1:C, k in 1:K, n in 1:Ny
        dw2[:,:,c,k] += rot180(_conv2(x[:,:,c,n], dy[:,:,k,n]; pad=pad, stride=stride, xcorr=true))
    end
    return dw2
end

# dx = xcorr(dy, w, 'full')
function kudnnConvolutionBackwardData(dy,w,dx; cd=nothing) # TODO change the order of parameters
    nd, pads, strides, upscale, mode, dtype = cd != nothing ? cudnnGetConvolutionNdDescriptor(cd) : (2, (0,0), (1,1), (1,1), CUDNN_CONVOLUTION, Float32)
    pad, stride = pads[1], strides[1]
    @assert pad==0 && stride==1
    @assert mode == CUDNN_CONVOLUTION
    Wy,Hy,Ky,N = size(dy)
    Ww,Hw,C,Kw = size(w)
    @assert Ky==Kw
    dx2 = zeros(dx)
    for n in 1:N, c in 1:C, k in 1:Kw
        t = conv2(dy[:,:,k,n], rot180(w[:,:,c,k]))
        # t = _conv2(dy[:,:,k,n], w[:,:,c,k]; pad=Ww-1, stride=stride, xcorr=true)
        dx2[:,:,c,n] += t
    end
    return dx2
end
