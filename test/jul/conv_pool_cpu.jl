

function _conv2(x, w; pad=0, stride=1, xcorr=false)
    max_pad = map(x->x-1-pad,size(w))
    y = conv2(x, xcorr ? rot180(w) : w)
    return y[1+max_pad[1]:stride:end-max_pad[1], 1+max_pad[2]:stride:end-max_pad[2]]
end

function cudnnConvolutionForward(x::Array{Float32,4}, w, y; padding=0, stride=1, 
upscale=1, mode=0, cd=nothing, algorithm="okuru13", workSpace=0, workSpaceSizeInBytes=0, alpha=1, beta=1)
    # x: (W,H,C,N)
    # w: (W,H,C,K) 
    # y: (W,H,K,N) 
    println("conv on cpu")
    @assert padding==0 && stride==1 && upscale==1&& mode==0&& algorithm=="okuru13"&& workSpace==0&& workSpaceSizeInBytes==0&& alpha==1&& beta==1
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    for n in 1:N, k in 1:K, c in 1:Cx
        y[:,:,k,n] += _conv2(x[:,:,c,n], w[:,:,c,k]; pad=padding, stride=stride, xcorr=mode!=0)
    end
    return y
end

# dw = rot180(xcorr(x,dy))
function cudnnConvolutionBackwardFilter(x::Array{Float32,4}, dy, dw; padding=0, stride=1, upscale=1, mode=0)
    # x:    (Wx,Hx,Cx,N)
    # dy:   (Wy,Hy,K,N) 
    # dw:    (Ww,Hw,Cw,K) 
    @assert padding==0&& stride==1&& upscale==1&& mode==0
    Wx,Hx,C,Nx = size(x)
    Wy,Hy,K,Ny = size(dy)
    for c in 1:C, k in 1:K, n in 1:Ny
        dw[:,:,c,k] += rot180(_conv2(x[:,:,c,n], dy[:,:,k,n]; pad=padding, stride=stride, xcorr=true))
    end
    return dw
end

# dx = xcorr(dy, w, 'full')
function cudnnConvolutionBackwardData(w::Array{Float32,4}, dy, dx; padding=0, stride=1, upscale=1, mode=0)
    @assert padding==0&& stride==1&& upscale==1&& mode==0
    Wy,Hy,Ky,N = size(dy)
    Ww,Hw,C,Kw = size(w)
    @assert Ky==Kw
    for n in 1:N, c in 1:C, k in 1:Kw
        t = conv2(dy[:,:,k,n], rot180(w[:,:,c,k]))
        # t = _conv2(dy[:,:,k,n], w[:,:,c,k]; pad=Ww-1, stride=stride, xcorr=true)
        dx[:,:,c,n] += t
    end
    return dx
end

function cudnnGetConvolutionNdForwardOutputDim(x::Array{Float32,4},w; padding=padding,stride=stride)
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw
    Wy,Hy = floor(Int, 1 + (Int[Wx,Hx] + 2*padding - Int[Ww,Hw]) / stride)
end

function cudnnPoolingForward(x::Array{Float32,4}, y; window=2, padding=0, stride=1, mode=0)
    @assert padding==0 && stride==1 && mode==0
    # x: (W,H,C,N)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert Nx == Ny && C==K
    for n in 1:Nx, c in 1:C, j in 1:Hx-psize+1, i in 1:Wx-psize+1
        y[i,j,c,n] = maximum(x[i:i+psize-1,j:j+psize-1,c,n])
    end
    return y
end

function cudnnPoolingBackward(y::Array{Float32,4}, dy, x, dx; window=2, padding=0, stride=1, mode=0)
    @assert padding==0 && stride==1 && mode==0
    # x: (W,H,C,N)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert Nx == Ny && C==K
    for n in 1:Nx, c in 1:C, j in 1:Hx-window+1, i in 1:Wx-window+1
        # y2[i,j,c,n] = maximum()
        a = x[i:i+window-1,j:j+window-1,c,n]
        mi,mj = ind2sub(a,indmax(a))
        dx[i+mi-1,j+mj-1,c,n] = dy[i,j,c,n]
    end
    return dx
end

function GetPoolingNdForwardOutputDim(x::Array{Float32,4}; window=2, padding=0, stride=1, mode=0)
    @assert padding==0 && stride==1 && mode==0
    dims = [size(x)...]
    # (mode, pdims, window, padding, stride) = cudnnGetPoolingNdDescriptor(pd)
    for i=1:length(dims)-2
        dims[i] = 1 + ceil((dims[i] + 2*padding - window) / stride)
    end
    tuple(dims...)
end
