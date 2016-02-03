
function cudnnPoolingForward(x, y; window=2, padding=0, stride=1, mode=0)
    @assert padding==0 && stride==1 && mode==0
    # x: (W,H,C,N)
    y2 = zeros(y)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert Nx == Ny && C==K
    for n in 1:Nx, c in 1:C, j in 1:Hx-psize+1, i in 1:Wx-psize+1
        y2[i,j,c,n] = maximum(x[i:i+psize-1,j:j+psize-1,c,n])
    end
    return y2
end

function cudnnPoolingBackward(y, dy, x, dx; window=2, padding=0, stride=1, mode=0)
    @assert padding==0 && stride==1 && mode==0
    # x: (W,H,C,N)
    dx2 = zeros(x)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert Nx == Ny && C==K
    for n in 1:Nx, c in 1:C, j in 1:Hx-window+1, i in 1:Wx-window+1
        # y2[i,j,c,n] = maximum()
        a = x[i:i+window-1,j:j+window-1,c,n]
        mi,mj = ind2sub(a,indmax(a))
        dx2[i+mi-1,j+mj-1,c,n] = dy[i,j,c,n]
    end
    return dx2
end
