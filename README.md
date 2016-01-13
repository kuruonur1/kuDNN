# kudnn
kuDNN exposes Convolution/Pooling Forward and Backward primitives for 5D data (3D images) which was not available at the time (cuDNN 2).
It has the same interface with NVIDIA's [cuDNN] (https://developer.nvidia.com/cudnn).
Currently in use by [KNet.jl] (https://github.com/denizyuret/Knet.jl)

To create yourself a shared object file: `cd src; make libkudnn.so`
