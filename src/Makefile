CFLAGS=-g -L.
CUDNN_HOME=/share/apps/cuDNN/cudnn-6.5-linux-x64-v2/

libkudnn.so: kudnn.o  pool.o  xcorr.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

%.o: %.cu
	nvcc -I ${CUDNN_HOME} -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	rm -f *.so
	rm -f *.o
