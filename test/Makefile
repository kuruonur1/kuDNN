
runtests: runtests.o testpool.o testxcorr.o testutil.o
	nvcc -o $@ $+ -L /share/apps/cuDNN/cudnn-6.5-linux-x64-v2/ -L ../src/ -lcudnn -lkudnn

%.o: %.cu
	nvcc -c $< -o $@ -I /share/apps/cuDNN/cudnn-6.5-linux-x64-v2/ 

clean:
	rm -f runtests
	rm -f *.o
