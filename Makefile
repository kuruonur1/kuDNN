
runtests:
	cd src; make libkudnn.so;
	cd test; make runtests;

clean:
	cd src; make clean
	cd test; make clean
	
