libYacuDecu.so: deconv.o
	gcc -fPIC -shared -o libYacuDecu.so deconv.o -L/usr/local/cuda-9.1/lib64/ -lcufft -lcudart_static

deconv.o: 
	nvcc -O3 -Xcompiler=-fPIC -c -o deconv.o ../src/deconv.cu

install:
	mv libYacuDecu.so ../../lib/

clean:
	-rm *.o $(objects) *.so
