build:
	nvcc -c filter.cu -o filter.o
	clang++ -c filter_test.cc -o filter_test.o -std=c++14
	clang++ -o runnable_filter_test filter.o filter_test.o \
		-L/usr/local/cuda/lib64 \
		-lcudart

run:
	./runnable_filter_test

clean:
	rm -rf filter.o filter_test.o runnable_filter_test