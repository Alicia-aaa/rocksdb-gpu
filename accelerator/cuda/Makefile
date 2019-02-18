# This Makefile uses only test for CUDA-g++ link testing...
# Don't use this makefile for build up rocksDB

build:
	nvcc -dc block_decoder.cu -o block_decoder.o
	nvcc -dc filter.cu -o filter.o
	nvcc -dlink block_decoder.o filter.o -o cuda_library.o
	clang++ -c filter_test.cc -o filter_test.o -std=c++11
	clang++ -o runnable_filter_test cuda_library.o filter_test.o  \
		-L/usr/local/cuda/lib64                               \
		-lcudart

run:
	./runnable_filter_test

clean:
	rm -rf filter.o filter_test.o runnable_filter_test