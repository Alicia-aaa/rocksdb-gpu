# This Makefile uses only test for icpc-g++ link testing...
# Don't use this makefile for build up rocksDB

##################################################################
# These lines is referenced by cuda/Makefile
# build:
# 	nvcc -dc block_decoder.cu -o block_decoder.o
# 	nvcc -dc filter.cu -o filter.o
# 	nvcc -dlink block_decoder.o filter.o -o cuda_library.o
# 	clang++ -c filter_test.cc -o filter_test.o -std=c++11
# 	clang++ -o runnable_filter_test cuda_library.o filter_test.o  \
# 		-L/usr/local/cuda/lib64                               \
# 		-lcudart

# run:
# 	./runnable_filter_test

# clean:
# 	rm -rf filter.o filter_test.o runnable_filter_test
##################################################################

CXX = g++
CFLAGS = -std=c++11 -mavx2

.SUFFIXES: .cc .o

SHELL := /bin/bash

AM_CXX = @echo "  CC      " $@;
AM_CCLD = @echo "  CCLD    " $@;

SOURCES = filter.cc avx_test.cc
OBJECTS = $(SOURCES:.cc=.o)
CLEAN_FILES = avx_test

.PHONY: avx_test

avx_test: $(OBJECTS)
	$(AM_CCLD)$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -rf *.o
	rm -rf $(CLEAN_FILES)

.cc.o:
	$(AM_CXX)$(CXX) $(CFLAGS) -c $< -o $@
