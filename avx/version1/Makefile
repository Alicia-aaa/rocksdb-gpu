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
CFLAGS = -std=c++11
ICPC = icpc
ICPCFLAGS = -restrict -xCORE-AVX2 -std=c++11 -g -O2

.SUFFIXES: .cc .cxx .o .asm

SHELL := /bin/bash
AVX_COMPILER_DIR = /opt/intel/bin

AM_CXX = @echo "  CXX     " $@;
AM_CCLD = @echo "  CCLD    " $@;
AM_ICPC = @echo "  ICPC    " $@;
AM_ICPCLD = @echo "  ICPCLD  " $@;
AM_ICPCASM = @echo "  ICPCASM " $@;

SOURCES = simple_test.cc
OBJECTS = $(SOURCES:.cc=.o)

ICC_SOURCES = simple_filter.cxx
ICC_OBJECTS = $(ICC_SOURCES:.cxx=.o)
ICC_ASMS = $(ICC_SOURCES:.cxx=.asm)

.PHONY: initiate_avx_compiler64 simple_test inspect
initiate_avx_compiler64:
	source ${AVX_COMPILER_DIR}/compilervars.sh intel64

simple_test: $(OBJECTS) $(ICC_OBJECTS)
	$(AM_CCLD)$(CXX) $^ -o $@

inspect: $(ICC_ASMS)

clean:
	rm -rf *.o
	rm -rf *.asm

.cc.o:
	$(AM_CXX)$(CXX) $(CFLAGS) -c $< -o $@

.cxx.o:
	$(AM_ICPC)$(ICPC) $(ICPCFLAGS) -c $< -o $@

.cxx.asm:
	$(AM_ICPCASM)$(ICPC) -S -fsource-asm -c $<