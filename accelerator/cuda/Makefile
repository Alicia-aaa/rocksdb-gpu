# Deprecated!!!
# This Makefile uses only test for CUDA-g++ link testing...
# Don't use this makefile for build up rocksDB

.SUFFIXES: .cu .o

AR = ar
ARFLAGS = rs
ARCHITECTURE = sm_50
NVCC = nvcc
NVCCFLAGS = -dc -arch=${ARCHITECTURE} --default-stream per-thread \
	-I. -I../ -I../../ -I../../include \
	-std=c++11 \
	-DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DLZ4 -DZSTD -DNUMA -DTBB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DHAVE_SSE42 -DHAVE_PCLMUL -DROCKSDB_SUPPORT_THREAD_LOCAL \
	-isystem ../../third-party/gtest-1.7.0/fused-src
CUDA_SOURCES = \
	block_decoder.cu \
	block_filter.cu \
	block_record_filter.cu \
	block_stream_filter.cu \
	block_stream_filter_v2.cu \
	cuda_util.cu \
	simple_filter.cu
CUDAOBJECTS = $(CUDA_SOURCES:.cu=.o)
CUDABIND = cuda_bind.o
CUDALIBRARY = libruda.a

.cu.o:
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(CUDABIND): $(CUDAOBJECTS)
	$(NVCC) -dlink $^ -o $@ -lcudart -lcudadevrt -arch=${ARCHITECTURE}

$(CUDALIBRARY): $(CUDAOBJECTS) $(CUDABIND)
	$(AR) $(ARFLAGS) $@ $(CUDAOBJECTS) $(CUDABIND)

build: $(CUDALIBRARY)

clean:
	rm $(CUDALIBRARY) $(CUDABIND) $(CUDAOBJECTS)