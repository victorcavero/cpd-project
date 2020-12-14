###CUDA_PATH=/usr/local/cuda
###SDK_PATH=/usr/local/cudasdk

CUDA_PATH=/usr/common/usg/cuda/3.2/
SDK_PATH=/home/veles/NVIDIA_GPU_Computing_SDK/C
NVCC=nvcc -arch=sm_20 -I. -Xcompiler "-Wall -O3"

###CUDA_PATH=/export/opt/cuda
###SDK_PATH=/export/opt/cuda/NVIDIA_CUDA_SDK

###CUDA_PATH=/usr/local/cuda
###SDK_PATH=/usr/local/NVIDIA_CUDA_SDK

CXXFLAGS = -O3 -Wall 
###-fopenmp

cpu: cpu-4th cpu-6th cpu-8th

clean:
	rm -f *.o *.s *.cubin .*.swp

cpu-8th: phi-GPU.cpp hermite8.h
	mpicxx -fopenmp $(CXXFLAGS) -DEIGHTH  -o $@ $<

cpu-6th: phi-GPU.cpp hermite6.h
	mpicxx -fopenmp $(CXXFLAGS) -DSIXTH  -o $@ $<

cpu-4th: phi-GPU.cpp hermite4.h

	mpicxx -fopenmp $(CXXFLAGS) -DFOURTH -o $@ $<

