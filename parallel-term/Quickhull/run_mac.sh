#!
/usr/local/cuda/bin/nvcc -I /Developer/NVIDIA/CUDA-4.0/samples/C/common/inc $1.cu -o $1 -arch=sm_11 -Xptxas -v

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib ./$1 Hull10.txt
