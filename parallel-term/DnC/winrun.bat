"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc" -I "C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 5.5\C\common\inc" -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v5.5\common\inc" %1.cu  -o %1.exe -arch=sm_20

%1.exe
