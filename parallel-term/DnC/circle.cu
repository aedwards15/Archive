
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#define NUM_ELEMENTS 8388608
#define PI 3.141592654
#define r 1048576


//__global__ void divideAndConquer()
//{
//	int x;
//	int y;
//
//	double d = (2 * PI) * (NUM_ELEMENTS - 1);
//
//	if (threadIdx.x == 0 && blockIdx.x == 0)
//		x = 0 + r * cos(0);
//
//	x = 0 + r * cos((threadIdx.x + (blockDim.x * blockIdx.x)) * d);
//
//	if (threadIdx.x == 0 && blockIdx.x == 0)
//		y = 0 + r * sin(0);
//
//	y = 0 + r * sin((threadIdx.x + (blockDim.x * blockIdx.x)) * d);
//
//	__syncthreads();
//
//	printf("%d %d", x, y);
//}

int main()
{
	//divideAndConquer<<<4096, 1024>>>();
	int x = 0;
	int y = 0;
	int count = 0;
	double d = (2 * PI) / (NUM_ELEMENTS - 1);
	printf("2\n6741438\n");
	for (int i=0; i<NUM_ELEMENTS; i++)
	{
		int tempx = 1048576 + r * cos(i * d);
		int tempy = 1048576 + r * sin(i * d);
		
		if (tempx != x || tempy != y)
		{
			x = tempx;
			y = tempy;

			printf("%d %d\n", x, y);
			//count++;
		}

		
	}
	//printf("%d\n", count);
	//system("PAUSE");
    return 0;
}