
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>

#define NUM_ELEMENTS 7
#define MAX_ELEMENTS_BLOCK 2048

struct Point
{
	unsigned int X;
	unsigned int Y;
	unsigned int leftID; // counter-clockwise neighbor
	unsigned int rightID; // clockwise neighbor
};

extern __shared__ Point hullData[];

cudaError_t convexHull(Point* h_data, int numPoints);

Point* h_data;

Point* d_data;

__device__ void findHull(int &currA, int &currB)
{
	int result;
	//int startIndex;
	int currAorig = currA;
	int currBorig = currB;
	Point c;
	bool isEven = (threadIdx.x % 2) == 0;

	if (isEven)
	{
		c = hullData[hullData[currA].leftID];
	}
	else
	{
		c = hullData[hullData[currA].rightID];
	}

	bool hullFound = false;
	while (!hullFound)
	{
		result = ((hullData[currB].X - hullData[currA].X)*(c.Y - hullData[currA].Y) - (hullData[currB].Y - hullData[currA].Y)*(c.X - hullData[currA].X));

		/*if (i == 1 && (idx == 49 || idx == 48))
			printf("idx: %d      a: %d   b: %d\n", idx, currA, currB);*/

		if (isEven)
		{
			if (result >= 0 && hullData[currA].leftID != currAorig)
			{
				currA = hullData[currA].leftID;
				c = hullData[hullData[currA].leftID];
			}
			else
			{
				c = hullData[hullData[currB].rightID];
				
				//result = ((b.X - a.X)*(c.Y - a.Y) - (b.Y - a.Y)*(c.X - a.X));
				result = ((hullData[currB].X - hullData[currA].X)*(c.Y - hullData[currA].Y) - (hullData[currB].Y - hullData[currA].Y)*(c.X - hullData[currA].X));
				
				if (result >= 0 && hullData[currB].rightID != currBorig)
				{
					currB = hullData[currB].rightID;
					c = hullData[hullData[currA].leftID];
				}
				else
				{
					hullFound = true;
				}
			}
		}
		else
		{
			if (result <= 0 && hullData[currA].rightID != currAorig)
			{
				currA = hullData[currA].rightID;
				c = hullData[hullData[currA].rightID];
			}
			else
			{
				c = hullData[hullData[currB].leftID];
				
				result = ((hullData[currB].X - hullData[currA].X)*(c.Y - hullData[currA].Y) - (hullData[currB].Y - hullData[currA].Y)*(c.X - hullData[currA].X));
				
				if (result <= 0 && hullData[currB].leftID != currBorig)
				{
					currB = hullData[currB].leftID;
					c = hullData[hullData[currA].rightID];
				}
				else
				{
					hullFound = true;
				}
			}
		}
	}
}

__device__ void findHull(int &currA, int &currB, Point* data)
{
	int result;
	//int startIndex;
	int currAorig = currA;
	int currBorig = currB;
	Point c;
	bool isEven = (threadIdx.x % 2) == 0;

	if (isEven)
	{
		c = data[data[currA].leftID];
	}
	else
	{
		c = data[data[currA].rightID];
	}

	//if (threadIdx.x == 0)
	//		printf("thread: %d\n   currA: %d ( %d, %d )     currB: %d ( %d, %d )    c: ( %d, %d )\n", threadIdx.x, currA, data[currA].X, data[currA].Y, currB, data[currB].X, data[currB].Y, c.X, c.Y);

	/*if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		int j = 0;
		int stop = j;
		int count = 0;
		//printf("%d\n", h_data[0].rightID);
		for (int i=0; i<50; i++)
		{
			if (j == stop && i != 0)
				break;

			printf("i: %d  ( %d, %d )\nr: %d   l: %d\n\n", i, data[j].X, data[j].Y, data[j].rightID, data[j].leftID);
			j = data[j].rightID; 
			count++;
		}

		printf("\nCount: %d\n", count);
	}*/

	bool hullFound = false;
	while (!hullFound)
	{
		result = ((data[currB].X - data[currA].X)*(c.Y - data[currA].Y) - (data[currB].Y - data[currA].Y)*(c.X - data[currA].X));

		/*if (i == 1 && (idx == 49 || idx == 48))
			printf("idx: %d      a: %d   b: %d\n", idx, currA, currB);*/

		if (isEven)
		{
			if (result >= 0 && data[currA].leftID != currAorig)
			{
				currA = data[currA].leftID;
				c = data[data[currA].leftID];
			}
			else
			{
				c = data[data[currB].rightID];
				
				//result = ((b.X - a.X)*(c.Y - a.Y) - (b.Y - a.Y)*(c.X - a.X));
				result = ((data[currB].X - data[currA].X)*(c.Y - data[currA].Y) - (data[currB].Y - data[currA].Y)*(c.X - data[currA].X));
				
				if (result >= 0 && data[currB].rightID != currBorig)
				{
					currB = data[currB].rightID;
					c = data[data[currA].leftID];
				}
				else
				{
					hullFound = true;
				}
			}
		}
		else
		{
			if (result <= 0 && data[currA].rightID != currAorig)
			{
				currA = data[currA].rightID;
				c = data[data[currA].rightID];
			}
			else
			{
				c = data[data[currB].leftID];
				
				result = ((data[currB].X - data[currA].X)*(c.Y - data[currA].Y) - (data[currB].Y - data[currA].Y)*(c.X - data[currA].X));
				
				if (result <= 0 && data[currB].leftID != currBorig)
				{
					currB = data[currB].leftID;
					c = data[data[currA].rightID];
				}
				else
				{
					hullFound = true;
				}
			}
		}
	}
}

__global__ void divideAndConquer(Point* data, int numElements)
{
	int idx = threadIdx.x;
	int bidx = blockIdx.x;
	int numElementsPBlock = blockDim.x * 2;
	int numThreads = blockDim.x;
	//int numBlocks = gridDim.x;
	bool isEven = (idx % 2) == 0;
/*
	if (idx == 0)
	{
		printf("%d\n", idx);
		printf("%d\n", bidx);
		printf("%d\n", numElementsPBlock);
		printf("%d\n", numElements);
		printf("%d\n", numThreads);
		printf("%d\n", numBlocks);
	}*/

	hullData[idx] = data[idx + (numElementsPBlock * bidx)];

	if ((idx + (numElementsPBlock * bidx)) + numThreads < numElements)
		hullData[idx + numThreads] = data[(idx + (numElementsPBlock * bidx)) + numThreads];
	/*hullData[idx + (2 * blockDim.x)] = data[idx + (2 * blockDim.x)];
	hullData[idx + (3 * blockDim.x)] = data[idx + (3 * blockDim.x)];*/
	__syncthreads();
	
	if ((idx << 1) + 1 < numElementsPBlock)
	{
		hullData[(idx << 1)].leftID = (idx << 1) + 1;
		hullData[(idx << 1)].rightID = (idx << 1) + 1;
		hullData[(idx << 1) + 1].leftID = (idx << 1);
		hullData[(idx << 1) + 1].rightID = (idx << 1);
	}
	else
	{
		hullData[(idx << 1)].leftID = (idx << 1);
		hullData[(idx << 1)].rightID = (idx << 1);
	}
	
	//printf("thread: %d\n  (%d, %d)\n  neighborRight: %d\n  neighborLeft: %d\n  (%d, %d)\n  neighborRight: %d\n  neighborLeft: %d\n", idx, hullData[(idx << 1)].X, hullData[(idx << 1)].Y, hullData[(idx << 1)].rightID, hullData[(idx << 1)].leftID, hullData[(idx << 1) + 1].X, hullData[(idx << 1) + 1].Y, hullData[(idx << 1) + 1].rightID, hullData[(idx << 1) + 1].leftID);
	
	
	/*int currA = startIndex + 1;
	int currB = startIndex + 2;
	Point c = hullData[hullData[currA].leftID];
	
	if (!isEven)
	{
		c = hullData[hullData[currA].rightID];
	}*/
	//int startIndex;
	int currA;
	int currB;

	__syncthreads();
	for (int i = 1; i < ((numElementsPBlock + 1) / 2); i *= 2)
	{
		int index = 4 * i * (idx / 2);

		/*if (idx == 0)
			printf("-------------------- i = %d --------------------\n", i);
		__syncthreads();
		if (i == 2 && (idx == 49 || idx == 48))
		printf("thread: %d\n  %d: (%d, %d)\n  neighborRight: %d\n  neighborLeft: %d\n  %d: (%d, %d)\n  neighborRight: %d\n  neighborLeft: %d\n", idx, (idx << 1), hullData[(idx << 1)].X, hullData[(idx << 1)].Y, hullData[(idx << 1)].rightID, hullData[(idx << 1)].leftID, ((idx << 1) + 1), hullData[(idx << 1) + 1].X, hullData[(idx << 1) + 1].Y, hullData[(idx << 1) + 1].rightID, hullData[(idx << 1) + 1].leftID);*/
		if (index + (i << 1) < numElementsPBlock)
		{
			currA = index + (i << 1) - 1;
			currB = index + (i << 1);

			findHull(currA, currB);
		}

		__syncthreads();
		if (index + (i << 1) < numElementsPBlock)
		{
			if (isEven)
			{
				hullData[currA].rightID = currB;
				hullData[currB].leftID = currA;
			}
			else
			{
				hullData[currA].leftID = currB;
				hullData[currB].rightID = currA;
			}
		}

		//__syncthreads();
		//	if (isEven)
		//	{
		//		int j = 0;
		//		int count = 0;
		//		//printf("%d\n", h_data[0].rightID);
		//		for (int i=0; i<numElements; i++)
		//		{
		//			if (j != 0 || i == 0)
		//			printf("id: %d    %d, %d\n", idx, hullData[j].X, hullData[j].Y);
		//			else 
		//				break;

		//			j = hullData[j].rightID; 
		//			count++;
		//			//system("PAUSE");
		//			
		//		}
		//		__syncthreads();
		//		printf("\nCount: %d\n", count);
		//	}
	}
	
	__syncthreads();

	hullData[idx].rightID = (hullData[idx].rightID + (numElementsPBlock * blockIdx.x));
	hullData[idx].leftID = (hullData[idx].leftID + (numElementsPBlock * blockIdx.x));

	if (idx + numThreads < numElementsPBlock)
	{
		hullData[idx + numThreads].rightID = (hullData[idx + numThreads].rightID + (numElementsPBlock * blockIdx.x));
		hullData[idx + numThreads].leftID = (hullData[idx + numThreads].leftID + (numElementsPBlock * blockIdx.x));
	}
	__syncthreads();

	data[idx + (numElementsPBlock * bidx)] = hullData[idx];

	if ((idx + (numElementsPBlock * bidx)) + numThreads < numElements)
		data[(idx + (numElementsPBlock * bidx)) + numThreads] = hullData[idx + numThreads];
		
	__syncthreads();
}

__global__ void divideAndConquerBlocks(Point* data, int numElements, int iteration)
{
	int idx = threadIdx.x;
	//int bidx = blockIdx.x;
	bool isEven = (idx % 2) == 0;
	
	int currA;
	int currB;

	//currA = ((((MAX_ELEMENTS_BLOCK * 2) * ((idx / 2) + 1)) + (MAX_ELEMENTS_BLOCK * (MAX_ELEMENTS_BLOCK / 4) * bidx)) * iteration) - 1;
	//currB = ((((MAX_ELEMENTS_BLOCK * 2) * ((idx / 2) + 1)) + (MAX_ELEMENTS_BLOCK * (MAX_ELEMENTS_BLOCK / 4) * bidx)) * iteration);
	
	currA = (((MAX_ELEMENTS_BLOCK * 2 * (idx / 2)) + MAX_ELEMENTS_BLOCK) * iteration) - 1;
	currB = (((MAX_ELEMENTS_BLOCK * 2 * (idx / 2)) + MAX_ELEMENTS_BLOCK) * iteration);
	
	  //printf("Id: %d  Before FindHull--- currA: %d   currB: %d\n", idx, currA, currB);

	findHull(currA, currB, data);

	   //printf("Id: %d  After FindHull--- currA: %d   currB: %d\n", idx, currA, currB);

	__syncthreads();
	
	if (isEven)
	{
		data[currA].rightID = currB;
		data[currB].leftID = currA;
	}
	else
	{
		data[currA].leftID = currB;
		data[currB].rightID = currA;
	}
	
	__syncthreads();
}

int main(int argc, char** argv)
{
	FILE* input;
	if (argc > 1)
	{
		input = fopen(argv[1], "r");
	}
	else
	{
		input = fopen("sorted_8192.txt", "r");
	}

	//get number of points
	int numPoints;
	fscanf(input, "%d", &numPoints);
	fscanf(input, "%d", &numPoints);
	//printf("%d\n", numPoints);
	//system("PAUSE");

	h_data = (Point*)malloc(sizeof(Point) * numPoints);

	//initialize input
	for (int i = 0; i < numPoints; i++){
		fscanf(input, "%d %d", &h_data[i].X, &h_data[i].Y);
	}
	cudaError_t cudaStatus = convexHull(h_data, numPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
		//system("PAUSE");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		//system("PAUSE");
        return 1;
    }

	free(h_data);
	//system("PAUSE");
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t convexHull(Point* h_data, int numPoints)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

	cudaStatus = cudaMalloc((void**)&d_data, numPoints * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
	
	cudaStatus = cudaMemcpy(d_data, h_data, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	//printf("\n\nNum Threads to be launched: %d\n\n", numThreads);

	int numBlocks = 1;
	if ((numPoints % MAX_ELEMENTS_BLOCK) == 0)
		numBlocks = (numPoints / MAX_ELEMENTS_BLOCK);
	else
		numBlocks = ((numPoints / MAX_ELEMENTS_BLOCK) + 1);

	int numThreads = 1;
	if (numBlocks > 1)
		numThreads = (MAX_ELEMENTS_BLOCK / 2);
	else
		numThreads = ((numPoints + 1) / 2);

	//printf("\n----------Starting first DnC---------\nnumBlocks: %d    numThreads: %d\n\n", numBlocks, numThreads);
	divideAndConquer<<<numBlocks, numThreads, sizeof(Point) * (numThreads * 2)>>>(d_data, numPoints);

	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "divideAndConquer launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

	int j = 1;
	for (int i = 1; i < ((numBlocks + 2) / 2); i *= 2)
	{
		int newNumBlocks = (numBlocks / i) / (MAX_ELEMENTS_BLOCK / 4);

		if (newNumBlocks == 0)
			newNumBlocks++;

		int newNumThreads = 1;
		int num = 1024;
		if (newNumBlocks > 1)
			newNumThreads = (MAX_ELEMENTS_BLOCK / 2);
		else
		{
			if (numBlocks > 1024)
			{
				newNumThreads = (num / j);
			}
			else
			{
				newNumThreads = ((numBlocks * 2) / j);
			}

			j *= 2;
		}

		//printf("\n----------Starting second DnC---------\nnewNumBlocks: %d    newNumThreads: %d\n\n", newNumBlocks, (newNumThreads / 2));
		divideAndConquerBlocks<<<newNumBlocks, (newNumThreads / 2)>>>(d_data, numPoints, i);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "divideAndConquer launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		}
	}
	
	cudaStatus = cudaMemcpy(h_data, d_data, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaFree(d_data);
	
	j = 0;
	int count = 0;
	FILE* f = fopen("out.dat", "w");
	for (int i=0; i<50; i++)
	{
		if (j == 0 && i != 0)
			break;

		fprintf(f, "%d %d\n",h_data[j].X, h_data[j].Y);
		j = h_data[j].rightID; 
		//system("PAUSE");
	}
	fprintf(f, "%d %d\n", h_data[0].X, h_data[0].Y);
	fclose(f);
	//printf("\nCount: %d\n", count);

	/*for (int i = 0; i < numPoints; i++)
	{
		printf("%d, %d\n", h_data[i].X, h_data[i].Y);
		system("PAUSE");
	}*/

    return cudaStatus;
}
