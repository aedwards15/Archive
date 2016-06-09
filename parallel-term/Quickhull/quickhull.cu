#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
*	Quickhull.cu
*	Author: Michael Gruesen
*	A quickhull implementation for NVIDIA GPGPU Compute Capability 2.0
*	
*	Serial QSort Code Adapted 
*	from : Aashish Barnwal
*	source: http://www.geeksforgeeks.org/iterative-quick-sort/
*
*	Parallel QSort Code Adapted
*	from : Seiral QSort Code
**/

struct Point {
	int x, y;
	bool operator==(Point a){ return x == a.x && y == a.y;}
};

struct Edge {
	Point p;
	Point q;
	bool operator==(Edge a){return p == a.p && q == a.q;}
};

__global__ void quickhull(Point* d_input, Point* d_output, int n){
}

__host__ void launchQuickHull(Point* d_input, Point* d_output, int n){
	dim3 dimGrid;
	dim3 dimBlock;
	size_t sharedSize = n * sizeof(Edge);

	cudaError_t cErr;
	quickhull<<< dimBlock, dimGrid, sharedSize >>>(d_input, d_output, n);
	cErr = cudaDeviceSynchronize();
	if (cErr != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
}

void allocate(Point* d_input, Point* h_input, Edge* d_output, int n){
	size_t memSize = n*sizeof(Point);
	cudaMalloc((void**)&d_input, memSize);
	cudaMalloc((void**)&d_output, memSize);
	cudaMemcpy(d_input, h_input, memSize, cudaMemcpyHostToDevice);
}

void deallocate(Point* d_input, Point*d_output){
	cudaFree(d_input);
	cudaFree(d_output);
}

void printHull(Edge* out, int n){
	for (int i = 0; i < n; i++)
		fprintf(stderr, "%d,%d->%d,%d%s", out[i].p.x, out[i].p.y, out[i].q.x, out[i].q.y, ((i + 1 == n) ? "\n" : ", "));
}

void swap(Point* a, Point* b){
	Point temp = *a;
	*a = *b;
	*b = temp;
}
/**
*	Modification: Added extra conditional to allow
*	sorting by x then y if a.x == b.x
**/
int partition(Point* input, int l, int r){
	int x = input[r].x;
	int y = input[r].y;
	int i = (l - 1);
	for (int j = l; j <= r-1; j++){
		//was : if(input[j].x <= x)
		if(input[j].x < x){
			i++;
			swap(&input[i], &input[j]);
		}
		else if (input[j].x == x){
			if (input[j].y < y){
				i++;
				swap(&input[i], &input[j]);
			}
		}
	}
	swap(&input[i+1], &input[r]);
	return i+1;
}
/**
*	Code for iterative serial quicksort comes from 
*	http://www.geeksforgeeks.org/iterative-quick-sort/
*	Author: Aashish Barnwal
**/

void quickSortIterative(Point* input, int l, int r){
	int stack[r - l + 1];
	int top = -1;
	stack[++top] = l;
	stack[++top] = r;
	while (top >=0){
		r = stack[top--];
		l = stack[top--];
		int p = partition(input, l, r);

		if (p-1 > l){
			stack[++top] = l;
			stack[++top] = p-1;
		}
		if (p+1 < r){
			stack[++top] = p+1;
			stack[++top] = r;
		}
	}
}

void checkSort(Point* in, int n){
	for(int i = 0; i < n; i++){
		fprintf(stderr, "%d %d\n", in[i].x, in[i].y);
	}
}

int computeDistance(Point a, Point b, Point c){
	return (b.x - a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}

int insert(Edge* hull, Point c, Edge old, int hullCounter){
	//printHull(hull, hullCounter);
	//fprintf(stderr, "Inserting %d,%d\n", c.x, c.y);
	int insertIdx;
	for(insertIdx = 0; insertIdx < hullCounter; insertIdx++){
		if (hull[insertIdx] == old) break;
	}
	for(int i = hullCounter; i > insertIdx + 1; i--){
		hull[i] = hull[i - 1];
	}
	Edge e;
	e.q = old.q;
	e.p = c;
	old.q = c;
	hull[insertIdx] = old;
	hull[insertIdx + 1] = e;
	//printHull(hull, hullCounter+1);
	return ++hullCounter;
}

int serialFindHull(Point* set, Point a, Point b, Edge* hull, int setCounter, int setMaxIdx, int hullCounter){
	if (setCounter == 0){
		return hullCounter;	
	}
	Point c = set[setMaxIdx];
	Edge old;
	old.p = a;
	old.q = b;
	hullCounter = insert(hull, c, old, hullCounter); 
	Point* s1;
	Point* s2;

    s1 = (Point*)malloc((setCounter-2)*sizeof(Point));
    int s1counter = 0;
    int s1MaxIdx = -1;
    int s1MaxVal = 0;
    s2 = (Point*)malloc((setCounter-2)*sizeof(Point));
    int s2counter = 0;
    int s2MaxIdx = -1;
    int s2MaxVal = 0;
	for (int i = 0; i < setCounter; i++){
		Point p = set[i];
		int res;
		if ((res = computeDistance(a, c, p)) > 0){
			s1[s1counter++] = p;
			if (res > s1MaxVal){
				s1MaxIdx = s1counter - 1;
				s1MaxVal = res;
			}
		}
		else if ((res = computeDistance(c, b, p)) > 0){
			s2[s2counter++] = p;
			if (res > s2MaxVal){
				s2MaxIdx = s2counter - 1;	
				s2MaxVal = res;
			} 
		}
	}
	hullCounter = serialFindHull(s1, a, c, hull, s1counter, s1MaxIdx, hullCounter);
	hullCounter = serialFindHull(s2, c, b, hull, s2counter, s2MaxIdx, hullCounter);
	free(s1);
	free(s2);
	return hullCounter;
}

int serialHull(Point* in, Edge* out, int n){
	//memset(out, NULL, n);
    int hullCounter = 0;
	Edge a = {in[0], in[n-1]};
	a.p = in[0];
	a.q = in[n-1];
	out[hullCounter++] = a;
	Point* s1;
	Point* s2;
    s1 = (Point*)malloc((n-2)*sizeof(Point));
    int s1counter = 0;
    int s1MaxIdx = 0;
    int s1MaxVal = 0;
    s2 = (Point*)malloc((n-2)*sizeof(Point));
    int s2counter = 0;
    int s2MaxIdx = 0;
    int s2MaxVal = 0;
	for (int i = 1; i < n-2; i++){
		Point p = in[i];
		int res;
		if ((res = computeDistance(in[0], in[n-1], p)) > 0){
			s1[s1counter++] = p;
			if (res > s1MaxVal) {
				s1MaxIdx = s1counter - 1;
				s1MaxVal = res;
			}
		}
		else if ((res = computeDistance(in[n-1], in[0], p)) > 0){
			s2[s2counter++] = p;
			if (res > s2MaxVal){
				s2MaxIdx = s2counter - 1;
				s2MaxVal = res;
			}
		}
	}
	hullCounter = serialFindHull(s1, in[0], in[n-1], out, s1counter, s1MaxIdx, hullCounter);
	a.p = in[n-1];
	a.q = in[0];
	out[hullCounter++] = a;
	hullCounter = serialFindHull(s2, in[n-1], in[0], out, s2counter, s2MaxIdx, hullCounter);
	free(s1);
	free(s2);
	return hullCounter;
}

void doSerialQuickHull(Point* in, Edge* out, int n){
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//fprintf(stderr, "starting serial quick sort\n");
	cudaEventRecord(start, 0);
	quickSortIterative(in, 0, n-1);
	//fprintf(stderr, "finished serial quick sort\n");
	//checkSort(in, n);
	//fprintf(stderr, "starting serial quick hull\n");
	int hulls = serialHull(in, out, n);
	cudaEventRecord(stop, 0);
	//fprintf(stderr, "finsihed serial quick hull\n");
	//printHull(out, hulls);
	cudaEventElapsedTime(&time, start, stop);
	fprintf(stderr, "serial quickhull runtime: %f ms\n", time);
}

int main(int argc, char** argv){
	//get input passed as arg
	FILE* input = fopen(argv[1], "r");

	//get number of points
	int numPoints;
	fscanf(input, "%d", &numPoints);
	size_t memSize = numPoints * sizeof(Point);
	size_t outSize = numPoints * sizeof(Edge);
	//host input/output
	Point* h_input = (Point*)malloc(memSize);
	Edge* h_output = (Edge*)malloc(outSize);
	Edge* h_ref = (Edge*)malloc(outSize);

	//initialize input
	for (int i = 0; i < numPoints; i++){
		Point p;
		fscanf(input, "%d %d", &p.x, &p.y);
		h_input[i] = p;
	}
	fprintf(stderr, "Read input\n");
	doSerialQuickHull(h_input, h_ref, numPoints);
	fprintf(stderr, "Quick Hull completed\n");
	//device ptrs
	//Point* d_input;
	//Edge* d_output;

	//allocate and copy to card
	//allocate(d_input, h_input, d_output, numPoints);

	//launch
	//launchQuickHull(d_input, d_output, numPoints);

	//copy back
	//cudaMemcpy(h_output, d_output, numPoints*sizeof(Edge), cudaMemcpyDeviceToHost);

	//deallocate card
	//deallocate(d_input, d_output);

	//print results
	/*
	for (int i = 0; i < numPoints; i++){
		Edge e = h_output[i];
		fprintf(stderr, "%d %d\n", e.x, e.y);
	}
	*/

	return 0;
}
