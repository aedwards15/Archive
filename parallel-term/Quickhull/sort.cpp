#include <stdio.h>
#include <stdlib.h>
int stack[6742016];
struct Point {
	int x, y;
	//bool operator==(Point a){ return x == a.x && y == a.y;}
};

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
	fprintf(stderr, "starting sort; l:%d r:%d\n", l, r);
	
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
	FILE* f = fopen("sorted.dat", "w");
	fprintf(f, "2\n%d\n", n	);
	for(int i = 0; i < n; i++){
		fprintf(f, "%d %d\n", in[i].x, in[i].y);
	}
	fclose(f);
	free(f);
}
int main(int argc, char** argv){
	FILE* f = fopen(argv[1], "r");
	fprintf(stderr, "Opened %s\n", argv[1]);
	int dim;
	int numPoints;
	fscanf(f, "%d", &dim);
	fscanf(f, "%d", &numPoints);
	fprintf(stderr, "%d\n", numPoints);
	Point* h_input = (Point*)malloc(numPoints * sizeof(Point));
	fprintf(stderr, "setup input array\n");
	for (long i = 0; i < numPoints; i++){
		Point p;
		fscanf(f, "%d %d", &p.x, &p.y);
		h_input[i] = p;
	}
	fclose(f);
	fprintf(stderr, "read input\n");
	quickSortIterative(h_input, 0, numPoints-1);
	fprintf(stderr, "sorted input\n");
	f = fopen("sorted.dat", "w");
	fprintf(f, "2\n%d\n", numPoints);
	for(int i = 0; i < numPoints; i++){
		fprintf(f, "%d %d\n", h_input[i].x, h_input[i].y);
	}
	fclose(f);
	free(h_input);
	fprintf(stderr, "wrote sorted input\n");
	//free(f);

	return 0;	
}