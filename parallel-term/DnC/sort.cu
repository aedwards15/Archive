#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

typedef int (*compfn)(const void*, const void*);

struct Point
{
	int x;
	int y;
};

Point* h_data;

int compare(struct Point * elem1, struct Point * elem2)
{
	if ( elem1->x < elem2->x)
      return -1;
   else if (elem1->x > elem2->x)
      return 1;
   else if (elem1->x == elem2->x)
   {
		if ( elem1->y < elem2->y)
			return -1;
		else if (elem1->y > elem2->y)
			return 1;
   }
   else
      return 0;
}

int main()
{
	FILE* input;

	input = fopen("circle_65536.txt", "r");

	//get number of points
	int numPoints;
	fscanf(input, "%d", &numPoints);
	fscanf(input, "%d", &numPoints);
	//printf("%d\n", numPoints);
	//system("PAUSE");

	h_data = (Point*)malloc(sizeof(Point) * numPoints);

	//initialize input
	for (int i = 0; i < numPoints; i++){
		fscanf(input, "%d %d", &h_data[i].x, &h_data[i].y);
	}

	qsort((void *) h_data, 65536, sizeof(Point), (compfn)compare);

	for (int i = 0; i < numPoints; i++)
	{
		cout << h_data[i].x << " " << h_data[i].y << endl;
	}

	return 0;
}