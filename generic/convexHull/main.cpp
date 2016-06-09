#include "cHull.hpp"

using namespace std;

int main() {

  int test[10] = { 1, 2, 1, 1, 5, 6, 2, 4, 6, 2 };

  Point<int>* p = FindHull(test, 2, 5);

  int j = 0;
	int count = 0;
	//printf("%d\n", h_data[0].rightID);
	for (int i=0; i<5; i++)
	{
		if (j == 0 && i != 0)
			break;
		
		cout << "j: " << j << "  ( " << p[j].Coordinates[0] << ", " << p[j].Coordinates[1] << " ) " << endl;
		//printf("j: %d  ( %d, %d )\nr: %d   l: %d\n\n", j, h_data[j].X, h_data[j].Y, h_data[j].rightID, h_data[j].leftID);
		j = p[j].RightID; 
		count++;
		//system("PAUSE");
	}
	cout << "Count: " << count << endl;
	//printf("\nCount: %d\n", count);
  return 0;
}
