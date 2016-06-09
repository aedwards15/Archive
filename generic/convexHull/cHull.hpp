#ifndef CHULL_HPP
#define CHULL_HPP

#include <iostream>
#include <vector>

#include "concepts.hpp"
#include "composite.hpp"

template<typename T>
  concept bool String() { return __is_same_as(T, const char*); }

void hello(String str) {
  std::cout << str << '\n';
}

template<typename T>
    requires Ordered<T>() && (Integer<T>() || Real<T>())
  struct Point {
    std::vector<T> Coordinates;
    int RightID;  //clockwise
    int LeftID;   //counter-clockwise

    void AddCoordinate(T coordinate){
	Coordinates.push_back(coordinate);
    }
};

template<typename T>
void FindTangent_2d(int &currA, int &currB, Point<T> *h_data, bool isUpper)
{
    int result;
    int currAorig = currA;
    int currBorig = currB;
    Point<T> c;

    if (isUpper)
    {
	c = h_data[h_data[currA].LeftID];
    }
    else
    {
	c = h_data[h_data[currA].RightID];
    }

    bool hullFound = false;
    while (!hullFound)
    {
	result = ((h_data[currB].Coordinates[0] - h_data[currA].Coordinates[0])*(c.Coordinates[1] - h_data[currA].Coordinates[1]) -  (h_data[currB].Coordinates[1] - h_data[currA].Coordinates[1])*(c.Coordinates[0] - h_data[currA].Coordinates[0]));


	if (isUpper)
	{
	    if (result >= 0 && h_data[currA].LeftID != currAorig)
	    {
		currA = h_data[currA].LeftID;
		c = h_data[h_data[currA].LeftID];
	    }
	    else
	    {
		c = h_data[h_data[currB].RightID];
			
		result = ((h_data[currB].Coordinates[0] - h_data[currA].Coordinates[0])*(c.Coordinates[1] - h_data[currA].Coordinates[1]) - (h_data[currB].Coordinates[1] - h_data[currA].Coordinates[1])*(c.Coordinates[0] - h_data[currA].Coordinates[0]));
			
		if (result >= 0 && h_data[currB].RightID != currBorig)
		{
		    currB = h_data[currB].RightID;
		    c = h_data[h_data[currA].LeftID];
		}
		else
		{
		    hullFound = true;
		}
	    }
	}
	else
	{
	    if (result <= 0 && h_data[currA].RightID != currAorig)
	    {
		currA = h_data[currA].RightID;
		c = h_data[h_data[currA].RightID];
	    }
	    else
	    {
		c = h_data[h_data[currB].LeftID];
			
		result = ((h_data[currB].Coordinates[0] - h_data[currA].Coordinates[0])*(c.Coordinates[1] - h_data[currA].Coordinates[1]) - (h_data[currB].Coordinates[1] - h_data[currA].Coordinates[1])*(c.Coordinates[0] - h_data[currA].Coordinates[0]));
			
		if (result <= 0 && h_data[currB].LeftID != currBorig)
		{
		    currB = h_data[currB].LeftID;
		    c = h_data[h_data[currA].RightID];
		}
		else
		{
		    hullFound = true;
		}
	    }
	}
    }
}

template<typename T>
void FindHull_2d(Point<T> *h_data, int numPoints)
{
    for (int i=2; i<numPoints; i*=2)
    {
	for (int j=0; j<(numPoints/(i*2)); j++)
        {
	    int currA;
	    int currB;

	    currA = ((j*2) + i) - 1;
	    currB = (j*2) + i;
	    FindTangent_2d(currA, currB, h_data, true);
	    h_data[currA].RightID = currB;
	    h_data[currB].LeftID = currA;
	    
	    currA = ((j*2) + i) - 1;
	    currB = (j*2) + i;
	    FindTangent_2d(currA, currB, h_data, false);
	    h_data[currA].LeftID = currB;
	    h_data[currB].RightID = currA;
	}
    }
}

template<typename T>
    requires Ordered<T>() && (Integer<T>() || Real<T>())
Point<T>* FindHull(std::vector<T> data, int dimensions, int numPoints)
{
    Point<T> *h_data = (Point<T>*)malloc(sizeof(Point<T>) * numPoints);


    for (int i=0; i<numPoints; i++)
    {
	for (int j=0; j<dimensions; j++)
	{
	    h_data[i].Coordinates.push_back(data[j + (i*dimensions)]);
	}
	
	if (i%2 == 0)
	    {
		h_data[i].RightID = i+1;
		h_data[i].LeftID = i+1;
	    }
	    else
	    {
		h_data[i].RightID = i-1;
		h_data[i].LeftID = i-1;
	    }
    }

    FindHull_2d(h_data, numPoints);
    
    return h_data;
}

template<typename T>
    requires Ordered<T>() && (Integer<T>() || Real<T>())
Point<T>* FindHull(T* data, int dimensions, int numPoints)
{
    std::vector<T> v;

    for (int i=0; i<(dimensions * numPoints); i++)
	v.push_back(data[i]);
    //std::vector<T> v (data, data + sizeof(data) / sizeof(T) );

    return FindHull(v, dimensions, numPoints);
}

#endif
