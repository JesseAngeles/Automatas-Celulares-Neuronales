// main.cpp

#include <iostream>

#include "./ca.h"

using namespace std;

float fun(float *);

int main()
{
    cout << "Hola mundo";

    int width = 10;
    int height = 10;

    float *space = new float[width * height];    
    int *neighbourhood = new int(3);

    CA ca(width, height, space, neighbourhood, 3, fun);

    return 0;
}