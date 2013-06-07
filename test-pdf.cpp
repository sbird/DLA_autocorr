#include "moments.h"
#include <math.h>
#include <iostream>
#include <map>

using namespace std;

int main()
{
    const double field[5]={1,1.5,2,2.5,3};
    map<double, int> hist = histogram(field, 5, 0.9, 4, 5);    
    for (std::map<double,int>::iterator it=hist.begin(); it!=hist.end(); ++it)
        cout << it->first << " " << it->second << endl;
}
