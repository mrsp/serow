#ifndef  __butLPF_H__
#define  __butLPF_H__


#include <string.h>
#include <math.h>
#include <iostream>
using namespace std;

class butterworthLPF
{
    
private:
    double x_p, x_pp, y_p,y_pp;
    double a1, a2, b0, b1, b2, ff, ita, q, a;
    double fx, fs;
    int i;
    
public:
    string name;
    void reset();
    
    /** @fn void filter(double y)
     *  @brief filters the  measurement with a 2nd order Butterworth filter
     */
    double filter(double y);
    
    butterworthLPF();
    void init(string name_ ,double fsampling, double fcutoff);
    
};
#endif
