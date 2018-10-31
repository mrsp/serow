#ifndef  __butLPF_H__
#define  __butLPF_H__


#include <string.h>
#include <math.h>
#include <iostream>
using namespace std;

class butterworthLPF
{
    
private:
    float x_p, x_pp, y_p,y_pp;
    float a1, a2, b0, b1, b2, ff, ita, q, a;
    float fx, fs;
    int i;
    
public:
    string name;
    void reset();
    
    /** @fn void filter(float y)
     *  @brief filters the  measurement with a 2nd order Butterworth filter
     */
    float filter(float y);
    
    butterworthLPF();
    void init(string name_ ,float fsampling, float fcutoff);
    
};
#endif
