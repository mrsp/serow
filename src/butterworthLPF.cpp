#include <humanoid_state_estimation/butterworthLPF.h>


butterworthLPF::butterworthLPF()
{
    
    
    
    fx = 0;
    fs = 0;
    a1 = 0;
    a2 = 0;
    b0 = 0;
    b1 = 0;
    b2 = 0;
    ff = 0;
    ita = 0;
    q = 0;
    i = 0;
    y_p = 0;
    y_pp = 0;
    x_p = 0;
    x_pp = 0;
    
    
    
}



void butterworthLPF::reset()
{
    
    
    fx = 0;
    fs = 0;
    a1 = 0;
    a2 = 0;
    b0 = 0;
    b1 = 0;
    b2 = 0;
    ff = 0;
    ita = 0;
    q = 0;
    i = 0;
    y_p = 0;
    y_pp = 0;
    x_p = 0;
    x_pp = 0;
    cout<<name<<" Butterworth Filter Reseted"<<endl;
    
    
}

void butterworthLPF::init(string name_, float fsampling, float fcutoff)
{
    fs = fsampling;
    fx = fcutoff;
    
    ff = 2*fx/fs;
    ita =1.0/ tan(3.14159265359*ff);
    q=sqrt(2.0);
    b0 = 1.0 / (1.0 + q*ita + ita*ita);
    b1= 2*b0;
    b2= b0;
    a1 = 2.0 * (ita*ita - 1.0) * b0;
    a2 = -(1.0 - q*ita + ita*ita) * b0;
    name = name_;
    a = (3.14159265359*ff)/(3.14159265359*ff+1);
    cout<<name<<" Butterworth Filter Initialized Successfully"<<endl;
    
}


/** butterworthLPF filter to  deal with the Noise **/
float butterworthLPF::filter(float  y)
{
    float out;
    if(i>2)
        out = b0 * y + b1 * y_p + b2* y_pp + a1 * x_p + a2 * x_pp;
    else{
        out = x_p + a * (y - x_p);
        i++;
    }
    
    
    y_pp = y_p;
    y_p = y;
    x_pp = x_p;
    x_p = out;
    
    return out;
    
    
    /** ------------------------------------------------------------- **/
}
