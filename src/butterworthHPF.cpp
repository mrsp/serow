#include <serow/butterworthHPF.h>


butterworthHPF::butterworthHPF()
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



void butterworthHPF::reset()
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
    cout<<name<<"High-Pass Butterworth Filter Reseted"<<endl;
}

void butterworthHPF::init(string name_, double fsampling, double fcutoff)
{
    fs = fsampling;
    fx = fcutoff;
    ff = 2*fx/fs;
    ita =1.0/ tan(3.14159265359*ff);
    q=sqrt(2.0);
    b0 = 1.0 / (1.0 + q*ita + ita*ita);
    b1= 2*b0;
    b2= b0;
    
    b0 = b0*ita*ita;
    b1 = -b1*ita*ita;
    b2 = b2*ita*ita;
    
    a1 = 2.0 * (ita*ita - 1.0) * b0;
    a2 = -(1.0 - q*ita + ita*ita) * b0;
    name = name_;
    a = (1.0)/(3.14159265359*ff+1);
    cout<<name<<"High-Pass Butterworth Filter Initialized Successfully"<<endl;
    
}


/** butterworthLPF filter to  deal with the Noise **/
double butterworthHPF::filter(double  y)
{
    double out;
    if(i>2)
        out = b0 * y + b1 * y_p + b2* y_pp + a1 * x_p + a2 * x_pp;
    else{
        out = a*x_p + a * (y - y_p);
        i++;
    }
    
    
    y_pp = y_p;
    y_p = y;
    x_pp = x_p;
    x_p = out;
    
    return out;
    
    
    /** ------------------------------------------------------------- **/
}
