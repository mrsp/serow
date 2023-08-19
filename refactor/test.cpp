#include "State.hpp"
#include "ContactEKF.hpp"
#include <iostream>

int main() 
{
    State state;
    std::cout<<"Output is "<<std::endl;
    std::cout<<state.getBaseAngularVelocity().transpose()<<std::endl;

    ContactEKF base_ekf;
    base_ekf.init(state);

    return 0;
}
