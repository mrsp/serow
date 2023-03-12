#include "State.hpp"
#include <iostream>

int main() 
{
    State state;
    std::cout<<state.getBaseAngularVelocity().transpose()<<std::endl;
    return 0;
}