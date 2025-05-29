# Install Dependencies
```
sudo apt install libboost-dev cmake cmake-curses-gui g++ octave liboctave-dev freeglut3-dev
```

# Install Bayesopt
```
git clone https://github.com/rmcantin/bayesopt.git && cd bayesopt && git checkout 2b8cfc5
```
### Do not build the bayesopt examples (they will not compile successfully)
Change the flag in the CMakeLists.txt to OFF:
```
13 option(BAYESOPT_BUILD_EXAMPLES "Build examples and demos?" OFF)
```

### Build the bayesopt
```
cd <path>/bayesopt/
```
```
mkdir build && cd build
```
```
cmake ..
```
```
make
```

```
sudo make install
```

### Build the hyper tuner package
```
cd serow_hypertuner && mkdir build/ && cd build/
```
Compile the package:
```
cmake .. && make
```
### Run the hypertuner
Compile the serow_hypertuner package by 

```
cd serow_hypertuner && mkdir build/ && cd build/
```
Compile:
```
cmake .. && make 
```
And run the executable:
```
./hypertuner
```

You can specify which robot, dataset, and hyperparams-to-optimize inside the serow_hypertuner/config folder. The best hyperparameter configuration will be saved automatically when the hypertuner exits successfully at **serow/config/robot-name_best_config.json**

### New robot config
Create a .json file for your robot with name = **<robot_name>.json** udner the serow_hypertuner/config/robots/ and fill in the foot frames and joint names with the following order:

-->  Front Left
 
-->  Front Right
 
-->  Rear Left
 
-->  Rear Right

### Debug

If you have boost-related errors you will need to patch boost manually. Below you will find instructions on how to address the issues. Find the file  **boost/numeric/ublas/storage.hpp**:

#### ERROR DESTROY
```
error: ‘class std::allocator<double>’ has no member named ‘destroy’
  224 |                                     alloc_.destroy(si);
```
replace 
```
alloc_.destroy(si);
``` 
with 
```
std::allocator_traits<allocator_type>::destroy(alloc_, si);
```

#### ERROR CONSTRUCT
```
 error: ‘class std::allocator<double>’ has no member named ‘construct’
   79 |                       alloc_.construct(d, value_type());
```
replace every instance of
```
alloc_.construct(value1, value2);
```
where **value1** and **value2** are the names of the variables. 
with: 
```
std::allocator_traits<allocator_type>::construct(alloc_, value1, value2);
```
eg for the above error you need to replace 
```alloc_.construct(d, value_type());``` with ```std::allocator_traits<allocator_type>::construct(alloc_, d, value_type());```
