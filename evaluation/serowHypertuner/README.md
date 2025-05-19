# Install Dependencies
```
sudo apt install libboost-dev python-dev python-numpy cmake cmake-curses-gui g++ cython octave liboctave-dev freeglut3-dev
```

# Install Bayesopt
```
git clone https://github.com/rmcantin/bayesopt.git
```
### Remove the contents of example/ folder (they will not compile successfully)
```
cd <path>/bayesopt/examples
```
```
rm -rf *
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


### Run the hypertuner
Place the .h5 dataset under fsc_test/anymal_fsc/fsc.h5 and then:


Go to fsc_test/ and build.

* The fsc_test executable runs serow on anymal_fsc/fsc.h5 dataset
* to change the config file ./fsc_test best_config.json (which is saved at serow/config folder)
* When you run the ./hypertuner it creates a anymal_b_temp.json in config/ which changes all the time. After the hypertuning is finished, it stores the best results in the config/best_config.json
* then you run ./fsc_test best_config.json and then python3 serow_viz.py to watch the results


### Debug

If you have this error, due to your boost version:
```
error: ‘class std::allocator<double>’ has no member named ‘destroy’
  224 |                                     alloc_.destroy(si);
```


You will need to patch boost : boost/numeric/ublas/storage.hpp

replace **alloc_.destroy(si);** with **si->~value_type();**