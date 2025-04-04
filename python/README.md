# Python Bindings

## Install the python package

Open a terminal in the current directory

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.bashrc
python3 setup.py build_ext --inplace
```

## Tests
Run the example with: 

```
python3 serow/example.py
```