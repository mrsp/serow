# Python Bindings

## Install the Python Package

Open a terminal in the current directory:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.bashrc
```

Install the dependencies:

```
pip3 install -r requirements.txt
```

# Build SEROW with the Python Schemas 

Remove existing build (if present):

```
cd SEROW_PATH
rm -rf build
mkdir build && cd build
```

Generate the Python schemas:

```
cmake .. -DGENERATE_PYTHON_SCHEMAS=ON
make -j8
sudo make install
```

Fix Python schemas and build the package:

```
cd SEROW_PATH/python
./add_imports.sh
python3 setup.py build_ext --inplace
```

## Tests

Run the example with: 

```
python3 serow/example.py
```
