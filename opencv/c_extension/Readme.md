# pybind11_opencv_numpy

An example of pybind11 for cv::Mat <-> np.array

```bash
/project folder
├── src
│   └── pybind11
│       ├── numpy.h
│       ├── stl.h
│       └── ...
├── build
├── example
│   ├── exemple.so  # generate with make
│   └── example.cpython-36m-x86_64-linux-gnu.so  # generate with setup.py
├── CMakeLists.txt
├── setup.py
└── ...
```

## Generation with make

### Link pybind11 source in src folder
```
ln -s path/to/pybind11/include/pybind11 src/pybind11
```

### Compile

```bash
mkdir build
cd build
# configure make
cmake ..
# generate the example.so library
make
# move example.so library in example folder
make install
```

### Run
```bash
python3 test.py
```

## Generation with setup.py

### install pybind11

```
pip3 install pybind11
```

### Compile

```
python3 setup.py build
```

### Install

```
python3 setup.py install 
```

or

```
mv build/lib.linux-x86_64-3.5/example/example.cpython-36m-x86_64-linux-gnu.so example/example.cpython-36m-x86_64-linux-gnu.so
```

### Run 

```
python3 test.py
```
