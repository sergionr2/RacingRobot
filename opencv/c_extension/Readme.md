# C++ extension for fast image processing

It is based on [pybind11_opencv_numpy](https://github.com/edmBernard/pybind11_opencv_numpy).
that allows interpolation between cv::Mat <-> np.array.

Depending on the resize strategy (nearest neighbors or bilinear), the speedup compare to pure python + numpy is between x2 and x6.

## Dependencies

- [PyBind11 2.2.1](https://github.com/pybind/pybind11)
- OpenCV 3 with Eigen support

### Compile with CMake

```bash
mkdir build && cd build
# configure make
cmake ..
# generate the fast_image_processing.so library
make
# move fast_image_processing.so library in example folder
make install
```

### Compile with setup.py
WARNING: you have to manually edit the path to OpenCV + Eigen
```
./compile.sh
```

### Run
```bash
python test.py
```
