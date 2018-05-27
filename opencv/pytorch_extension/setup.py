from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='custom_cnn',
    ext_modules=[
        CppExtension('custom_cnn', ['custom_cnn.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
