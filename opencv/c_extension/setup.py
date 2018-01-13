
#
# Much of this copied from https://github.com/pybind/python_example.git
#

import fnmatch
import os
from os.path import dirname, exists, join
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import setuptools
import glob
import sysconfig

#
# pybind-specific compilation stuff
#
#
opencv_lib_folder = '/usr/local/lib/'
opencv_include_dir = '/usr/local/include/opencv2'
cvlibs = list()
for file in glob.glob(os.path.join(opencv_lib_folder, 'libopencv_*')):
    cvlibs.append(file.split('.')[0])

cvlibs = list(set(cvlibs))
cvlibs = ['-L{}'.format(opencv_lib_folder)] + \
         ['opencv_{}'.format(lib.split(os.path.sep)[-1].split('libopencv_')[-1]) for lib in cvlibs]

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
import glob
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is preferred over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-s') # strip
            opts.append('-g0') # remove debug symbols
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", "-Wextra"]
extra_compile_args += ["-DNDEBUG", "-O3"]

ext_modules = [
    Extension(
        'test_module',
        [
            'example.cpp',
            'ndarray_converter.cpp',
        ],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            opencv_include_dir,
            '/usr/include/eigen3/'
        ],
        library_dirs=[opencv_lib_folder],
        libraries=cvlibs, #['opencv_core', 'opencv_highgui']
        language='c++11',
        extra_compile_args=extra_compile_args
    ),
]

setup(
    name='pybind11-opencv-numpy-example',
    version='0.1',
    author='Dustin Spicuzza',
    author_email='dustin@virtualroadside.com',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=None,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
