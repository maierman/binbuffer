
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='binbuffer',
    ext_modules=cythonize('binbuffer.pyx', language_level=3)
)
