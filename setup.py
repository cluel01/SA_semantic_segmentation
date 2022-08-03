from setuptools import find_packages, setup
import numpy as np


setup(
    name='pytorch_segmentation',
    packages=find_packages(),
    version='0.1.0',
    description='--- ',
    author='Christian Lülf',
    license='',
    include_dirs=[np.get_include()],
    zip_safe=False,
)