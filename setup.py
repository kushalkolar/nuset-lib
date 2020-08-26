from setuptools import setup, find_packages


install_requires = \
    [
        "numpy", # newer numpy won't work with scikit-learn 0.13
        "scikit-image~=0.15.0", # newer scikit-image don't have the `min_size` kwarg for skimage.morphology.remove_small_holes
        "Pillow",
        "tqdm",
        "requests"
    ]

classifiers = \
    [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ]

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(
    name='nuset-lib',
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='MIT License',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    install_requires=install_requires
)
