# nuset-lib
[![PyPI version](https://badge.fury.io/py/nuset-lib)](https://badge.fury.io/py/nuset-lib)

NuSeT packaged as a library with an easy to use API

`nuset-lib` is based on the NuSeT package by Linfeng Yang: https://github.com/yanglf1121/NuSeT

Their paper:  https://www.biorxiv.org/content/10.1101/749754v1

Please cite their paper if you use `nuset-lib`

Training is not yet implemented but it can be used for predicting.

## Installation

`nuset-lib` can be installed via `pip`.

You will need to install `tensorflow-gpu` or `tensorflow` before installing `nuset-lib`

```
# tensorflow 1.15 will not install with older pip & setuptools
pip install --upgrade pip setuptools wheel
pip install tensorflow-gpu~=1.15 # or tensorflow~=1.15
pip install nuset-lib
```

## After installation
~1GB of network weights will be downloaded the first time that you import `nuset`. 
By default these network weight files are kept in your user home directory. 
If you do not want these files to be stored in your home directory (such as with shared computing systems, limited user quotas etc.), 
you may specify a different location by setting the following environment variable:

```
export NUSET_CONFIG=/path/to/dir
```

**On RTX 2000 series cards you will need to set the following environment variable due to a bug in tensorflow:**

```
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Basic Usage

```python
from nuset import Nuset
import numpy as np
from matplotlib import pyplot as plt
import tifffile

img = tifffile.imread('path to file')

nuset = Nuset()

mask = nuset.predict(
    image=img,
    watershed=True,
    min_score=0.8,
    nms_threshold=0.1,
    rescale_ratio=2.5
)

thr = 0.7

mask[mask < thr] = 0
mask[mask > thr] = 1

fig = plt.figure(figsize=(15, 15))

plt.imshow(img, cmap='viridis')
plt.imshow(mask, alpha=0.5)
plt.show()

```

You may benefit from preprocessing the image to adjust gamma, equalize the histogram etc.

See the example notebook for more details: https://github.com/kushalkolar/nuset-lib/blob/master/example.ipynb
