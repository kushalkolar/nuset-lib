import os
import requests
from tqdm import tqdm
from warnings import warn
from shutil import copy
from .. import network


if os.name == 'nt':
    IS_WINDOWS = True
    HOME = 'USERPROFILE'
else:
    IS_WINDOWS = False
    HOME = 'HOME'


# if config dir is specific in environment var
if 'NUSET_CONFIG' in os.environ:
    DIR = os.environ['NUSET_CONFIG']
else:
    DIR = os.path.join(os.environ[HOME], '.nuset')


if not os.path.isdir(DIR):
    os.makedirs(DIR, exist_ok=True)

# store the default network files here
DEFAULT_NETWORK_DIR = os.path.join(DIR, 'networks', 'nuset_default')

if not os.path.isdir(DEFAULT_NETWORK_DIR):
    os.makedirs(DEFAULT_NETWORK_DIR, exist_ok=True)

# default network files
large_network_fnames = \
    [
        'foreground.ckpt.data-00000-of-00001',
        'whole_norm.ckpt.data-00000-of-00001',
    ]

small_network_files = \
[
    'foreground.ckpt.index',
    'foreground.ckpt.meta',
    'whole_norm.ckpt.index',
    'whole_norm.ckpt.meta',
]


def download_network_file(fname: str):
    """
    Download the large network files from Zenodo
    """
    print(f'Downloading default network file: {fname}')
    url = f'https://zenodo.org/record/3996370/files/{fname}'

    # basically from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(os.path.join(DEFAULT_NETWORK_DIR, fname), 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise warn(f'Could not download the following network file: {fname}\n'
                   f'NuSeT features will not be functional. You can try downloading'
                   f' the file again.')


def copy_network_file(fname):
    """
    Move the small network files from the repo to the config dir
    """
    basepath = os.path.dirname(network.__file__)
    copy(
        os.path.join(basepath, fname),
        os.path.join(DEFAULT_NETWORK_DIR, fname)
    )


for _f in large_network_fnames:
    if not os.path.isfile(os.path.join(DEFAULT_NETWORK_DIR, _f)):
        download_network_file(_f)

for _f in small_network_files:
    if not os.path.isfile(os.path.join(DEFAULT_NETWORK_DIR, _f)):
        copy_network_file(_f)
