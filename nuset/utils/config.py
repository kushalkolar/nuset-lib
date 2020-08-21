import os
import requests
import tqdm


if os.name == 'nt':
    IS_WINDOWS = True
    HOME = 'USERPROFILE'
else:
    IS_WINDOWS = False
    HOME = 'HOME'


DIR = os.path.join(os.environ[HOME], '.nuset')

if not os.path.isdir(DIR):
    os.makedirs(DIR, exist_ok=True)

# store the default network files here
DEFAULT_NETWORK_DIR = os.path.join(DIR, 'networks', 'default')

if not os.path.isdir(DEFAULT_NETWORK_DIR):
    os.makedirs(DEFAULT_NETWORK_DIR, exist_ok=True)

# default network files
default_network_fnames = \
    [
        'foreground.ckpt.data-00000-of-00001',
        'whole_norm.ckpt.meta',
        'whole_norm.ckpt.data-00000-of-00001',
        'foreground.ckpt.index',
        'foreground.ckpt.meta',
        'whole_norm.ckpt.index'
    ]


def download_network_file(fname: str):
    print('Downloading default network file')
    url = f'https://zenodo.org/record/<RECORD_ID>/files/{fname}'

    # basically from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open('test.dat', 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise IOError(f'Could not download the following network file: {fname}')


for _f in default_network_fnames:
    if not os.path.isfile(os.path.join(DEFAULT_NETWORK_DIR, _f)):
        download_network_file(_f)
