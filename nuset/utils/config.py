import os


if os.name == 'nt':
    IS_WINDOWS = True
    HOME = 'USERPROFILE'
else:
    IS_WINDOWS = False
    HOME = 'HOME'


DIR = os.path.join(os.environ[HOME], '.nuset')

if not os.path.isdir(DIR):
    os.makedirs(DIR)

# store the default network files here
DEFAULT_NETWORK_DIR = os.path.join(DIR, 'models', 'default')

if not os.path.isdir(DEFAULT_NETWORK_DIR):
    os.makedirs(DIR)

# default network files
_default_network_fnames = \
    [
        'foreground.ckpt.data-00000-of-00001',
        'whole_norm.ckpt.meta',
        'whole_norm.ckpt.data-00000-of-00001',
        'foreground.ckpt.index',
        'foreground.ckpt.meta',
        'whole_norm.ckpt.index'
    ]


def _download_network_file(fname: str):
    pass


for _f in _default_network_fnames:
    if not os.path.isfile(_f):
        _download_network_file(_f)
