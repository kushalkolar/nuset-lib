from .bases import BaseModel
from ..utils import config
import tensorflow as tf
import numpy as np
from skimage import transform


class Nuset(BaseModel):
    def __init__(
            self,
            network_path: str = None,
    ):
        super().__init__()

    def train(self):
        pass

    def predict(
            self,
            img: np.ndarray,
            watershed: bool = True,
            min_score: float = 0.85,
            nms_threshold: float = 0.1,
            rescale_ratio: float = 1.0
    ) -> np.ndarray:
        if self.network_path is None:
            raise ValueError(
                'No network is currently loaded. \n'
                'Train a new network or load an existing one from disk.'
            )

        with tf.Graph().as_default():

            if rescale_ratio != 1.0:
                img = transform.rescale(img, rescale_ratio)
                height, width = img.shape[0], img.shape[1]

            height = height // 16 * 16
            width = width // 16 * 16

            img = img[:height, :width]



    def load_network(self, path: str = None):
        """
        Load an existing model. Loads the default model if ``path == None``

        Parameters
        ----------
        path : str
            path to the dir containing model files

        Returns
        -------

        """
        if path is None:
            path = config.DEFAULT_NETWORK_DIR

        self.model_path = path

