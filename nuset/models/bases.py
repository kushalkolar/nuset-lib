from ..utils import config
import numpy as np


class BaseModel:
    def __init__(
            self,
            network_path: str = None,
    ):
        self.network_path: str = network_path

    def train(self):
        pass

    def predict(self, *args, **kwargs) -> np.ndarray:
        pass

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
