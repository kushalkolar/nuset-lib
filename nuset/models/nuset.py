from .bases import BaseModel
from ..utils import config
import tensorflow as tf
import numpy as np
from skimage import transform
from tqdm import tqdm
import os
from glob import glob
from typing import *

from ..layers.models import UNET
from ..layers.model_RPN import RPN
from ..layers.anchor_size import anchor_size
from ..layers.rpn_target import RPNTarget
from ..layers.rpn_proposal import RPNProposal
from ..layers.rpn_loss import RPNLoss
from ..layers.seg_loss import segmentation_loss
from ..layers.marker_watershed import marker_watershed
from ..layers.compute_metrics import compute_metrics

from ..utils.load_data import load_data_test
from ..utils.tf_utils import optimizer_fun
from ..utils.anchors import generate_anchors_reference
from ..utils.generate_anchors import generate_anchors
from ..utils.normalization import whole_image_norm, foreground_norm, clean_image
from ..utils.losses import smooth_l1_loss
from ..utils.image_vis import draw_rpn_bbox_pred, draw_gt_boxes, draw_top_nms_proposals, draw_rpn_bbox_targets, draw_rpn_bbox_pred_only


class Nuset(BaseModel):
    def __init__(
            self,
            network_path: str = None,
    ):
        """
        NuSeT Model

        Parameters
        ----------
        network_path : Optional[str]
            Path to a dir containing a trained network

        """
        super().__init__()

        self._network_path_whole_norm: str = None
        self._network_path_foreground: str = None

        self.load_network(network_path)

    def train(self):
        pass

    def predict(
            self,
            image: np.ndarray,
            watershed: bool = True,
            min_score: float = 0.85,
            nms_threshold: float = 0.1,
            rescale_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Predict a cell mask for the input ``image``
        
        Parameters
        ----------
        image : np.ndarray
            Input image, 2D array (grayscale)

        watershed : bool
            Use a watershed transform

        min_score : float
            Min detection score, lower score means more cells will be detected

        nms_threshold : float
            NMS ratio, higher NMS ratio combined with a lower ``min_score`` allows
            more cells to be detected and separated, however segmentation error will increase.

        rescale_ratio : float
            Rescale the input ``image`` before segmentation.
            Higher values may be useful for detecting dim cells.

        Returns
        -------
        np.ndarray
            mask, 2D binary array
        """

        if self.network_path is None:
            raise ValueError(
                'No network is currently loaded. \n'
                'Train a new network or load an existing one from disk.'
            )

        with tf.Graph().as_default():

            if rescale_ratio != 1.0:
                image = transform.rescale(image, rescale_ratio)
                height, width = image.shape[0], image.shape[1]

            else:
                height, width = image.shape[0], image.shape[1]

            height = height // 16 * 16
            width = width // 16 * 16

            images = [image[:height, :width]]

            # pred_dict and pred_dict_final save all the temp variables
            pred_dict_final = {}

            train_initial = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 1])

            input_shape = tf.shape(input=train_initial)

            input_height = input_shape[1]
            input_width = input_shape[2]
            im_shape = tf.cast([input_height, input_width], tf.float32)

            # number of classes needed to be classified, for our case this equals to 2
            # (foreground and background)
            nb_classes = 2

            # feed the initial image to U-Net, we expect 2 outputs:
            # 1. feat_map of shape (?,32,32,1024), which will be passed to the
            # region proposal network
            # 2. final_logits of shape(?,512,512,2), which is the prediction from U-net
            with tf.compat.v1.variable_scope('model_U-Net') as scope:
                final_logits, feat_map = UNET(nb_classes, train_initial)

            # The final_logits has 2 channels for foreground/background softmax scores,
            # then we get prediction with larger score for each pixel
            pred_masks = tf.argmax(input=final_logits, axis=3)
            pred_masks = tf.reshape(pred_masks, [input_height, input_width])
            pred_masks = tf.cast(pred_masks, dtype=tf.float32)

            # Dynamic anchor base size calculated from median cell lengths
            base_size = anchor_size(tf.reshape(pred_masks, [input_height, input_width]))

            # scales and ratios are used to generate different anchors
            scales = np.array([0.5, 1, 2])
            ratios = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])

            # stride is to control how sparse we want to place anchors across the image
            # stride = 16 means to place an anchor every 16 pixels on the original image
            stride = 16

            # Generate the anchor reference with respect to the original image
            ref_anchors = generate_anchors_reference(base_size, ratios, scales)
            num_ref_anchors = scales.shape[0] * ratios.shape[0]

            feat_height = input_height / stride
            feat_width = input_width / stride

            # Generate all the anchors based on ref_anchors
            all_anchors = generate_anchors(ref_anchors, stride, [feat_height, feat_width])

            num_anchors = all_anchors.shape[0]
            with tf.compat.v1.variable_scope('model_RPN') as scope:
                prediction_dict = RPN(feat_map, num_ref_anchors)

            # Get the tensors from the dict
            rpn_cls_prob = prediction_dict['rpn_cls_prob']
            rpn_bbox_pred = prediction_dict['rpn_bbox_pred']

            proposal_prediction = RPNProposal(rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape, nms_threshold)

            pred_dict_final['all_anchors'] = tf.cast(all_anchors, tf.float32)
            prediction_dict['proposals'] = proposal_prediction['proposals']
            prediction_dict['scores'] = proposal_prediction['scores']

            pred_dict_final['rpn_prediction'] = prediction_dict
            scores = pred_dict_final['rpn_prediction']['scores']
            proposals = pred_dict_final['rpn_prediction']['proposals']

            pred_masks_watershed = tf.cast(
                marker_watershed(scores, proposals, pred_masks, min_score=min_score), dtype=tf.float32
            )

            # start point for testing, and end point for graph

            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())

            num_batches_test = len(images)

            saver = tf.compat.v1.train.Saver()

            masks1 = []

            # Restore the per-image normalization model from the trained network
            saver.restore(sess, self._network_path_whole_norm)
            sess.run(tf.compat.v1.local_variables_initializer())
            for j in tqdm(range(0, num_batches_test)):
                # whole image normalization
                batch_data = images[j]
                batch_data_shape = batch_data.shape
                image_normalized_wn = whole_image_norm(batch_data)
                image_normalized_wn = np.reshape(image_normalized_wn, [1, batch_data_shape[0], batch_data_shape[1], 1])

                masks = sess.run(pred_masks, feed_dict={train_initial: image_normalized_wn})

                # First pass, get the coarse masks, and normalize the image on masks
                masks1.append(masks)

            # Restore the foreground normalization model from the trained network
            saver.restore(sess, self._network_path_foreground)
            # saver.restore(sess,'./Network/fg_norm_weights_fluorescent/'+str(30)+'.ckpt')
            sess.run(tf.compat.v1.local_variables_initializer())
            for j in tqdm(range(0, num_batches_test)):
                batch_data = images[j]
                batch_data_shape = batch_data.shape
                images = np.reshape(batch_data, [batch_data_shape[0], batch_data_shape[1]])

                # Final pass, foreground normalization to get final masks
                image_normalized_fg = foreground_norm(images, masks1[j])
                image_normalized_fg = np.reshape(image_normalized_fg, [1, batch_data_shape[0], batch_data_shape[1], 1])

                # If adding watershed, we save the watershed masks separately
                if watershed:
                    masks = sess.run(pred_masks_watershed, feed_dict={train_initial: image_normalized_fg})

                else:
                    masks = sess.run(pred_masks, feed_dict={train_initial: image_normalized_fg})

            sess.close()

        if rescale_ratio != 1.0:
            masks = transform.rescale(masks, 1 / rescale_ratio)

        return masks

    def load_network(self, path: str = None):
        """
        Load an existing network. Loads the default network if ``path == None``

        Parameters
        ----------
        path : str
            path to the dir containing network files

        Returns
        -------
        None

        """
        if path is None:
            path = config.DEFAULT_NETWORK_DIR

        self.network_path = path
        self._network_path_whole_norm = os.path.join(path, 'whole_norm.ckpt')
        self._network_path_foreground = os.path.join(path, 'foreground.ckpt')

        wn_files = glob(self._network_path_whole_norm + '*')
        fg_files = glob(self._network_path_foreground + '*')

        if (len(wn_files) < 3) or (len(fg_files) < 3):
            raise FileNotFoundError(
                "Invalid network directory.\n"
                "The specified network directory does not contain all the required network files"
            )

    def __repr__(self):
        s = \
            f"network_path: {self.network_path}" \

        return s
