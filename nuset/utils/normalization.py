import numpy as np
from skimage import __version__ as skimage_version
from skimage import morphology

from typing import List

BASE_CONNECTIVITY: int = 2
SKIMAGE_VERSION: List[int] = list(map(int, skimage_version.split(".")))

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def whole_image_norm(image):
    return (image - np.mean(image)) / np.std(image)

def foreground_norm(image, mask):
    foreground = image * mask
                                
    # Normalization based on the foreground average and std, not the global average
    foreground_flat = np.reshape(foreground, -1)
    foreground_nonzero = np.asarray(remove_values_from_list(foreground_flat, 0))

    im_median = np.median(foreground_nonzero)
    im_std = np.std(foreground_nonzero)

    # normalize
    im_foreground_norm = (image - im_median) / (im_std + 1e-5)
    
    return im_foreground_norm

def clean_image(image):
   """ perform small region removal and fill holes """
   image = image.astype(np.bool)
   im_label = morphology.label(image, connectivity=1)
   num_cells = np.max(im_label)
   mean_area = np.sum(image).astype(np.float32)/num_cells
   
   # Backwards compatibility with skimage =< 0.15
   if SKIMAGE_VERSION[0] == 0 and SKIMAGE_VERSION[1] >= 16:
      # if a region or hole < 1/5 of a normal cell, remove it
      image = morphology.remove_small_objects(image, area_threshold=mean_area/5, 
                                              connectivity=BASE_CONNECTIVITY)
      image = morphology.remove_small_holes(image, area_threshold=mean_area/5, 
                                            connectivity=BASE_CONNECTIVITY)
   else:
      image = morphology.remove_small_objects(image, min_size=mean_area/5, 
                                              connectivity=BASE_CONNECTIVITY)
      image = morphology.remove_small_holes(image, min_size=mean_area/5, 
                                            connectivity=BASE_CONNECTIVITY)
   return image.astype(np.uint8)
