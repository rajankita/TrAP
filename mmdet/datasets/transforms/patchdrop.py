import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union

from mmcv.transforms import BaseTransform, TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes # This might require mmdet installed
from mmcv.transforms.utils import cache_randomness


@TRANSFORMS.register_module() # Uncomment if you have MMDetection installed
class RandomPatchDrop(BaseTransform):
    """Randomly drops rectangular patches from an image.

    This version DOES NOT update annotations (bboxes, masks, seg_map).
    It only modifies the 'img' key in the results dictionary.

    Required Keys:
    - img

    Modified Keys:
    - img

    Args:
        patch_size (int): The size of the square patch to drop (e.g., 16 for 16x16).
        drop_percentage (float): The percentage of total possible patches to randomly drop,
                                 expressed as a float between 0.0 and 1.0 (e.g., 0.1 for 10%).
        prob (float): The probability of applying this augmentation to an image.
                      Defaults to 0.5.
        fill_value (int or float or tuple): The value(s) to fill the dropped regions in the image.
                                            If float, applies to all channels. If tuple, must be 3 elements.
                                            Defaults to 0 (black).
    """

    def __init__(
        self,
        patch_size: int,
        drop_percentage: float,
        prob: float = 0.5,
        fill_value: Union[int, float, Tuple] = 0, # Changed default to 0 for black
    ) -> None:
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch size must be a positive integer.")
        self.patch_h = patch_size
        self.patch_w = patch_size

        if not (0.0 <= drop_percentage <= 1.0):
            raise ValueError("drop_percentage must be between 0.0 and 1.0.")
        self.drop_percentage = drop_percentage

        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be between 0.0 and 1.0.")
        self.prob = prob

        self.fill_value = fill_value

    @cache_randomness
    def _get_patches_to_drop(self, img_shape: Tuple[int, int, int]) -> list:
        """Calculates the coordinates of patches to be dropped."""
        H, W = img_shape[:2] # Get H, W, ignoring channels if present

        num_patches_h = H // self.patch_h
        num_patches_w = W // self.patch_w

        if num_patches_h == 0 or num_patches_w == 0:
            return [] # No patches can be formed

        total_patches = num_patches_h * num_patches_w

        drop_count = int(round(self.drop_percentage * total_patches))
        drop_count = max(0, min(drop_count, total_patches))

        if drop_count == 0:
            return [] # No patches to drop after calculation

        dropped_patch_indices = np.random.choice(total_patches, drop_count, replace=False)

        patches_coords = []
        for idx in dropped_patch_indices:
            patch_row = idx // num_patches_w
            patch_col = idx % num_patches_w

            start_h = patch_row * self.patch_h
            start_w = patch_col * self.patch_w
            end_h = start_h + self.patch_h
            end_w = start_w + self.patch_h # Note: changed to patch_h for square patch end_w
            patches_coords.append([start_w, start_h, end_w, end_h]) # MMDetection typically uses (x1, y1, x2, y2)

        return patches_coords

    def transform(self, results: Dict) -> Dict:
        """Main transform function to apply patch dropping to the image."""
        if random.random() < self.prob:
            img = results['img']
            
            # Ensure img is a numpy array and has correct dimensions
            if not isinstance(img, np.ndarray):
                raise TypeError(f"Image should be a numpy array, but got {type(img)}")
            if img.ndim < 2 or img.ndim > 3:
                raise ValueError(f"Image should be 2D (grayscale) or 3D (color), but got {img.ndim}D")

            # _get_patches_to_drop expects H, W, (C)
            patches_coords = self._get_patches_to_drop(img.shape)
            
            if not patches_coords: # If no patches to drop, return original results
                return results

            h, w = img.shape[:2] # img is typically HxWxC or HxW for grayscale

            # Determine fill_value for all channels
            if img.ndim == 3: # Color image
                _fill_value = self.fill_value if isinstance(self.fill_value, tuple) else (self.fill_value,) * img.shape[2]
            else: # Grayscale or single channel
                _fill_value = self.fill_value if isinstance(self.fill_value, (int, float)) else self.fill_value[0] # Take first if tuple

            for patch in patches_coords:
                x1, y1, x2, y2 = patch
                # Clip coordinates to ensure they are within image bounds
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                # Apply fill value
                if img.ndim == 3:
                    img[y1:y2, x1:x2, :] = _fill_value
                else: # Grayscale
                    img[y1:y2, x1:x2] = _fill_value

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(patch_size=({self.patch_h}, {self.patch_w}), '
        repr_str += f'drop_percentage={self.drop_percentage:.2f}, '
        repr_str += f'prob={self.prob:.2f}, '
        repr_str += f'fill_value={self.fill_value})'
        return repr_str