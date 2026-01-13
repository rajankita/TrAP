import torch.nn as nn
import numpy as np
import random

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomPatchDropOld(nn.Module):
    def __init__(self, patch_size, drop_percentage, prob=1.0):
        """
        Initializes the RandomPatchDrop augmentation.

        Args:
            patch_size (int): The size of the square patch to drop (e.g., 16 for 16x16).
            drop_percentage (float): The percentage of total possible patches to randomly drop,
                                     expressed as a float between 0.0 and 1.0 (e.g., 0.1 for 10%).
            prob (float): The probability of applying this augmentation to an image.
                          Defaults to 0.5.
        """
        super().__init__()
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

    def forward(self, img):
        """
        Applies the random patch dropping to the input image.

        Args:
            img (torch.Tensor): The input image tensor (C, H, W).

        Returns:
            torch.Tensor: The augmented image tensor.
        """
        if random.random() < self.prob:
            C, H, W = img.shape

            # Calculate the number of patches along height and width
            num_patches_h = H // self.patch_h
            num_patches_w = W // self.patch_w

            if num_patches_h == 0 or num_patches_w == 0:
                print(f"Warning: Image dimensions ({H}x{W}) are smaller than patch size ({self.patch_h}x{self.patch_w}). No patches can be dropped.")
                return img # Return original image if patches cannot be formed

            total_patches = num_patches_h * num_patches_w

            # Calculate the number of patches to drop based on the percentage
            # Use max(1, ...) to ensure at least one patch is dropped if percentage > 0
            # and min(total_patches, ...) to not exceed total patches.
            # Rounding to the nearest integer.
            self.drop_count = int(round(self.drop_percentage * total_patches))
            self.drop_count = max(0, min(self.drop_count, total_patches)) # Ensure within bounds [0, total_patches]

            if self.drop_count == 0:
                return img # No patches to drop after calculation
            
            # Generate random indices for patches to drop
            # We use `np.random.choice` for non-replacement selection of indices
            dropped_patch_indices = np.random.choice(total_patches, self.drop_count, replace=False)

            # Create a copy of the image to modify
            augmented_img = img.clone()

            for idx in dropped_patch_indices:
                # Convert 1D patch index to 2D (row, col)
                patch_row = idx // num_patches_w
                patch_col = idx % num_patches_w

                # Calculate pixel coordinates for the top-left corner of the patch
                start_h = patch_row * self.patch_h
                start_w = patch_col * self.patch_w

                # Set the pixels within the patch to zero
                end_h = start_h + self.patch_h
                end_w = start_w + self.patch_w

                augmented_img[:, start_h:end_h, start_w:end_w] = 0.0 # Set to black

            return augmented_img
        else:
            return img # Return original image if not applying augmentation

    def __repr__(self):
        return (f"{self.__class__.__name__}(patch_size=({self.patch_h}, {self.patch_w}), "
                f"drop_percentage={self.drop_percentage:.2f}, prob={self.prob})")
