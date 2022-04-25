# Smartcow - https://www.smartcow.ai/en/

import cv2
import numpy as np

from glob import glob

# CowGenerator class
# Generates a randomly colored cow using one of the samples available in NellieJay/resources/cows
class CowGenerator:
    def __init__(self):
        # Paths to the cow images
        self.base_cows_paths = sorted(glob("NellieJay/resources/cows/cow_*.png"))
        # Paths to the cow masks
        self.base_masks_paths = sorted(glob("NellieJay/resources/cows/mask_*.png"))

        # No of cows found
        self.no_of_cows = len(self.base_cows_paths)
        
        # Read the cows and mask images
        self.base_cows = [cv2.imread(tmp_pth, cv2.IMREAD_UNCHANGED) for tmp_pth in self.base_cows_paths]
        self.base_masks = [cv2.imread(tmp_pth) for tmp_pth in self.base_masks_paths]

    # Generate a random cow with a random color
    def get_cow(self):
        # Generate a random index
        tmp_rand_idx = np.random.randint(self.no_of_cows)
        
        # Get the cow template corresponding to the random index
        tmp_cow = self.base_cows[tmp_rand_idx].copy()
        tmp_mask = self.base_masks[tmp_rand_idx]

        # Generate a random color
        tmp_color = [np.random.randint(0, 256), np.random.randint(0, 256), 255]
        tmp_color = cv2.cvtColor( np.uint8([[tmp_color]] ), cv2.COLOR_HSV2BGR)[0][0]
        
        # Replace the cows color with the generated color
        tmp_cow[(tmp_mask == 255).all(-1), :3] = tmp_color

        # Resize the cow to a standard size
        ratio = 100/tmp_cow.shape[1]
        tmp_cow = cv2.resize(tmp_cow, (100, int(ratio*tmp_cow.shape[0])), interpolation=cv2.INTER_AREA)

        return tmp_cow

        
    
