import scipy.ndimage as ndimage
import numpy as np

def augment_image(img):
    """Apply data augmentation suitable for medical images."""
    # 1. random rotation (−15° ~ +15°)
    angle = np.random.uniform(-15, 15)
    img = ndimage.rotate(img, angle, reshape=False, mode="nearest")

    # 2. random horizontal flip
    if np.random.rand() < 0.5:
        img = np.fliplr(img)

    # 3. add gaussian noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.05, img.shape)
        img = img + noise

    return np.clip(img, 0, 1)