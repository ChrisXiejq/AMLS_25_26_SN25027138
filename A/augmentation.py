import scipy.ndimage as ndimage
import numpy as np

def augment_image(img):
    """
    Apply data augmentation to a single image.
    Args:
        img (numpy.ndarray): Input image array.
    Returns:
        numpy.ndarray: Augmented image array.
    """
    # random rotation (−10° ~ +10°)
    angle = np.random.uniform(-10, 10)
    img = ndimage.rotate(img, angle, reshape=False, mode="nearest")

    # random horizontal flip
    if np.random.rand() < 0.5:
        img = np.fliplr(img)

    # add gaussian noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.03, img.shape)
        img = img + noise

    return np.clip(img, 0, 1)