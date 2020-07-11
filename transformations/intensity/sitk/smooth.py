
import SimpleITK as sitk

#import skimage.filters

def gaussian(image, sigma):
    """
    Multidimensional gaussian smoothing.
    :param image: sitk image
    :param sigma: list of sigmas per dimension, or scalar for equal sigma in each dimension
    :return: smoothed sitk image
    """
    if sigma[0]==sigma[1]:
        return sitk.SmoothingRecursiveGaussian(image, sigma[0])

    return 0;

    #return skimage.filters.gaussian(image, sigma)
