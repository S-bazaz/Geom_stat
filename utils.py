import cupy as cp
import collections

def complement_stain_matrix(w):

    stain0 = w[:, 0]
    stain1 = w[:, 1]
    stain2 = cp.cross(stain0, stain1)
    # Normalize new vector to have unit norm
    return cp.array([stain0, stain1, stain2 / cp.linalg.norm(stain2)]).T

def magnitude(m):
    """Get the magnitude of each column vector in a matrix"""
    return cp.sqrt((m ** 2).sum(0))


def normalize(m):
    """Normalize each column vector in a matrix"""
    return m / magnitude(m)

def convert_image_to_matrix(im):
    """Convert an image (MxNx3 array) to a column matrix of pixels
    (3x(M*N)).  It will pass through a 2D array unchanged.

    """
    if im.ndim == 2:
        return im

    return im.reshape((-1, im.shape[-1])).T

def convert_matrix_to_image(m, shape):
    """Convert a column matrix of pixels to a 3D image given by shape.
    The number of channels is taken from m, not shape.  If shape has
    length 2, the matrix is returned unchanged.  This is the inverse
    of convert_image_to_matrix:

    im == convert_matrix_to_image(convert_image_to_matrix(im),
    im.shape)

    """
    if len(shape) == 2:
        return m

    return m.T.reshape(shape[:-1] + (m.shape[0],))

def rgb_to_sda(im_rgb, I_0, allow_negatives=False):
    """Transform input RGB image or matrix `im_rgb` into SDA (stain
    darkness) space for color deconvolution.

    Parameters
    ----------
    im_rgb : array_like
        Image (MxNx3) or matrix (3xN) of pixels

    I_0 : float or array_like
        Background intensity, either per-channel or for all channels

    allow_negatives : bool
        If False, would-be negative values in the output are clipped to 0

    Returns
    -------
    im_sda : array_like
        Shaped like `im_rgb`, with output values 0..255 where `im_rgb` >= 1
    """
    is_matrix = im_rgb.ndim == 2
    if is_matrix:
        im_rgb = im_rgb.T

    if I_0 is None:  # rgb_to_od compatibility
        im_rgb = im_rgb.astype(float) + 1
        I_0 = 256

    im_rgb = cp.maximum(im_rgb, 1e-10)

    im_sda = -cp.log(im_rgb / (1. * I_0)) * 255 / cp.log(I_0)
    if not allow_negatives:
        im_sda = cp.maximum(im_sda, 0)
    return im_sda.T if is_matrix else im_sda

def sda_to_rgb(im_sda, I_0):
    """Transform input SDA image or matrix `im_sda` into RGB space.  This
    is the inverse of `rgb_to_sda` with respect to the first parameter

    Parameters
    ----------
    im_sda : array_like
        Image (MxNx3) or matrix (3xN) of pixels

    I_0 : float or array_like
        Background intensity, either per-channel or for all channels
    """
    is_matrix = im_sda.ndim == 2
    if is_matrix:
        im_sda = im_sda.T

    od = I_0 is None
    if od:  # od_to_rgb compatibility
        I_0 = 256

    im_rgb = I_0 ** (1 - im_sda / 255.)
    return (im_rgb.T if is_matrix else im_rgb) - od

def color_deconvolution(im_rgb, w, I_0=None):
    # complement stain matrix if needed
    w = cp.array(w)
    if w.shape[1] < 3:
        wc = cp.zeros((w.shape[0], 3))
        wc[:, :w.shape[1]] = w
        w = wc

    if cp.linalg.norm(w[:, 2]) <= 1e-16:
        wc = complement_stain_matrix(w)
    else:
        wc = w

    # normalize stains to unit-norm
    wc = normalize(wc)

    # invert stain matrix
    Q = cp.linalg.pinv(wc)

    # transform 3D input image to 2D RGB matrix format
    m = convert_image_to_matrix(im_rgb)[:3]

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = rgb_to_sda(m, I_0)
    sda_deconv = cp.dot(Q, sda_fwd)
    sda_inv = sda_to_rgb(sda_deconv,
                                          255 if I_0 is not None else None)

    # reshape output
    StainsFloat = convert_matrix_to_image(sda_inv, im_rgb.shape)

    # transform type
    Stains = StainsFloat.clip(0, 255).astype(cp.uint8)

    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, wc)

    return Output