from PIL import Image


def grayscale(image):
    """ Takes an image and converts to grayscale. """
    return image.convert('L')
