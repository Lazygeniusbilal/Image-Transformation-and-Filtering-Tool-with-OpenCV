# Import dependencies
import cv2
import numpy as np

# Grayscale conversion function
def grayscale_conversion(image_path):
    """
    This function converts an image to grayscale.

    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        numpy.ndarray: Image in grayscale, or None if an error occurs.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        return image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Brightness and contrast adjustment function
def adjust_brightness_contrast(image_path, brightness=0, contrast=0):
    """
    Adjusts the brightness and contrast of an image.
    
    Args:
        image_path (str): Path to the input image file.
        brightness (float): Factor to adjust brightness.
        contrast (float): Factor to adjust contrast.

    Returns:
        numpy.ndarray: Image with adjusted brightness and contrast, or None if an error occurs.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        adjusted_image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
        return adjusted_image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Image sharpening using filter2D
def image_sharpening_filter2D(image_path):
    """
    Sharpens the input image using a custom filter.

    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        numpy.ndarray: Sharpened image, or None if an error occurs.
    """
    try:
        filter2D_kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        sharpened_image = cv2.filter2D(image, -1, kernel=filter2D_kernel)
        return sharpened_image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Image sharpening using Laplacian
def image_sharpening_laplacian(image_path):
    """
    Sharpens the input image using the Laplacian method.

    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        numpy.ndarray: Sharpened image, or None if an error occurs.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian_image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Image noise reduction using median blur
def medianBlur_method(image_path):
    """
    Removes noise from the image using median blur.

    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        numpy.ndarray: Image with reduced noise, or None if an error occurs.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        denoised_image = cv2.medianBlur(image, ksize=11)
        return denoised_image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Image noise reduction using Gaussian blur
def GaussianBlur_method(image_path):
    """
    Removes noise from the image using Gaussian blur.

    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        numpy.ndarray: Image with reduced noise, or None if an error occurs.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
        return denoised_image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Image resizing function
def image_resize(image_path, width, height):
    """
    Resizes the input image to the specified width and height.

    Args:
        image_path (str): Path to the input image file.
        width (int): Desired width of the resized image.
        height (int): Desired height of the resized image.
    
    Returns:
        numpy.ndarray: The resized image, or None if an error occurs.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        resized_image = cv2.resize(image, (width, height))
        return resized_image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Image cropping function
def image_cropping(image_path, min_width, max_width, min_height, max_height):
    """
    Crops the input image to the specified dimensions.

    Args:
        image_path (str): Path to the input image file.
        min_width (int): The starting width (x-coordinate) for cropping.
        max_width (int): The ending width (x-coordinate) for cropping.
        min_height (int): The starting height (y-coordinate) for cropping.
        max_height (int): The ending height (y-coordinate) for cropping.

    Returns:
        numpy.ndarray: The cropped image, or None if an error occurs.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the provided path.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image from {image_path}")
        cropped_image = image[min_height:max_height, min_width:max_width]
        return cropped_image
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
