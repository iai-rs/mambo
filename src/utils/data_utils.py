import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu


def add_one_if_even(number):
    if number % 2 == 0:
        return number + 1
    else:
        return number

def create_kernel(img, factor):
    return add_one_if_even(img.shape[0] // factor), add_one_if_even(img.shape[1] // factor)

def binarize(img):
    b_img = img.astype(np.float32)
    b_img = 255 * (b_img - np.min(b_img)) / (np.max(b_img) - np.min(b_img))
    b_img = b_img.astype(np.uint8)

    blured = cv2.GaussianBlur(b_img, create_kernel(b_img, 50), 0)

    otsu_tr = threshold_otsu(blured) * 0.175
    mask = np.where(blured >= otsu_tr, 1, 0).astype(np.uint8)

    return mask


def dilate(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, create_kernel(mask, 200))
    dilated_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return dilated_mask


def keep_largest_blob(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask with only the largest contour
    largest_blob_mask = np.zeros_like(mask)
    cv2.drawContours(largest_blob_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    return largest_blob_mask


def get_breast_mask(image):
    mask = binarize(image)
    mask = dilate(mask)
    return keep_largest_blob(mask)


def keep_only_breast(image):
    mask = get_breast_mask(image)
    return image * mask, mask


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=5)
    return clahe.apply(image)


def crop_borders(image):
    l = 0.01
    r = 0.01
    u = 0.04
    d = 0.04

    n_rows, n_cols = image.shape

    l_crop = int(n_cols * l)
    r_crop = int(n_cols * (1 - r))
    u_crop = int(n_rows * u)
    d_crop = int(n_rows * (1 - d))

    cropped_img = image[u_crop:d_crop, l_crop:r_crop]

    return cropped_img, (l_crop, n_cols - r_crop, u_crop, n_rows - d_crop)


def should_flip(image):
    x_center = image.shape[1] // 2
    col_sum = image.sum(axis=0)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    return left_sum < right_sum


def pad(image):
    n_rows, n_cols = image.shape
    if n_rows == n_cols:
        return image

    target_shape = (np.maximum(n_rows, n_cols),)*2

    padded_img = np.zeros(shape=target_shape).astype(image.dtype)
    padded_img[:n_rows, :n_cols] = image

    return padded_img


def negate_if_should(image):
    hist, bins = np.histogram(image.ravel(), bins=10, range=[image.min(), image.max()])

    return image if hist[0] > hist[-1] else np.max(image) - image


def preprocess_scan_with_mask(image, mass_mask):
    image = negate_if_should(image)
    image, borders = crop_borders(image)
    mass_mask, _ = crop_borders(mass_mask)
    image, breast_mask = keep_only_breast(image)
    flip = should_flip(image)

    if flip:
        image = np.fliplr(image)
        breast_mask = np.fliplr(breast_mask)
        mass_mask = np.fliplr(mass_mask)

    #image = apply_clahe(image)
    image = image * breast_mask  # clahe manages to change black to slight gray
    shape_before_padding = image.shape
    image = pad(image)
    mass_mask = pad(mass_mask)

    spatial_changes = (borders, flip, shape_before_padding, image.shape)

    return image, mass_mask, spatial_changes


def preprocess_scan(image):
    image, _, spatial_changes = preprocess_scan_with_mask(image, image)
    return image, spatial_changes


def reverse_spatial_changes(image, spatial_changes):
    borders, flip, shape_before_padding, shape = spatial_changes
    l, r, u, d = borders
    image = cv2.resize(src=image, dsize=shape, interpolation=cv2.INTER_NEAREST)
    image = image[: shape_before_padding[0], : shape_before_padding[1]]
    if flip:
        image = np.fliplr(image)
    image = np.pad(image, ((u, d), (l, r)))
    return image


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


# map original bbox coordinates to processed image
def map_coordinates(orig_coords, spatial_changes, orig_shape):
    """
    Maps the original coordinates of bounding box to new coordinates after spatial transformations of the image (after preprocess_scan)

    Parameters:
    - orig_coords (tuple): Original coordinates of the bounding box in the format (xmin, ymin, xmax, ymax).
    - spatial_changes (tuple): A tuple containing information about transformations, structured as:
        - borders (tuple): Padding applied to the image in the format (left, right, top, bottom).
        - flip (bool): Indicates whether the image was flipped horizontally (True) or not (False).
        - shape_before_padding (tuple): Image shape (height, width) before padding was applied.
        - shape (tuple): Final image shape (height, width) after all transformations.
    - orig_shape (tuple): The original image size, after loaded from dicom file, in the format (height, width).

    Returns:
    - new_coords (tuple): The transformed coordinates after applying the specified changes.
    """
    xmin, ymin, xmax, ymax = orig_coords
    borders, flip, shape_before_padding, shape = spatial_changes
    width, height = orig_shape[1], orig_shape[0]
    
    # Remove borders
    l, r, u, d = borders
    width = width - l - r
    height = height - u - d
    xmin = xmin - l
    ymin = ymin - u
    xmax = xmax - l
    ymax = ymax - u
    
    # Apply horizontal flipping if needed
    if flip:
        xmin_new = width - xmax
        xmax_new = width - xmin
    else:
        xmin_new = xmin
        xmax_new = xmax

    return (xmin_new, ymin, xmax_new, ymax)
    

def shift_image(image, x_center, y_center):
    """
    Shifts an image so that a given point (x_center, y_center) is moved to the center of the image.
    
    Parameters:
    image (numpy.ndarray): Input grayscale image represented as a 2D NumPy array.
    x_center (int): X-coordinate of the point to be moved to the image center.
    y_center (int): Y-coordinate of the point to be moved to the image center.
    
    Returns:
    numpy.ndarray: A new image with the specified shift applied.
    """

    H, W = image.shape  # Get image dimensions (Height and Width)
    new_image = np.zeros_like(image) + image.min()  # Create a new blank (black) image of the same size

    # Compute shift distances relative to the image center
    dx = W // 2 - x_center
    dy = H // 2 - y_center

    # Determine valid source image boundaries (ensuring we do not access out-of-bounds pixels)
    x_start_src = max(0, -dx)
    x_end_src = min(W, W - dx)
    y_start_src = max(0, -dy)
    y_end_src = min(H, H - dy)

    # Determine valid destination image boundaries
    x_start_dst = max(0, dx)
    x_end_dst = min(W, W + dx)
    y_start_dst = max(0, dy)
    y_end_dst = min(H, H + dy)

    # Copy only the visible portion of the image to the new shifted position
    new_image[x_start_dst:x_end_dst, y_start_dst:y_end_dst] = image[x_start_src:x_end_src, y_start_src:y_end_src]

    return new_image
