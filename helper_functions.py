# Import main packages
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from typing import Callable
import os
from datetime import datetime
from skimage.morphology import *
from skimage.color import rgb2hsv
import cv2

def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    # ------------------
    data_red = img[:,:,0]
    data_green = img[:,:,1]
    data_blue = img[:,:,2]
    # ------------------
    
    return data_red, data_green, data_blue

# Plot color space distribution 
def plot_colors_histo(
    img: np.ndarray,
    func: Callable,
    labels: list[str],
):
    """
    Plot the original image (top) as well as the channel's color distributions (bottom).

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    func: Callable
        A callable function that extracts D channels from the input image
    labels: list of str
        List of D labels indicating the name of the channel
    """

    # Extract colors
    channels = func(img=img)
    C2 = len(channels)
    M, N, C1 = img.shape
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, C2)

    # Use random seed to downsample image colors (increase run speed - 10%)
    mask = np.random.RandomState(seed=0).rand(M, N) < 0.1
    
    # Plot base image
    ax = fig.add_subplot(gs[:2, :])
    ax.imshow(img)
    # Remove axis
    ax.axis('off')
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2])

    # Plot channel distributions
    ax1.scatter(channels[0][mask].flatten(), channels[1][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title("{} vs {}".format(labels[0], labels[1]))
    ax1.grid()
    ax2.scatter(channels[0][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[2])
    ax2.set_title("{} vs {}".format(labels[0], labels[2]))
    ax2.grid()
    ax3.scatter(channels[1][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax3.set_xlabel(labels[1])
    ax3.set_ylabel(labels[2])
    ax3.set_title("{} vs {}".format(labels[1], labels[2]))
    ax3.grid()
        
    plt.tight_layout()


def apply_rgb_threshold(img, rgb):
    """
    Apply threshold to RGB input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract RGB channels
    data_red, data_green, data_blue = extract_rgb_channels(img=img)
    
    # ------------------
    red_T = rgb[0]
    green_T = rgb[1]
    blue_T = rgb[2]
    
    # Threshold data
    red_th = data_red < red_T
    green_th = data_green < green_T
    blue_th = data_blue < blue_T
    
    # Reconstruct image
    img_th = np.logical_and(red_th, np.logical_and(green_th, blue_th))
    # ------------------

    return  img_th

# Plot color space distribution 
def plot_thresholded_image(
    img: np.ndarray,
    func: Callable,
    title: str,
    rgb: np.ndarray,
):
    """
    Plot the original image and its thresholded version

    Args
    ----
    img: np.ndarray (M, N, 3)
        Input image of shape MxNx3.
    func: Callable
        Thresholded image.
    title: str
        Title of the plot
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[1].imshow(func(img,rgb))
    [a.axis('off') for a in axes]
    plt.suptitle(title)
    plt.tight_layout()

def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # ------------------
    hsv_img = rgb2hsv(img)
    data_h = hsv_img[:, :, 0]
    data_s = hsv_img[:, :, 1]
    data_v = hsv_img[:, :, 2]
    # ------------------
    
    return data_h, data_s, data_v

def apply_hsv_threshold(img, hsv):
    """
    Apply threshold to the input image in hsv colorspace.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)
    
    # ------------------
    # Set thresholds for each component
    h_T = hsv[0]
    s_T = hsv[1]
    v_T = hsv[2]
    
    h_th = data_h < h_T
    s_th = data_s > s_T
    v_th = data_v < v_T
    # Reconstruct image

    img_th = np.logical_and(h_th, np.logical_and(s_th, v_th))
    # ------------------
    
    return  img_th

def project_apply_hsv_threshold(img, hs_limits, show = False):
    """
    Apply threshold to the input image in HSV color space.

    Args:
        img (np.ndarray): Input image of shape (M, N, 3) with RGB color space.
        hsv_limits (tuple): Tuple of tuples defining the min and max HSV thresholds:
                            ((h_min, h_max), (s_min, s_max), (v_min, v_max))
        show (bool): Whether to display the thresholded image.

    Returns:
        np.ndarray: Thresholded binary image to 0 or 255.
    """
    # Extract HSV channels
    h, s, _ = extract_hsv_channels(img)
    
    # Apply thresholds for each channel
    h_min, h_max = hs_limits[0]
    s_min, s_max = hs_limits[1]
    # v_min, v_max = hs_limits[2]

    h_th = (h > h_min) & (h < h_max)
    s_th = (s > s_min) & (s < s_max)
    # v_th = (v > v_min) & (v < v_max)

    # Reconstruct image by combining the thresholded channels
    img_th = np.logical_and(h_th, s_th)

    img_th_uint8 = np.uint8(img_th * 255)  # Convert boolean to 0 or 255

    if show == True:
        plt.imshow(img_th_uint8)
        plt.axis('off')
        plt.show()

    return img_th_uint8


def project_resize_and_blur_image(img_path, scale_percent, blur_kernel_size):
    """
    Load an image, resize it, and apply Gaussian blur.

    Args:
        img_path (str): Path to the image file.
        scale_percent (int): Percentage by which to scale the image (e.g., 50 for 50%).
        blur_kernel_size (tuple): Size of the kernel used for blurring (e.g., (3,3)).

    Returns:
        np.ndarray: The resized and blurred image.
    """
    # Load image
    img = np.array(Image.open(img_path))
    
    # Calculate new dimensions
    new_width = int(img.shape[1] * scale_percent / 100)
    new_height = int(img.shape[0] * scale_percent / 100)
    
    # Resize image
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Blur image
    img_blurred = cv2.blur(img_resized, blur_kernel_size)
    
    return img_blurred

def project_detect_and_annotate_circles(original_img, processed_img, scale_percent, param1=30, show = False):
    """
    Detects circles in a processed image, scales the detections back to the original image size,
    annotates them, and displays the annotated image.

    Args:
        original_img (np.ndarray): The original image.
        processed_img (np.ndarray): Pre-processed image for circle detection (e.g., thresholded).
        scale_percent (int): The percentage scale used when resizing the original image.
        param1 (int): Parameter for the internal Canny edge detector in HoughCircles.
        show (bool): Whether to display the annotated image.

    Displays:
        The resized original image with annotated detections.

    Returns:
        Detected circles
    """
    original_height, original_width = original_img.shape[:2]
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)

    # Convert the processed image to grayscale if it is not already
    if len(processed_img.shape) == 3:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    # Calculate scale factors
    scale_x = original_width / new_width
    scale_y = original_height / new_height

    # Detect circles in the processed image
    detected_circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                                        param1=param1, param2=int(param1/2), minRadius=17, maxRadius=40)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        # print(f"Number of circles detected: {len(detected_circles[0])}")
        # for i, (x_resized, y_resized, r_resized) in enumerate(detected_circles[0, :]):
        #     x_original = int(x_resized * scale_x)
        #     y_original = int(y_resized * scale_y)
        #     r_original = int(r_resized * scale_x)  # assuming uniform scaling
        #     print(f"Circle {i+1}: Center at ({x_original}, {y_original}), Radius: {r_original}")

        # Visualize results on the resized image
        img_annotated = original_img.copy()
        img_annotated = cv2.resize(img_annotated, (new_width, new_height))
        for i, (x, y, r) in enumerate(detected_circles[0, :]):
            cv2.circle(img_annotated, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img_annotated, (x, y), 1, (0, 0, 255), 3)
            label = f"C{i+1}-r:{int(r*scale_x)}"
            cv2.putText(img_annotated, label, (x - 50, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if show:
            plt.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
            plt.axis('on')
            plt.grid(False)
            plt.show()
        
        return detected_circles
    else:
        print("No circles detected.")
        return 0
    
def project_extract_circles_with_transparency(img_path, detected_circles, scale_factor, desired_radius=400, save = False, save_dir='images', show=False):
    """
    Extract circles from an image based on detected circles and save them with transparency.

    Args:
        img_path (str): Path to the image file.
        detected_circles (np.ndarray): Detected circles from the image.
        scale_factor (int): Percentage by which the image was scaled.
        desired_radius (int): Desired radius of the extracted circles.
        save (bool): Whether to save the extracted circles.
        save_dir (str): Directory to save the extracted circles.
        show (bool): Whether to display the extracted circles.
    
    Returns:
        List of extracted circles.
    """

    img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_original is None:
        print("Failed to load image:", img_path)
        return []
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scale_x = scale_factor
    scale_y = scale_factor
    circle_images = []

    # Ensure detected_circles is a proper array-like structure with expected dimensions
    if not isinstance(detected_circles, np.ndarray) or len(detected_circles.shape) != 3:
        print("No circles detected or invalid circle data for image:", img_path)
        return []

    for i, (x_resized, y_resized, r_resized) in enumerate(detected_circles[0,:]):
        x_original = int(x_resized * scale_x)
        y_original = int(y_resized * scale_y)

        # Define the mask size
        mask_size = desired_radius * 2
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)

        top_left_x = x_original - desired_radius
        top_left_y = y_original - desired_radius
        bottom_right_x = x_original + desired_radius
        bottom_right_y = y_original + desired_radius

        extracted_circle = img_original[max(top_left_y, 0):min(bottom_right_y, img_original.shape[0]),
                                        max(top_left_x, 0):min(bottom_right_x, img_original.shape[1])]

        if extracted_circle.size == 0:
            print("Extracted circle is empty for image:", img_path)
            continue

        # Ensure the mask is the correct size
        mask = mask[:extracted_circle.shape[0], :extracted_circle.shape[1]]

        # Create the circle mask for the alpha channel
        cv2.circle(mask, (mask.shape[1]//2, mask.shape[0]//2), desired_radius, 255, -1)

        # Apply the mask
        masked_circle = cv2.bitwise_and(extracted_circle, extracted_circle, mask=mask)

        if masked_circle.size == 0:
            print("Masked circle is empty for image:", img_path)
            continue

        circle_images.append(masked_circle)

        if save :
            # Save the masked circle to the designated directory
            save_path = os.path.join(save_dir, f"circle_{i}_{os.path.basename(img_path)}")
            if not cv2.imwrite(save_path, cv2.cvtColor(masked_circle, cv2.COLOR_RGB2BGR)):
                print("Failed to save image:", save_path)

    if show:
      # Display all masked circles in a grid layout
      num_circles = len(circle_images)
      cols = 5  # Number of columns for the grid layout
      rows = (num_circles // cols) + (1 if num_circles % cols != 0 else 0)  # Calculate rows needed

      fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

      for idx, circle_img in enumerate(circle_images):
          row = idx // cols
          col = idx % cols
          ax = axes[row, col] if rows > 1 else axes[col]
          ax.imshow(cv2.cvtColor(circle_img, cv2.COLOR_BGR2RGB))
          ax.set_title(f"Circle {idx + 1}")
          ax.axis('off')

      # Remove unused subplots
      for idx in range(num_circles, rows * cols):
          row = idx // cols
          col = idx % cols
          ax = axes[row, col] if rows > 1 else axes[col]
          ax.axis('off')

      plt.tight_layout()
      plt.show()
    
    return circle_images

def project_classify_picture(image, threshold1=50, threshold2=30, threshold3=45):
    """
    Classify the picture based on the standard deviation of the hue channel.

    Args:
    image (np.array): Image to classify in BGR color format.
    threshold1 (int): Threshold for the 'hand' category.
    threshold2 (int): Lower threshold for the 'noisy' category.
    threshold3 (int): Upper threshold for the 'noisy' category.

    Returns:
    str: Category of the picture ('hand', 'noisy', 'neutral').
    """
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the standard deviation of the hue channel
    hue_std = np.std(hsv[:, :, 0].ravel())

    # print(f"Hue Standard Deviation: {hue_std}")

    # Classify based on the hue standard deviation
    if hue_std > threshold1 :
        return "hand"
    elif hue_std > threshold2 and hue_std <= threshold3:
        return "noisy"
    else:
        return "neutral"

# # Example usage
# # Load an image
# image_path = './train/2. noisy_bg/L1010325.JPG'
# image_path = './train/2. noisy_bg/L1010370.JPG'
# # image_path = './train/6. hand_outliers/L1010521.JPG'
# # image_path = './ref/ref_chf.JPG'
# image_path = './train/3. hand/L1010373.JPG'
# # image_path = './train/6. hand_outliers/L1010521.JPG'

# # SUPER IMPORTANT, OTHERWISE THE IMAGE WILL BE READ IN BGR FORMAT !!!
# image = np.array(Image.open(image_path))
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# # Check if the image was loaded properly
# if image is not None:
#     classification = project_classify_picture(image)
#     print(f"Classification: {classification}")
# else:
#     print("Error: Image not found or could not be loaded.")

def project_show_image_with_predictions(original_img, detected_circles, scale_percent, predictions, show=False):
    """
    Annotates an original image with detections framed by squares, including predictions and scores listed above each detection.

    Args:
        original_img (np.ndarray): The original image.
        detected_circles (np.ndarray): Array of detected circles, each formatted as (x, y, radius).
        scale_percent (int): The percentage scale used when resizing the original image for detection.
        predictions (list): List of predicted classes for each detected circle.
        scores (list): List of scores corresponding to each prediction.
        show (bool): Whether to display the annotated image.

    Displays:
        The resized original image with annotated detections and classification results.
    """
    original_height, original_width = original_img.shape[:2]
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)

    # Prepare the image for annotation
    img_annotated = original_img.copy()
    img_annotated = cv2.resize(img_annotated, (new_width, new_height))

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for i, (x, y, r) in enumerate(detected_circles[0, :]):
            
            # Define top left and bottom right points for square
            top_left = (x - r, y - r)
            bottom_right = (x + r, y + r)

            # Draw the square
            cv2.rectangle(img_annotated, top_left, bottom_right, (0, 255, 0), 2)

            # Prepare the label text with prediction and score
            label_prediction = f"{predictions[i]}"
            # Position text above the square for prediction
            cv2.putText(img_annotated, label_prediction, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if show:
            plt.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Coins Detection")
            plt.show()

    else:
        print("No circles detected.")





#___________________________________________________________________________________
#___________________________________________________________________________________

# morphology
def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for closing

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)

    # ------------------
    structel = disk(disk_size)
    img_closing = closing(img_th, structel)
    # ------------------

    return img_closing


def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N)
        Image after opening operation
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)

    # ------------------
    structel = disk(disk_size)
    img_opening = opening(img_th, structel)
    # ------------------

    return img_opening

def plot_images(
    imgs: np.ndarray,
    sizes: list[int],
    title: str,
):
    """
    Plot multiple images. The title of each subplot is defined by the disk_size elements.

    Args
    ----
    imgs: np.ndarray (D, M, N)
        List of D images of size MxN.
    disk_sizes: list of int
        List of D int that are the size of the disk used for the operation
    title:
        The overall title of the figure
    """
    D = len(imgs)
    ncols = int(np.ceil(D/2))
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 4*ncols))
    
    # Remove axis
    axes = axes.ravel()
    [ax.axis('off') for ax in axes]
    
    for i in range(D):
        axes[i].imshow(imgs[i])
        axes[i].set_title("Size: {}".format(sizes[i]))
    
    plt.suptitle(title)
    plt.tight_layout()       

def remove_holes(img_th, size):
    """
    Remove holes from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of holes

    Return
    ------
    img_holes: np.ndarray (M, N)
        Image after remove holes operation
    """

    # Define default value for input image
    img_holes = np.zeros_like(img_th)

    # ------------------
    img_holes = remove_small_holes(img_th, size)
    # ------------------

    return img_holes


def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of objects

    Return
    ------
    img_obj: np.ndarray (M, N)
        Image after remove small objects operation
    """

    # Define default value for input image
    img_obj = np.zeros_like(img_th)

    # ------------------
    img_obj = remove_small_objects(img_th, size)
    # ------------------

    return img_obj


#___________________________________________________________________________________
#___________________________________________________________________________________
# region growing 
#8 connectivity
def decision(pixel):
    # Threshold comparison with array of pixels
    hsv_pixel = rgb2hsv(np.array([pixel]))[0]
    h_T = 0.9
    s_T = 0.3
    v_T = 0.7

    h_th = hsv_pixel[0] < h_T
    s_th = hsv_pixel[1] > s_T
    v_th = hsv_pixel[2] < v_T
    
    if h_th and v_th and s_th:
        return True
    return False

def region_growing(seeds, img, n_max=10, **kwargs):
    M, N, C = img.shape
    rg = np.zeros((M, N), dtype=bool)

    # Initialize rg for seeds
    for seed in seeds:
        x, y = seed
        rg[x, y] = True
    
    iterations = 0
    pixels_to_check = set(seeds)
    # print(pixels_to_check)
    
    # Directions for 8-connectivity: includes diagonals
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while pixels_to_check and iterations < n_max:
        new_pixels = set()
        for x, y in pixels_to_check:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Check if the neighbor is within image bounds and not already in the region
                if 0 <= nx < M and 0 <= ny < N and not rg[nx, ny]:
                    # Use the decision function to check if the pixel should be added
                    if decision(img[nx, ny]):
                        rg[nx, ny] = True
                        new_pixels.add((nx, ny))
        pixels_to_check = new_pixels
        iterations += 1
    
    return rg

def plot_region_growing(
    seeds: list[tuple],
    img: np.ndarray,
    func: Callable,
    iters: list[int],
):
    """
    Plot the region growing results based on seeds, function and iterations
    
    Args
    ----
    seeds: list of tuple
        List of seed points
    img: np.ndarray (M, N, C)
        RGB image of size M, N, C
    func: callable
        Region growing function
    iters: list of ints
        Number of iteration to plot
    """

    # Define plot size
    n = len(iters) + 1
    n_rows = np.ceil(n // 2).astype(int)
    _, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
    axes = axes.ravel()
    [a.axis('off') for a in axes]   

    # Reference image
    axes[0].imshow(img)
    axes[0].set_title("Input image")

    # Plot all iterations
    for i, it in enumerate(iters):
        t1 = datetime.now()
        img_rg = region_growing(seeds=seeds, img=img, n_max=iters[i])
        # Compute time difference in seconds
        t2 = datetime.now()
        seconds = (t2 - t1).total_seconds()
        axes[i+1].imshow(img_rg)
        axes[i+1].set_title("RG {} iter in {:.2f} seconds".format(iters[i], seconds))
                            
    plt.tight_layout()
    