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



def plot_2_imgs(img_chf, img_eur, chf_title, eur_title, color_map):
    """
    Create 1 figure with 2 images.

    Args
    ----
    img 1: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    title 1: float
        Title of the 1st image
    img 2: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    title 2: float
        Title of the 1st image

    Return
    ------
    """
    # Create a figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display the images
    ax[0].imshow(img_chf, cmap = color_map)
    ax[0].set_title(chf_title)  # Set title for the first image
    ax[1].imshow(img_eur, cmap =  color_map)
    ax[1].set_title(eur_title)  # Set title for the second image

    # Remove the axis ticks
    ax[0].axis('off')
    ax[1].axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

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
    ax2.scatter(channels[0][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[2])
    ax2.set_title("{} vs {}".format(labels[0], labels[2]))
    ax3.scatter(channels[1][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax3.set_xlabel(labels[1])
    ax3.set_ylabel(labels[2])
    ax3.set_title("{} vs {}".format(labels[1], labels[2]))
        
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
    
    # Threshold data
    h_th = data_h < h_T
    s_th = data_s > s_T
    v_th = data_v < v_T
    # Reconstruct image

    img_th = np.logical_and(h_th, np.logical_and(s_th, v_th))
    # ------------------
    
    return  img_th
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
    