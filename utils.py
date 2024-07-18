import json
import math
import os
import struct
from glob import glob

import cv2
import Imath
import numpy as np
import OpenEXR
from PIL import Image

def load_masks_dict(json_path):
    """
    Load a JSON file containing mask data and return it as a Python dictionary.

    Args:
        json_path (str): The file path to the JSON file.

    Returns:
        dict: A dictionary containing the mask data from the JSON file.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        json.JSONDecodeError: If the JSON file is not properly formatted.
    """
    try:
        with open(json_path, 'r') as json_file:
            masks_dict = json.load(json_file)
        return masks_dict
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: The file {json_path} is not a valid JSON file.")
        print(f"JSON decode error: {str(e)}")
        raise

def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array

    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)

    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr

def calculate_camera_parameters(masks_dict=None, image_width=None, image_height=None):
    """
    Calculate camera parameters from the provided dictionary or direct image dimensions.

    Args:
    masks_dict (dict, optional): Dictionary containing camera and image information.
    image_width (int, optional): Width of the image in pixels.
    image_height (int, optional): Height of the image in pixels.

    Returns:
    tuple:
        fx (int): The focal length along x-axis in pixels of camera used to capture image.
        fy (int): The focal length along y-axis in pixels of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    """
    if masks_dict is None and (image_width is None or image_height is None):
        raise ValueError("Either masks_dict or both image_width and image_height must be provided")

    if masks_dict is not None:
        try:
            camera_dict = masks_dict['camera']
            fov_x_rad = camera_dict['field_of_view']['x_axis_rads']
            fov_y_rad = camera_dict['field_of_view']['y_axis_rads']
        except KeyError as e:
            raise KeyError(f"Missing key in masks_dict: {e}")

        if image_width is None or image_height is None:
            try:
                image_dict = masks_dict['image']
                image_width = image_dict['width_px']
                image_height = image_dict['height_px']
            except KeyError as e:
                raise KeyError(f"Missing key in masks_dict: {e}")
    else:
        # If masks_dict is not provided, we need to ensure both image dimensions are given
        if image_width is None or image_height is None:
            raise ValueError("Both image_width and image_height must be provided when masks_dict is not given")
        
        # In this case, we don't have fov values, so we can't calculate fx and fy
        return None, None, int(image_width / 2), int(image_height / 2)

    # Calculate focal lengths using the radian values
    fx = int((image_width / 2) / math.tan(fov_x_rad / 2))
    fy = int((image_height / 2) / math.tan(fov_y_rad / 2))

    # Calculate principal point
    cx = int(image_width / 2)
    cy = int(image_height / 2)

    return fx, fy, cx, cy

def load_image_cv2(image_path):
    """
    Load an image from the specified path using OpenCV and convert it to RGB format.

    Args:
    image_path (str): The path to the image file.

    Returns:
    numpy.ndarray: The image in RGB format as a NumPy array with dtype uint8.
    """
    # Read the image
    img = cv2.imread(image_path)

    # OpenCV reads images in BGR format, so we need to convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensure the image is in uint8 format
    img_rgb = img_rgb.astype(np.uint8)

    return img_rgb


def _get_point_cloud(color_image, depth_image, fx, fy, cx, cy):
    """Creates point cloud from rgb images and depth image

    Args:
        color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
        depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
        fx (int): The focal len along x-axis in pixels of camera used to capture image.
        fy (int): The focal len along y-axis in pixels of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    Returns:
        numpy.ndarray: camera_points - The XYZ location of each pixel. Shape: (num of pixels, 3)
        numpy.ndarray: color_points - The RGB color of each pixel. Shape: (num of pixels, 3)
    """
    # camera instrinsic parameters
    # camera_intrinsics  = [[fx 0  cx],
    #                       [0  fy cy],
    #                       [0  0  1]]
    camera_intrinsics = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, image_width - 1, image_width),
                                   np.linspace(0, image_height - 1, image_height))
    camera_points_x = np.multiply(pixel_x - camera_intrinsics[0, 2], (depth_image / camera_intrinsics[0, 0]))
    camera_points_y = np.multiply(pixel_y - camera_intrinsics[1, 2], (depth_image / camera_intrinsics[1, 1]))
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x, camera_points_y, camera_points_z]).transpose(1, 2, 0).reshape(-1, 3)

    color_points = color_image.reshape(-1, 3)

    return camera_points, color_points


def write_point_cloud(filename, color_image, depth_image, fx, fy, cx, cy):
    """Creates and Writes a .ply point cloud file using RGB and Depth images.

    Args:
        filename (str): The path to the file which should be written. It should end with extension '.ply'
        color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
        depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
        fx (int): The focal len along x-axis in pixels of camera used to capture image.
        fy (int): The focal len along y-axis in pixels of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    """
    xyz_points, rgb_points = _get_point_cloud(color_image, depth_image, fx, fy, cx, cy)

    # Write header of .ply file
    with open(filename, 'wb') as fid:
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(
                bytearray(
                    struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(), rgb_points[i, 2].tostring())))
