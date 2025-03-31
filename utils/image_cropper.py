import numpy as np
import cv2

def process_image(video_output, define_crop_size):
    # Convert raw image data to a numpy array
    image_to_analyze = video_output[:, :, :3]

    # Get image dimensions
    height, width = image_to_analyze.shape[:2]

    # Define source points (trapezoid in the original image)
    # Adjust these points to capture the road area properly
    src_points = np.float32([
        [width * 0.35, height * 0.55],  # Top-left
        [width * 0.65, height * 0.55],  # Top-right
        [width * 0.85, height * 0.95],  # Bottom-right
        [width * 0.15, height * 0.95]   # Bottom-left
    ])

    # Define destination points (rectangle in the transformed image)
    dst_points = np.float32([
        [0, 0],                      # Top-left
        [define_crop_size, 0],       # Top-right
        [define_crop_size, define_crop_size],  # Bottom-right
        [0, define_crop_size]        # Bottom-left
    ])

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image_to_analyze, M, (define_crop_size, define_crop_size))

    # Convert to CHW format for model input
    warped_chw = warped.transpose(2, 0, 1)  # HWC to CHW

    return warped_chw

