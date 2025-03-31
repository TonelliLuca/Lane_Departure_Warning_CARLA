import numpy as np
import cv2

# Global flag to track if visualization has been saved
_trapezoid_saved = False

def process_image(video_output, define_crop_size):
    global _trapezoid_saved

    # Convert raw image data to a numpy array
    image_to_analyze = video_output[:, :, :3]

    # Get image dimensions
    height, width = image_to_analyze.shape[:2]

    # Define source points (trapezoid in the original image)
    src_points = np.float32([
        [width * 0.35, height * 0.55],  # Top-left
        [width * 0.65, height * 0.55],  # Top-right
        [width * 0.85, height * 0.95],  # Bottom-right
        [width * 0.15, height * 0.95]   # Bottom-left
    ])

    # Save visualization of trapezoid only once
    if not _trapezoid_saved:
        # Create a copy of the image to draw on
        visualization = image_to_analyze.copy()

        # Convert source points to integer format
        src_points_int = np.array(src_points, dtype=np.int32)

        # Draw the trapezoid
        cv2.polylines(visualization, [src_points_int], True, (0, 255, 0), 2)

        # Add points markers
        for i, point in enumerate(src_points_int):
            cv2.circle(visualization, tuple(point), 5, (255, 0, 0), -1)
            cv2.putText(visualization, f"P{i}", (point[0]+5, point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the visualization
        cv2.imwrite('./log/untracked/trapezoid_visualization.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print("Trapezoid visualization saved to 'trapezoid_visualization.png'")

        _trapezoid_saved = True

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