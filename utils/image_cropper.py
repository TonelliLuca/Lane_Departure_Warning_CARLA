import numpy as np
from PIL import Image

def process_image(video_output, define_crop_size):
    # Convert raw image data to a numpy array
    image_to_analyze = video_output[:, :, :3]

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(image_to_analyze)

    # Define the size and position for cropping
    crop_size = min(define_crop_size, pil_image.width, pil_image.height)
    width, height = pil_image.size
    left = (width - crop_size) // 2
    top = height - crop_size
    right = left + crop_size
    bottom = height

    cropped_image = pil_image.crop((left, top, right, bottom))

    # Convert the cropped image back to a numpy array for analysis
    image_to_analyze = np.array(cropped_image).transpose(2, 0, 1)  # HWC to CHW

    return image_to_analyze