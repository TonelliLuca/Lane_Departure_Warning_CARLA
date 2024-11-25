import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# Load the YOLOP model from torch hub
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

# Load and preprocess the image
img_path = "rgb_camera.png"  # Replace with your image path
img = Image.open(img_path).convert("RGB")

# Define the transformations (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize image to 640x640 as expected by YOLOP
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Apply the transformations to the image
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Inference
det_out, da_seg_out, ll_seg_out = model(img_tensor)

# Print the lane-line segmentation output
print(ll_seg_out)

lane_lines = ll_seg_out[0].detach().cpu().numpy()  # Move to CPU and detach from the computation graph

# Normalize to range [0, 1] for visualization (if necessary)
lane_lines = np.clip(lane_lines, 0, 1)

# Visualize the segmentation mask
plt.imshow(lane_lines[0], cmap='gray')  # Display the first channel (if it's a multi-channel output)
plt.colorbar()
plt.title("Lane Line Segmentation Output")
plt.show()