import torch
import cv2
import numpy as np
from collections import deque
from utils.utils import \
    time_synchronized, select_device, non_max_suppression, split_for_trace_model,\
    driving_area_mask, lane_line_mask,\
    letterbox

device = None
model = None
half = False
lane_status_history = deque(maxlen=5)  # Add this line

def initializeYOLOPModel(weights='data/weights/yolopv2.pt', imgsz=320):
    global device, model, half
    # Load the model once
    device = select_device('0')
    print(device)
    model = torch.jit.load(weights).to(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16
    model.eval()

    # Run inference once to warm up the model
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

def filter_nested_boxes(boxes, iou_threshold=0.8):
    """Remove nested bounding boxes, keeping only the largest."""
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Coordinates of the intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Compute intersection area
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Compute areas of both boxes
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Compute union area
        union_area = box1_area + box2_area - intersection_area

        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0

    # Sort boxes by area in descending order
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    filtered_boxes = []
    for i, box in enumerate(boxes):
        keep = True
        for j in range(i):
            if calculate_iou(box, boxes[j]) > iou_threshold:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    return filtered_boxes




def analyzeImage(image):
    """Improved detect function to filter horizontal lines and handle central alignment."""
    crossing = False
    try:
        with torch.no_grad():
            img0 = image.transpose(1, 2, 0)  # CHW to HWC
            img = letterbox(img0, new_shape=img0.shape[:2])[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # Normalize to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            outputs = model(img)
            if len(outputs) == 3:
                [pred, anchor_grid], seg, ll = outputs
            else:
                # Handle different model output formats
                return img0  # Return original image if format is unexpected
            t2 = time_synchronized()

            # Apply NMS
            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)

            # CUDA synchronization
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Process segmentation masks
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)

            # Clean up GPU memory
            del seg, ll, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Resize segmentation masks to match input image size
            if img0.shape[:2] != (0, 0):
                da_seg_mask_resized = cv2.resize(da_seg_mask, img0.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                ll_seg_mask_resized = cv2.resize(ll_seg_mask, img0.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            else:
                return np.zeros((720, 1280, 3), dtype=np.uint8)  # Safe default

            # Overlay segmentation masks on the original image
            mask = np.zeros_like(img0, dtype=np.uint8)
            mask[da_seg_mask_resized == 1] = (0, 255, 0)  # Green for drivable area
            mask[ll_seg_mask_resized == 1] = (0, 0, 255)  # Red for lane lines
            combined = cv2.addWeighted(img0, 0.7, mask, 0.3, 0)

            # Process the lane line mask for red line detection
            red_lane_mask = cv2.inRange(ll_seg_mask_resized, 1, 255)

            # Clean up noise with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            red_lane_mask = cv2.morphologyEx(red_lane_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours of red lines
            contours, _ = cv2.findContours(red_lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter bounding boxes based on size and orientation
            MIN_BOX_WIDTH = 40   # Minimum box width
            MIN_BOX_HEIGHT = 40  # Minimum box height
            ORIENTATION_THRESHOLD = 4.0  # Threshold for detecting horizontal boxes

            red_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > MIN_BOX_WIDTH and h > MIN_BOX_HEIGHT:
                    aspect_ratio = w / h  # Calculate width-to-height ratio
                    if aspect_ratio < ORIENTATION_THRESHOLD:  # Avoid horizontal boxes
                        red_boxes.append((x, y, w, h))

            # Remove nested boxes, keeping only the largest
            red_boxes = filter_nested_boxes(red_boxes)

            for (x, y, w, h) in red_boxes:
                cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

            if red_boxes:
                # Calculate lane center
                leftmost_red = min([x for x, y, w, h in red_boxes])
                rightmost_red = max([x + w for x, y, w, h in red_boxes])
                lane_center_x = (leftmost_red + rightmost_red) // 2
                img_center_x = combined.shape[1] // 2

                # Distance from center
                center_distance = lane_center_x - img_center_x

                # Thresholds for different states
                CENTERED_THRESHOLD = 30  # More tolerant centered threshold
                CROSSING_THRESHOLD = 53  # Higher threshold for definite crossing

                # Add state tracking
                if len(red_boxes) == 1:  # Single box scenario
                    alignment_status = "CROSSING: SBX"
                    crossing = True
                    #print(alignment_status)
                else:
                    if abs(center_distance) < CENTERED_THRESHOLD:
                        alignment_status = "CENTERED"
                        #print(alignment_status)
                    elif abs(center_distance) < CROSSING_THRESHOLD:
                        if center_distance > 0:
                            alignment_status = "CROSSING: SL"
                            #print(alignment_status)
                        else:
                            alignment_status = "CROSSING: SR"
                            #print(alignment_status)
                        crossing = True
                    else:
                        if center_distance > 0:

                            alignment_status = "CROSSING: LEFT"
                            #print("Car is CROSSING to the LEFT lane")
                        else:
                            alignment_status = "CROSSING: RIGHT"
                            #print("Car is CROSSING to the RIGHT lane")
                        crossing = True

                # Draw reference lines and text
                cv2.line(combined, (lane_center_x, 0), (lane_center_x, combined.shape[0]), (0, 0, 255), 2)
                cv2.line(combined, (img_center_x, 0), (img_center_x, combined.shape[0]), (255, 0, 0), 2)
                cv2.putText(combined, alignment_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

            return combined, crossing

    except Exception as e:
        print(f"Error in analyzeImage: {str(e)}")
        # Return a safe default image
        return np.zeros((720, 1280, 3), dtype=np.uint8)