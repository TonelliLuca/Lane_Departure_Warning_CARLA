import carla, time, pygame, math, random, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from utils.utils import \
    time_synchronized,select_device,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,show_seg_result,\
    letterbox

from threading import Thread

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
spectator = world.get_spectator()


def move_spectator_to(transform, distance=5.0, x=0, y=0, z=4, yaw=0, pitch=-30, roll=0):
    back_location = transform.location - transform.get_forward_vector() * distance
    
    back_location.x += x
    back_location.y += y
    back_location.z += z
    transform.rotation.yaw += yaw
    transform.rotation.pitch = pitch
    transform.rotation.roll = roll
    
    spectator_transform = carla.Transform(back_location, transform.rotation)
    
    spectator.set_transform(spectator_transform)

def spawn_vehicle(vehicle_index=0, spawn_index=0, pattern='vehicle.*'):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter(pattern)[vehicle_index]
    spawn_point = world.get_map().get_spawn_points()[spawn_index]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle

def draw_on_screen(world, transform, content='O', color=carla.Color(0, 255, 0), life_time=20):
    world.debug.draw_string(transform.location, content, color=color, life_time=life_time)


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Load the model once
weights = 'data/weights/yolopv2.pt'
imgsz = 352
device = select_device('0')
model = torch.jit.load(weights).to(device)
half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()  # to FP16
model.eval()

# Run inference once to warm up the model
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

# Define the desired crop size
define_crop_size = 352

def detect(image):
    """Modified detect function to handle dynamic crop sizes."""
    with torch.no_grad():
        # Adjust input image dimensions and preprocessing
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
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # Apply NMS
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Resize segmentation masks to match input image size
        da_seg_mask_resized = cv2.resize(da_seg_mask, img0.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        ll_seg_mask_resized = cv2.resize(ll_seg_mask, img0.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

         # Overlay segmentation masks on the original image
        mask = np.zeros_like(img0, dtype=np.uint8)
        mask[da_seg_mask_resized == 1] = (0, 255, 0)  # Green for drivable area
        mask[ll_seg_mask_resized == 1] = (0, 0, 255)  # Red for lane lines
        
        # Merge segmentation mask with the original image
        combined = cv2.addWeighted(img0, 0.7, mask, 0.3, 0)

        # Process the lane line mask for red line detection
        red_lane_mask = cv2.inRange(ll_seg_mask_resized, 1, 255)

        # Clean up noise with morphological operationsq
        kernel = np.ones((5, 5), np.uint8)
        red_lane_mask = cv2.morphologyEx(red_lane_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of red lines
        contours, _ = cv2.findContours(red_lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes and alignment information
        red_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        for (x, y, w, h) in red_boxes:
            cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

        # Lane alignment analysis
        if red_boxes:
            leftmost_red = min([x for x, y, w, h in red_boxes])
            rightmost_red = max([x + w for x, y, w, h in red_boxes])
            lane_center_x = (leftmost_red + rightmost_red) // 2
            img_center_x = combined.shape[1] // 2

            # Alignment status
            if abs(lane_center_x - img_center_x) < 20:
                alignment_status = "Car is CENTERED in the lane"
                print("Car is CENTERED in the lane")
            elif lane_center_x > img_center_x:
                alignment_status = "Car is CROSSING to the LEFT lane"
                print("Car is CROSSING to the LEFT lane")
            else:
                alignment_status = "Car is CROSSING to the RIGHT lane"
                print("Car is CROSSING to the RIGHT lane")

            # Draw alignment lines and text
            cv2.line(combined, (lane_center_x, 0), (lane_center_x, combined.shape[0]), (0, 0, 255), 2)
            cv2.line(combined, (img_center_x, 0), (img_center_x, combined.shape[0]), (255, 0, 0), 2)
            cv2.putText(combined, alignment_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display segmentation mask for debugging
        cv2.imshow("Lane Line Mask", red_lane_mask)

        return combined



def spawn_camera(attach_to=None, transform=carla.Transform(carla.Location(x=1.2, z=2), carla.Rotation(pitch=-10)), width=640, height=640):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    return camera

vehicle = spawn_vehicle()
camera = spawn_camera(attach_to=vehicle)

video_output = np.zeros((640, 640, 4), dtype=np.uint8)
video_output_seg = np.zeros((720, 1280, 3), dtype=np.uint8)


def camera_callback(image):
    global video_output

    # Convert raw image data to a numpy array
    video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
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

    # Start a new thread for inference
    Thread(target=run_inference, args=(image_to_analyze,)).start()


def run_inference(image_to_analyze):
    global video_output_seg
    video_output_seg = detect(image_to_analyze)
    

camera.listen(lambda image: camera_callback(image))

vehicle.set_autopilot(False)

cv2.namedWindow('RGB analyzed output', cv2.WINDOW_AUTOSIZE)

running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    vehicle.apply_control(carla.VehicleControl(throttle=1.0))
                elif event.key == pygame.K_s:
                    vehicle.apply_control(carla.VehicleControl(brake=1.0))
                elif event.key == pygame.K_a:
                    vehicle.apply_control(carla.VehicleControl(steer=-1.0))
                elif event.key == pygame.K_d:
                    vehicle.apply_control(carla.VehicleControl(steer=1.0))
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w or event.key == pygame.K_s:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
                elif event.key == pygame.K_a or event.key == pygame.K_d:
                    vehicle.apply_control(carla.VehicleControl(steer=0.0))

        if cv2.waitKey(1) == ord('q'):
            running = False
            break
        
        cv2.imshow('RGB analyzed output', video_output_seg)
finally:
    cv2.destroyAllWindows()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()