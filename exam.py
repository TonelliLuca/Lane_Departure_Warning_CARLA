import carla, time, pygame, math, random, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages,\
    letterbox

from threading import Thread

def cleanup():
    """Ensure proper cleanup of resources upon termination."""
    try:
        if camera is not None:
            camera.destroy()
            print("Camera destroyed")
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle destroyed")
        pygame.quit()
        cv2.destroyAllWindows()
        print("Resources cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")

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
imgsz = 640
device = select_device('0')
model = torch.jit.load(weights).to(device)
half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()  # to FP16
model.eval()

# Run inference once to warm up the model
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

def detect(image):
    with torch.no_grad():
        random_bgr = np.transpose(image, (1, 2, 0))
        img0 = cv2.resize(random_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
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
        show_seg_result(img0, (da_seg_mask, ll_seg_mask), is_demo=True)
        return img0

def spawn_camera(attach_to=None, transform=carla.Transform(carla.Location(x=1.2, z=2), carla.Rotation(pitch=-10)), width=640, height=640):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    return camera

vehicle = spawn_vehicle()
camera = spawn_camera(attach_to=vehicle)

video_output = np.zeros((640, 640, 4), dtype=np.uint8)
video_output_seg = np.zeros((1280, 720, 3), dtype=np.uint8)

def camera_callback(image):
    global video_output
    
    video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    image_to_analyze = video_output[:, :, :3]
    image_to_analyze = np.transpose(image_to_analyze, (2, 0, 1))

    pil_image = Image.fromarray(image_to_analyze.transpose(1, 2, 0))  # Convert to PIL image
 
    # Define the size of the square to crop (e.g., 200x200)
    square_size = 450
 
    # Calculate the crop box for the square in the bottom center
    left = (pil_image.width - square_size) // 2  # Center horizontally
    upper = pil_image.height - square_size  # Align to the bottom
    right = left + square_size  # Width of the square
    lower = upper + square_size  # Height of the squareq
 
    # Crop the image
    crop_box = (left, upper, right, lower)
    cropped_image = pil_image.crop(crop_box)
    cropped_image.save('tone3.png')
    cropped_not_transposed = np.array(cropped_image)
    cropped_image_np = np.transpose(cropped_image, (2, 0, 1))
    print(cropped_not_transposed.shape)

    Thread(target=run_inference, args=(cropped_image_np,)).start()
    # Start a new thread for the detect function

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