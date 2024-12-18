import carla, pygame, cv2
import numpy as np

from utils.carla import spawn_vehicle, spawn_camera
from utils.image_cropper import process_image
from utils.YOLOPModel import initializeYOLOPModel, analyzeImage

from threading import Thread

# Try to connect to the CARLA server
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()
except Exception as e:
    print(f"Failed to connect to CARLA server: {e}")
    exit(1)

# Initialize the YOLOP model
initializeYOLOPModel()

# Spawn the vehicle and camera in the CARLA world
vehicle = spawn_vehicle(world)
camera = spawn_camera(world, attach_to=vehicle)

# Initialize video output arrays
video_output = np.zeros((640, 640, 4), dtype=np.uint8)
video_output_seg = np.zeros((720, 1280, 3), dtype=np.uint8)

# Define the crop size for image processing
define_crop_size = 320

def camera_callback(image):
    """
    Callback function for the camera. Processes the image and starts a new thread for inference.
    """
    global video_output

    image_to_analyze = process_image(image, define_crop_size)

    # Start a new thread for inference
    Thread(target=run_analysis, args=(image_to_analyze,)).start()

def run_analysis(image_to_analyze):
    """
    Runs inference on the processed image.
    """
    global video_output_seg
    video_output_seg = analyzeImage(image_to_analyze)
    

camera.listen(lambda image: camera_callback(image))

vehicle.set_autopilot(False)

cv2.namedWindow('Original RGB feed', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('RGB Camera output', cv2.WINDOW_AUTOSIZE)

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
        cv2.imshow('Original RGB feed', video_output)
        cv2.imshow('RGB analyzed output', video_output_seg)
finally:
    cv2.destroyAllWindows()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()