import carla, pygame, cv2
import numpy as np
q
from utils.carla import spawn_vehicle, spawn_camera
from utils.image_cropper import process_image
from utils.YOLOPModel import initializeYOLOPModel, analyzeImage

from threading import Thread

pygame.init()
screen = pygame.display.set_mode((800, 600))
# Try to connect to the CARLA server
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    #town04 - highway
    #town06 long highways with lane exit
    client.load_world('Town05')
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
    video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


    image_to_analyze = process_image(video_output, define_crop_size)
    run_analysis(image_to_analyze)
    # Start a new thread for inference
    #Thread(target=run_analysis, args=(image_to_analyze,)).start()

def run_analysis(image_to_analyze):
    """
    Runs inference on the processed image.
    """
    global video_output_seg
    video_output_seg = analyzeImage(image_to_analyze)
    

camera.listen(lambda image: camera_callback(image))

vehicle.set_autopilot(False)

cv2.namedWindow('Original RGB feed', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('RGB analyzed output', cv2.WINDOW_AUTOSIZE)

running = True

try:
    # Initialize controller support
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Controller connected: {joystick.get_name()}")

    # Control state variables
    throttle_val = 0.0
    brake_val = 0.0
    steer_val = 0.0

    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Keyboard controls
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    throttle_val = 1.0
                    brake_val = 0.0
                elif event.key == pygame.K_s:
                    brake_val = 1.0
                    throttle_val = 0.0
                elif event.key == pygame.K_a:
                    steer_val = -1.0
                elif event.key == pygame.K_d:
                    steer_val = 1.0
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    throttle_val = 0.0
                elif event.key == pygame.K_s:
                    brake_val = 0.0
                elif event.key == pygame.K_a and steer_val < 0:
                    steer_val = 0.0
                elif event.key == pygame.K_d and steer_val > 0:
                    steer_val = 0.0

        if joystick:
            # Get controller name to handle different mappings
            controller_name = joystick.get_name().lower()

            # Handle trigger inputs (varies between controller drivers)
            # Xbox One triggers are typically on axes 2 (LT) and 5 (RT)
            # Some drivers map them to -1 (released) to 1 (pressed)
            # Others map them to 0 (released) to 1 (pressed)

            # Left trigger (brake)
            left_trigger = joystick.get_axis(2)
            # Convert from [-1, 1] to [0, 1] if needed
            if left_trigger < -0.5:  # If resting position is -1
                brake_val = (left_trigger + 1) / 2
            else:  # If resting position is 0
                brake_val = max(0, left_trigger)

            # Right trigger (throttle)
            right_trigger = joystick.get_axis(5)
            # Convert from [-1, 1] to [0, 1] if needed
            if right_trigger < -0.5:  # If resting position is -1
                throttle_val = (right_trigger + 1) / 2
            else:  # If resting position is 0
                throttle_val = max(0, right_trigger)

            # Left stick for steering (horizontal axis)
            raw_steer = joystick.get_axis(0)

            # Apply deadzone and smoothing to steering
            deadzone = 0.1
            if abs(raw_steer) < deadzone:
                steer_val = 0.0
            else:
                # Apply progressive steering for better control
                # Rescale from deadzone to 1.0
                steer_direction = 1.0 if raw_steer > 0 else -1.0
                steer_amount = (abs(raw_steer) - deadzone) / (1.0 - deadzone)
                # Make steering more precise by applying a curve
                steer_val = steer_direction * (steer_amount ** 1.5)

            # Optional: Check for handbrake (B button on Xbox controller)
            handbrake = 1.0 if joystick.get_button(1) else 0.0

        # Apply combined control values
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle_val,
            brake=brake_val,
            steer=steer_val
        ))

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