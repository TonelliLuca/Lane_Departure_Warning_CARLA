import carla, pygame, cv2
import numpy as np
from datetime import datetime

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
    client.load_world('Town04')
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
top_view_output = np.zeros((480, 640, 4), dtype=np.uint8)  # For top-down camera

# Define the crop size for image processing
define_crop_size = 320

# Lane invasion detection variables
lane_invasion_detected = False  # CARLA detection
carla_lane_invasion_timestamp = datetime.now()
yolop_lane_invasion_detected = False  # Our YOLOP detection
yolop_lane_invasion_timestamp = datetime.now()
blink_state = False
blink_timer = 0
last_detection_frames = 0
detection_threshold = 3  # Require consecutive detections to reduce false positives

def spawn_lane_invasion_sensor(world, attach_to=None):
    """Spawn a lane invasion sensor attached to the vehicle"""
    lane_sensor_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    lane_sensor = world.spawn_actor(lane_sensor_bp, carla.Transform(), attach_to=attach_to)
    return lane_sensor

def lane_invasion_callback(event):
    """Callback for when the vehicle crosses lane markings according to Carla"""
    global lane_invasion_detected, carla_lane_invasion_timestamp
    lane_invasion_detected = True
    carla_lane_invasion_timestamp = datetime.now()
    lane_types = set(x.type for x in event.crossed_lane_markings)
    text = ['%r' % str(x).split()[-1] for x in lane_types]
    print('Carla detected lane crossing: %s' % ', '.join(text))

    # Cross-validate with YOLOP detection
    if yolop_lane_invasion_detected:
        print("Both systems detected lane crossing - HIGH CONFIDENCE")

def detect_lane_crossing(seg_output):
    """
    More accurate lane crossing detection with temporal consistency
    Returns True if crossing detected, False otherwise
    """
    global yolop_lane_invasion_detected, yolop_lane_invasion_timestamp
    global last_detection_frames, detection_threshold

    try:
        # Check if input is valid
        if seg_output is None or seg_output.size == 0:
            return False

        # Get the lane line mask from segmentation output
        if len(seg_output.shape) == 3:
            height, width = seg_output.shape[:2]

            if height <= 0 or width <= 0:
                return False

            # Extract regions where lane lines are detected (red in segmentation)
            lane_mask = np.zeros((height, width), dtype=np.uint8)
            red_pixels = (seg_output[:, :, 2] > 200) & (seg_output[:, :, 0] < 50) & (seg_output[:, :, 1] < 50)
            lane_mask[red_pixels] = 255

            # Define the center area where vehicle is
            bottom_half = height // 2
            center_width = width // 3

            # Make sure indices are valid
            center_start = max(0, (width - center_width) // 2)
            center_end = min(width, (width + center_width) // 2)

            # Get vehicle area safely
            if bottom_half < height and center_end > center_start:
                vehicle_area = lane_mask[bottom_half:, center_start:center_end]

                # Count lane pixels in vehicle area
                lane_pixels = cv2.countNonZero(vehicle_area)

                # Pixel-based detection with minimum threshold
                if lane_pixels > 150:  # Higher threshold to avoid false positives
                    last_detection_frames += 1
                else:
                    last_detection_frames = max(0, last_detection_frames - 1)

                # Only trigger after consecutive detections to avoid flickering
                if last_detection_frames >= detection_threshold:
                    if not yolop_lane_invasion_detected:
                        print("YOLOP detected lane crossing")
                        yolop_lane_invasion_detected = True
                        yolop_lane_invasion_timestamp = datetime.now()
                    return True

        # Reset detection after timeout
        if yolop_lane_invasion_detected and (datetime.now() - yolop_lane_invasion_timestamp).total_seconds() > 3:
            yolop_lane_invasion_detected = False
            last_detection_frames = 0

    except Exception as e:
        print(f"Error in lane crossing detection: {str(e)}")
        # Reset detection on error
        last_detection_frames = 0

    return yolop_lane_invasion_detected

def spawn_top_down_camera(world, attach_to=None):
    """Spawn a top-down camera to view car position relative to lanes"""
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')

    # Position the camera high above the car looking down
    camera_transform = carla.Transform(
        carla.Location(x=0, y=0, z=15),  # 15 meters above car
        carla.Rotation(pitch=-90)  # Looking straight down
    )

    top_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=attach_to)
    return top_camera

def top_camera_callback(image):
    """Callback for the top-down camera"""
    global top_view_output
    top_view_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # Add lane status indicators to the top view
    status = "LANE OK"
    color = (0, 255, 0)  # Green

    if yolop_lane_invasion_detected:
        status = "YOLOP: CROSSING"
        color = (0, 255, 255)  # Yellow

    if lane_invasion_detected:
        status = "CARLA: CROSSING" if not yolop_lane_invasion_detected else "BOTH: CROSSING"
        color = (0, 0, 255) if not yolop_lane_invasion_detected else (255, 0, 255)  # Red or Purple

    cv2.putText(top_view_output, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def draw_warning_triangle(image, blinking=False):
    """Draw a yellow triangle warning if blinking is True"""
    if not blinking:
        return image

    # Convert image to numpy array if needed and ensure it's BGR format
    if isinstance(image, np.ndarray) and image.shape[2] >= 3:
        h, w = image.shape[:2]
        # Triangle parameters
        size = min(h, w) // 8
        center_x = w - size - 20
        center_y = size + 20

        # Define the triangle vertices
        pts = np.array([
            [center_x, center_y - size],
            [center_x - size, center_y + size],
            [center_x + size, center_y + size]
        ], np.int32)

        # Create a copy to avoid modifying the original
        image_copy = image.copy()

        # Draw filled yellow triangle
        cv2.fillPoly(image_copy, [pts], (0, 255, 255))
        # Draw black border
        cv2.polylines(image_copy, [pts], True, (0, 0, 0), 2)

        # Add exclamation mark
        cv2.putText(image_copy, "!", (center_x - 5, center_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

        return image_copy

    return image

def camera_callback(image):
    """
    Callback function for the camera. Processes the image and starts a new thread for inference.
    """
    global video_output, video_output_seg, yolop_lane_invasion_detected
    try:
        # Copy image data safely
        video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Process the image and run inference
        image_to_analyze = process_image(video_output, define_crop_size)

        # Skip processing if image is invalid
        if image_to_analyze.size == 0 or np.max(image_to_analyze) == 0:
            print("Warning: Invalid image received, skipping analysis")
            return

        # Perform image analysis with error handling
        result = analyzeImage(image_to_analyze)

        # Update global only if valid result returned
        if result is not None and result.size > 0:
            video_output_seg = result

            # Check for lane crossing in the segmentation output
            detect_lane_crossing(video_output_seg)

    except Exception as e:
        print(f"Error in camera callback: {str(e)}")

running = True

try:
    # Set up camera with callback
    camera.listen(lambda image: camera_callback(image))

    # Create and set up top-down camera
    top_camera = spawn_top_down_camera(world, attach_to=vehicle)
    top_camera.listen(lambda image: top_camera_callback(image))

    # Create and attach the lane invasion sensor
    lane_invasion_sensor = spawn_lane_invasion_sensor(world, attach_to=vehicle)
    lane_invasion_sensor.listen(lambda event: lane_invasion_callback(event))

    # Set up display windows
    cv2.namedWindow('Original RGB feed', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RGB analyzed output', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Top-down View', cv2.WINDOW_AUTOSIZE)

    vehicle.set_autopilot(False)

    # Initialize joystick support
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

            # Handle trigger inputs
            left_trigger = joystick.get_axis(2)
            if left_trigger < -0.5:
                brake_val = (left_trigger + 1) / 2
            else:
                brake_val = max(0, left_trigger)

            right_trigger = joystick.get_axis(5)
            if right_trigger < -0.5:
                throttle_val = (right_trigger + 1) / 2
            else:
                throttle_val = max(0, right_trigger)

            # Left stick for steering
            raw_steer = joystick.get_axis(0)
            deadzone = 0.1
            if abs(raw_steer) < deadzone:
                steer_val = 0.0
            else:
                steer_direction = 1.0 if raw_steer > 0 else -1.0
                steer_amount = (abs(raw_steer) - deadzone) / (1.0 - deadzone)
                steer_val = steer_direction * (steer_amount ** 1.5)

            # Optional: Check for handbrake
            handbrake = 1.0 if joystick.get_button(1) else 0.0

        # Apply controls to vehicle
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle_val,
            brake=brake_val,
            steer=steer_val
        ))

        # Update blinking state (toggle every 0.5 seconds)
        blink_timer += 1
        if blink_timer >= 15:  # Assuming ~30 fps, toggle every 0.5 seconds
            blink_timer = 0
            blink_state = not blink_state

        # Reset Carla lane invasion after timeout (3 seconds)
        if lane_invasion_detected and (datetime.now() - carla_lane_invasion_timestamp).total_seconds() > 3:
            lane_invasion_detected = False

        # Create copies for display with warning indicators
        display_original = video_output.copy()
        display_analyzed = video_output_seg.copy()

        # Draw warning triangle if lane invasion is detected by either method
        if lane_invasion_detected or yolop_lane_invasion_detected:
            display_original = draw_warning_triangle(display_original, blink_state)
            display_analyzed = draw_warning_triangle(display_analyzed, blink_state)

        # Display images
        if display_original.shape[2] >= 3:  # Make sure it's a valid image
            cv2.imshow('Original RGB feed', display_original)

        if display_analyzed.shape[2] >= 3:
            cv2.imshow('RGB analyzed output', display_analyzed)

        if top_view_output.shape[2] >= 3:
            cv2.imshow('Top-down View', top_view_output)

        if cv2.waitKey(1) == ord('q'):
            running = False
            break
finally:
    # Clean up resources
    cv2.destroyAllWindows()
    camera.destroy()
    top_camera.destroy()
    lane_invasion_sensor.destroy()
    vehicle.destroy()
    pygame.quit()