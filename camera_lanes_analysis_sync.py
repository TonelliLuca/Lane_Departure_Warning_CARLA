#!/usr/bin/env python
import argparse
# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import json
import os
import re

import sys
from datetime import datetime

import cv2

from utils.DetectionLogger import DetectionLogger
from utils.YOLOPModel import initializeYOLOPModel, analyzeImage
from utils.carla import spawn_camera
from utils.image_cropper import process_image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

# Initialize the video output
video_output = np.zeros((640, 640, 4), dtype=np.uint8)
video_output_seg = np.zeros((720, 1280, 3), dtype=np.uint8)
detection_logger = DetectionLogger()
yolop_lane_invasion_detected = False  # Our YOLOP detection



class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def camera_callback(image):
    """
    Callback function for the camera. Processes the image and starts a new thread for inference.
    """
    global video_output, video_output_seg, yolop_lane_invasion_detected
    try:
        # Copy image data safely
        video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Process the image and run inference
        image_to_analyze = process_image(video_output, 320)

        # Skip processing if image is invalid
        if image_to_analyze.size == 0 or np.max(image_to_analyze) == 0:
            print("Warning: Invalid image received, skipping analysis")
            return

        # Perform image analysis with error handling
        result, crossing = analyzeImage(image_to_analyze)
        if result is not None and result.size > 0:
            if crossing != yolop_lane_invasion_detected:
                detection_logger.log_detection("YOLOP", crossing)
            video_output_seg = result
            # Set the global YOLOP flag directly from the crossing result
            yolop_lane_invasion_detected = crossing


    except Exception as e:
        print(f"Error in camera callback: {str(e)}")

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
    # Log the detection
    detection_logger.log_detection("CARLA", True)

    # Cross-validate with YOLOP detection
    if yolop_lane_invasion_detected:
        print("Both systems detected lane crossing - HIGH CONFIDENCE")



def carla_img_to_opencv(carla_img):
    """Convert CARLA image to OpenCV format (BGR)"""
    array = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_img.height, carla_img.width, 4))
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

def setup_test_window():
    """Set up a dedicated window for displaying test results"""
    cv2.namedWindow('Test Results', cv2.WINDOW_AUTOSIZE)
    # Create a blank canvas for the test results
    test_display = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light gray background
    return test_display


def update_test_display(test_display):
    """Update the test results display with current statistics"""
    # Start with a clean slate
    test_display.fill(240)

    # Get statistics from the detector logger
    stats = detection_logger.get_stats()

    # Add title
    cv2.putText(test_display, "Lane Detection Comparison",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Add statistics
    cv2.putText(test_display, f"Total events: {stats['events']}",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    # Detection counts
    cv2.putText(test_display, f"YOLOP only: {stats.get('yolop_only', 0)}",
                (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 1, cv2.LINE_AA)
    cv2.putText(test_display, f"CARLA only: {stats.get('carla_only', 0)}",
                (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(test_display,
                f"Confirmed Crossings: {stats.get('agreements', 0)}",
                (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # Add time information
    time_now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(test_display, f"Time: {time_now}",
                (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

    # Draw a border
    cv2.rectangle(test_display, (10, 10), (590, 390), (0, 0, 0), 2)

    return test_display


def get_sequential_filename():
    """Generate a sequential filename for recordings"""
    # Create recorded subfolder if it doesn't exist
    recorded_dir = os.path.join("test_commands", "recorded")
    os.makedirs(recorded_dir, exist_ok=True)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check existing files in the recorded subfolder
    pattern = os.path.join(recorded_dir, f"control_log_*.json")
    existing_files = glob.glob(pattern)

    # Count how many files exist and add one
    next_number = len(existing_files) + 1

    # Create filename with timestamp and sequence number
    filename = f"control_log_{timestamp}_{next_number:03d}.json"
    return os.path.join(recorded_dir, filename)

def get_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def add_weather_argument(parser):
    # Get all weather presets
    weather_presets = get_weather_presets()
    weather_choices = [name for _, name in weather_presets]  # Just the names for the CLI choices
    parser.add_argument('--weather', choices=weather_choices, default='Clear Noon',
                        help="Select weather preset from available options (default: 'Clear Noon').")

def main(args, playback_data=None, playback_index=0):
    actor_list = []
    pygame.init()
    # rest of function remains the same

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    client.load_world('Town04')
    world = client.get_world()

    # Apply the selected weather preset
    weather_presets = get_weather_presets()
    print(weather_presets)
    preset_name = args.weather  # The weather preset passed in the arguments
    weather = next((weather for weather, name in weather_presets if name == preset_name), None)
    if weather:
        world.set_weather(weather)
        print(f"Weather set to: {preset_name}")
    else:
        print(f"Weather preset '{preset_name}' not found.")

    try:
        spawn_point = world.get_map().get_spawn_points()[0]

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi.a2')),
            spawn_point)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(True)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        if args.playback:
            test_display = setup_test_window()
            stats_display_counter = 0
            # Create and attach the lane invasion sensor
            lane_invasion_sensor = spawn_lane_invasion_sensor(client.get_world(), attach_to=vehicle)
            lane_invasion_sensor.listen(lambda event: lane_invasion_callback(event))

        initializeYOLOPModel()

        camera = spawn_camera(world, attach_to=vehicle)
        actor_list.append(camera)

        control = carla.VehicleControl()
        control.throttle = 0
        control.steer = 0
        control.brake = 0
        data_log = []

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera, fps=30) as sync_mode:
            cv2.namedWindow('Original RGB feed', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RGB analyzed output', cv2.WINDOW_AUTOSIZE)
            while True:
                if should_quit():
                    return
                clock.tick()

                if args.playback and playback_index < len(playback_data):
                    # Get the current control from playback data
                    control_data = playback_data[playback_index]

                    # Apply controls from playback data
                    control.throttle = control_data["throttle"]
                    control.brake = control_data["brake"]
                    control.steer = control_data["steer"]

                    playback_index += 1
                    print("Playback index ", playback_index, "/", len(playback_data))


                elif not args.playback:
                    # Use keyboard controls if not in playback mode
                    keys = pygame.key.get_pressed()

                    if keys[pygame.K_w]:  # Accelerate
                        control.throttle = min(control.throttle + 0.20, 1.0)
                    else:
                        control.throttle = max(control.throttle - 0.20, 0.0)

                    if keys[pygame.K_s]:  # Brake
                        control.brake = min(control.brake + 0.20, 1.0)
                    else:
                        control.brake = max(control.brake - 0.20, 0.0)

                    if keys[pygame.K_a]:  # Turn left
                        control.steer = max(control.steer - 0.05, -1.0)
                    elif keys[pygame.K_d]:  # Turn right
                        control.steer = min(control.steer + 0.05, 1.0)
                    else:
                        control.steer = 0  # Straighten wheel if no input

                vehicle.apply_control(control)

                # Get data from all sensors
                out = sync_mode.tick(timeout=2.0)
                snapshot, image_rgb= out[0], out[1]
                camera_image = out[2]

                # Choose the next waypoint and update the car location.
                # Process camera data directly here
                camera_callback(camera_image)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                if args.record:
                    data_log.append({
                        "timestamp": snapshot.timestamp.elapsed_seconds,
                        "throttle": control.throttle,
                        "brake": control.brake,
                        "steer": control.steer
                    })
                # Use this:
                if video_output.shape[2] == 4:
                    # Convert from RGBA to BGR format
                    bgr_image = cv2.cvtColor(video_output[:, :, :3], cv2.COLOR_RGB2BGR)
                    cv2.imshow('Original RGB feed', bgr_image)

                # For the segmented output:
                if video_output_seg.shape[2] == 3:
                    cv2.imshow('RGB analyzed output', video_output_seg)

                # Draw the display.
                draw_image(display, image_rgb)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
                cv2.waitKey(1)

                # Update and show test display if in test mode
                if args.playback:
                    stats_display_counter += 1
                    if stats_display_counter >= 10:  # Update every 30 frames (0.25 sec at 120fps)
                        test_display = update_test_display(test_display)
                        stats_display_counter = 0

                    cv2.imshow('Test Results', test_display)


    finally:
        if args.record:
            print('Saving recorded data...')
            new_file = get_sequential_filename()
            with open(new_file, 'w') as f:
                json.dump(data_log, f, indent=4)
        elif args.playback:
            try:
                log_dir = './log/untracked'
                os.makedirs(log_dir, exist_ok=True)
                log_file_path = os.path.join(log_dir, 'test_log.txt')

                # Extract the file name from the path, remove extension, and add a timestamp
                test_name = os.path.basename(playback_file)
                base_name = os.path.splitext(test_name)[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                final_stats = detection_logger.get_stats()
                log_line = (
                    f"TestName: {base_name}; "
                    f"Timestamp: {timestamp}; "
                    f"Weather: {preset_name}; "
                    f"PlaybackIndex: {playback_index}; "
                    f"Results: {final_stats}\n"
                )

                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_line)

            except Exception as e:
                print(f"Error while logging test data: {str(e)}")

        print('Destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Autonomous Driving Simulation')
    parser.add_argument('--record', action='store_true', help='Enable recording of control data')
    parser.add_argument('--playback', nargs='?', const='control_log.json',
                        help='Play back recorded control data from a given file')
    add_weather_argument(parser)
    args = parser.parse_args()

    playback_data = []
    playback_index = 0

    if args.playback:
        # Use the filename directly if it doesn't specify a directory
        playback_file = args.playback
        try:
            with open(playback_file, 'r') as f:
                playback_data = json.load(f)
            print(f"Loaded {len(playback_data)} control records from {playback_file}")
        except FileNotFoundError:
            print(f"Error: File not found: {playback_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: {playback_file} is not a valid JSON file.")
            sys.exit(1)

    try:
        main(args, playback_data, playback_index)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')