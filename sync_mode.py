#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
from datetime import datetime

import cv2

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
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
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
        print("image shape", image.height, image.width)
        print("image class", image.__class__)
        cv2.imwrite('saved_image.png', video_output)
        print("image saved")
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
            video_output_seg = result
            # Set the global YOLOP flag directly from the crossing result
            yolop_lane_invasion_detected = crossing

    except Exception as e:
        print(f"Error in camera callback: {str(e)}")

def carla_img_to_opencv(carla_img):
    """Convert CARLA image to OpenCV format (BGR)"""
    array = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_img.height, carla_img.width, 4))
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    client.load_world('Town04')
    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi.a2')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        initializeYOLOPModel()

        camera = spawn_camera(world, attach_to=vehicle)
        actor_list.append(camera)

        yolop_lane_invasion_detected = False

        #camera.listen(lambda image: camera_callback(image))

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera, fps=30) as sync_mode:
            cv2.namedWindow('Original RGB feed', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RGB analyzed output', cv2.WINDOW_AUTOSIZE)
            while True:
                if should_quit():
                    return
                clock.tick()

                # Get data from all sensors
                out = sync_mode.tick(timeout=2.0)
                print("out: ", out)
                snapshot, image_rgb= out[0], out[1]
                camera_image = out[2]
                print("camera_image: ", camera_image)

                # Choose the next waypoint and update the car location.
                # Process camera data directly here
                camera_callback(camera_image)
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                print("video output shape: ", video_output.shape)
                print("video output seg shape: ", video_output_seg.shape)
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


    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')