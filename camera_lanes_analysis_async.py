#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function



# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc
from utils.DetectionLogger import DetectionLogger

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import threading
import queue
import paho.mqtt.client as mqtt
import json
import ssl

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_z
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# Log file path
log_file_path = "./log/tracked/frame_performance_log.txt"
# Queue to buffer frame data for logging
log_queue = queue.Queue()
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def setup_mqtt_client():
    # Use the VERSION2 of the Callback API to avoid deprecation warning
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="carla_lane_detector")

    # Rest of the function remains the same
    mqtt_client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
    mqtt_client.username_pw_set("***REMOVED***", "***REMOVED***")
    mqtt_client.connect("68194d06420140d29c7cde00549b2f40.s1.eu.hivemq.cloud", 8883)
    mqtt_client.loop_start()
    return mqtt_client


# Initialize the MQTT client
mqtt_client = setup_mqtt_client()

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, test):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart(test)
        self.world.on_tick(hud.on_world_tick)

    def restart(self, test=False):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get the blueprint library
        blueprint_library = self.world.get_blueprint_library().filter(self._actor_filter)

        # Print the list of available vehicles
        print("Available vehicles:")
        for i, blueprint in enumerate(blueprint_library):
            if 'vehicle' in blueprint.id:  # Filter for vehicles only
                print(f"{i}: {blueprint.id}")

        # Select a fixed car (you can change the index here to choose a specific one)
        fixed_car_index = 0  # Example: Choose the first vehicle in the list
        blueprint = blueprint_library[fixed_car_index]
    
        blueprint.set_attribute('role_name', 'hero')
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_point = carla.Transform(
                carla.Location(x=-13.0, y=-180.0, z=2.0),
                carla.Rotation(yaw=90.0)
            )
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================

import json

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            commands = json.load(file)
        print(f'Loaded commands from {file_path}:', commands)
        return commands
    except Exception as e:
        print(f"Error loading commands file: {e}")
        return []


class DualControl(object):
    def __init__(self, world, start_in_autopilot, controller_type='wheel'):
        self._autopilot_enabled = start_in_autopilot
        self.controller_type = controller_type

        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count < 1:
            raise ValueError("No controller detected")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        # Get controller name
        self._controller_name = self._joystick.get_name()
        print(f"Detected controller: {self._controller_name}")

        if self.controller_type == 'wheel':
            self._parser = ConfigParser()

            self._parser.read('.\wheel_config.ini')
            self._steer_idx = int(
                self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(
                self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            self._handbrake_idx = int(
                self._parser.get('G29 Racing Wheel', 'handbrake'))
        else:
            self._xbox_steer_axis = 0  # Left stick horizontal
            self._xbox_throttle_axis = 5  # Right trigger
            self._xbox_brake_axis = 2  # Left trigger
            self._xbox_handbrake_button = 0  # A button

        self._step_counter = 0

    def parse_events(self, world, clock, test):
        if not hasattr(self, '_recording'):
            self._recording = False
            self._recorded_inputs = []
        if test:
            global commands
            # Only handle QUIT events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            # If we have not reached the end of commands, apply them
            if self._step_counter < len(commands):
                print(f"Step: {self._step_counter}")
                t_val, b_val, s_val = commands[self._step_counter]
                self._control.throttle = t_val
                self._control.brake = b_val
                self._control.steer = s_val
                print(f"Throttle: {t_val}, Brake: {b_val}, Steer: {s_val}")
                self._step_counter += 1

            # Apply control
            if isinstance(self._control, carla.VehicleControl):
                world.player.apply_control(self._control)
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_z:
                    self._recording = not self._recording
                    if self._recording:
                        self._recorded_inputs = []  # Clear previous recordings
                        world.hud.notification('Recording inputs started')
                    else:
                        if self._recorded_inputs:
                            # Save recorded inputs to a JSON file
                            with open('recorded_inputs.json', 'w') as f:
                                json.dump(self._recorded_inputs, f)
                            world.hud.notification(
                                f'Recording saved to recorded_inputs.json ({len(self._recorded_inputs)} frames)')
                        else:
                            world.hud.notification('Recording stopped (no inputs recorded)')
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                if self.controller_type == 'wheel':
                    self._parse_vehicle_wheel()
                elif self.controller_type == 'xbox':  # xbox controller
                    self._parse_vehicle_xbox()
                elif self.controller_type == 'keyboard':  # keyboard
                    self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                else:
                    raise ValueError("Invalid controller type. Use 'wheel', 'xbox', or 'keyboard'.")
                self._control.reverse = self._control.gear < 0

                # Record the current inputs if recording is active
                if self._recording:
                    self._recorded_inputs.append([
                        float(self._control.throttle),
                        float(self._control.brake),
                        float(self._control.steer)
                    ])
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_xbox(self):
        """Parse input from Xbox controller"""
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        # Xbox One controller typically uses:
        # - Right trigger (axis 5): -1 (not pressed) to 1 (fully pressed)
        # - Left trigger (axis 2): -1 (not pressed) to 1 (fully pressed)
        # - Left stick horizontal (axis 0): -1 (left) to 1 (right)

        # Map trigger values from [-1,1] to [0,1]
        self._control.throttle = max(0, (jsInputs[self._xbox_throttle_axis] + 1) / 2)
        #self._control.brake = max(0, (jsInputs[self._xbox_brake_axis] + 1) / 2)
        self._control.brake = 0
        # Apply deadzone to steering
        steer_raw = jsInputs[self._xbox_steer_axis]
        deadzone = 0.1
        if abs(steer_raw) < deadzone:
            self._control.steer = 0
        else:
            self._control.steer = steer_raw

        # Print control values for debugging
        # print(
        #     f"Control values - Throttle: {self._control.throttle}, Brake: {self._control.brake}, Steer: {self._control.steer}")

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        #throttleCmd = K2 + (2.05 * math.log10(
        #    -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        throttleCmd = K2 + (2.05 * math.log10(
            0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        #brakeCmd = 1.6 + (2.05 * math.log10(
        #    -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        brakeCmd = 1.6 + (2.05 * math.log10(
            0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            #'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


import cv2
from utils.YOLOPModel import initializeYOLOPModel, analyzeImage
from utils.carla import spawn_vehicle, spawn_camera
from datetime import datetime
from utils.image_cropper import process_image

# Define the crop size for image processing
define_crop_size = 320

# Initialize video output arrays
video_output = np.zeros((640, 640, 4), dtype=np.uint8)
video_output_seg = np.zeros((720, 1280, 3), dtype=np.uint8)
top_view_output = np.zeros((480, 640, 4), dtype=np.uint8)  # For top-down camera


 # Lane invasion detection variables
lane_invasion_detected = False  # CARLA detection
carla_lane_invasion_timestamp = datetime.now()
yolop_lane_invasion_detected = False  # Our YOLOP detection
yolop_lane_invasion_timestamp = datetime.now()
blink_state = False
blink_timer = 0
last_detection_frames = 0
detection_threshold = 3  # Require consecutive detections to reduce false positives
detection_logger = DetectionLogger()


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
# Function to write data to the log file
def log_frame_data(frame_data):
    with open(log_file_path, "a") as log_file:
        log_file.write(frame_data + "\n")

# Logging thread function
def logging_thread():
    while True:
        # Wait until there's frame data in the queue
        frame_data = log_queue.get()  # Blocks until data is available
        if frame_data == "STOP":
            break  # Exit the logging thread when "STOP" signal is received
        log_frame_data(frame_data)
        log_queue.task_done()

# Start the logging thread
log_thread = threading.Thread(target=logging_thread)
log_thread.daemon = True  # Daemonize the thread so it will exit when the main program ends
log_thread.start()

import time

def camera_callback(image):
    """
    Callback function for the camera. Processes the image and starts a new thread for inference.
    """
    global video_output, video_output_seg, yolop_lane_invasion_detected
    try:
        start_time = time.time()
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
                if crossing:
                    mqtt_message = {
                        "event": "lane_crossing",
                        "system": "YOLOP",
                        "crossing": crossing,
                        "timestamp": datetime.now().isoformat(),
                    }

                    mqtt_client.publish(
                        topic="carla/lane_detection",
                        payload=json.dumps(mqtt_message),
                        qos=1
                    )

            video_output_seg = result
            # Set the global YOLOP flag directly from the crossing result
            yolop_lane_invasion_detected = crossing

        # Record the end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Collect relevant frame data to log
        frame_data = (
            f"Frame Data:\n"
            f"Time taken for image processing and analysis: {elapsed_time:.4f} seconds\n"
            f"Image Height: {image.height}, Image Width: {image.width}\n"
            f"Image Size (after processing): {image_to_analyze.shape if image_to_analyze.size > 0 else 'Invalid'}\n"
            f"Max value in processed image: {np.max(image_to_analyze) if image_to_analyze.size > 0 else 'N/A'}\n"
            f"Result: {'Detected' if result is not None and result.size > 0 else 'No result'}\n"
            f"Crossing: {'Yes' if yolop_lane_invasion_detected else 'No'}\n"
            f"{'---' * 20}\n"
        )

        # Push the frame data into the queue to be logged by the logging thread
        log_queue.put(frame_data)

    except Exception as e:
        print(f"Error in camera callback: {str(e)}")


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

    # Agreement rate with color coding
    agreement_rate = stats.get('agreement_rate', 0) * 100
    color = (0, 100, 0) if agreement_rate > 70 else (0, 0, 200) if agreement_rate < 50 else (0, 150, 150)
    cv2.putText(test_display, f"Agreement rate: {agreement_rate:.1f}%",
                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Detection counts
    cv2.putText(test_display, f"YOLOP only: {stats.get('yolop_only', 0)}",
                (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 1, cv2.LINE_AA)
    cv2.putText(test_display, f"CARLA only: {stats.get('carla_only', 0)}",
                (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(test_display,
                f"Both systems: {stats.get('events', 0) - stats.get('yolop_only', 0) - stats.get('carla_only', 0)}",
                (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # Add time information
    time_now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(test_display, f"Time: {time_now}",
                (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

    # Draw a border
    cv2.rectangle(test_display, (10, 10), (590, 390), (0, 0, 0), 2)

    return test_display


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    global blink_timer, lane_invasion_detected, blink_state
    pygame.init()
    pygame.font.init()
    world = None
    # Initialize test display if in test mode
    test_display = None
    if args.test:
        global commands
        commands = read_json_file(args.test_file)
        test_display = setup_test_window()
        stats_display_counter = 0
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        #display = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)

        hud = HUD(args.width, args.height)
        client.load_world('Town04')
        initializeYOLOPModel()

        world = World(client.get_world(), hud, args.filter, args.test)

        controller = DualControl(world, args.autopilot, args.controller)
        
        # Spawn the vehicle and camera in the CARLA world
        camera = spawn_camera(client.get_world(), attach_to=world.player)

        # Set up camera with callback
        camera.listen(lambda image: camera_callback(image))


        # Create and attach the lane invasion sensor
        lane_invasion_sensor = spawn_lane_invasion_sensor(client.get_world(), attach_to=world.player)
        lane_invasion_sensor.listen(lambda event: lane_invasion_callback(event))
        # Create and set up top-down camera
        top_camera = spawn_camera(client.get_world(), attach_to=world.player,
                                            transform=carla.Transform(carla.Location(x=0, y=0, z=15),
                                                                        carla.Rotation(pitch=-90)), width=640, height=480)
        top_camera.listen(lambda image: top_camera_callback(image))

        # Set up display windows
        cv2.namedWindow('Original RGB feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('RGB analyzed output', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Top-down View', cv2.WINDOW_AUTOSIZE)

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(300)
            if controller.parse_events(world, clock, args.test):
                return


            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # Update blinking state (toggle every 0.5 seconds)
            blink_timer += 1
            if blink_timer >= 15:  # Assuming ~30 fps, toggle every 0.5 seconds
                blink_timer = 0
                blink_state = not blink_state

            # Reset Carla lane invasion after timeout (3 seconds)
            if lane_invasion_detected and (datetime.now() - carla_lane_invasion_timestamp).total_seconds() > 1:
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

            # Update and show test display if in test mode
            if args.test:
                stats_display_counter += 1
                if stats_display_counter >= 30:  # Update every 30 frames (0.25 sec at 120fps)
                    test_display = update_test_display(test_display)
                    stats_display_counter = 0

                cv2.imshow('Test Results', test_display)

                # Save results periodically or on key press
                key = cv2.waitKey(1)
                if key == ord('s'):  # Press 's' to save current stats
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'test_results_{timestamp}.png', test_display)
                    print(f"Test results saved as test_results_{timestamp}.png")


    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--controller',
        default='wheel',
        choices=['wheel', 'xbox', 'keyboard'],
        help='Control method: wheel (Logitech G29) or xbox (Xbox One controller) or keyboard (default: wheel)'
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='Run in test mode without controller')
    argparser.add_argument(
        '--test-file',
        type=str,
        default='commands.json',
        help='Specify the JSON file with commands for test mode')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
