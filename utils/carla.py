import carla

def move_spectator_to(spectator, transform, distance=5.0, x=0, y=0, z=4, yaw=0, pitch=-30, roll=0):
    back_location = transform.location - transform.get_forward_vector() * distance
    
    back_location.x += x
    back_location.y += y
    back_location.z += z
    transform.rotation.yaw += yaw
    transform.rotation.pitch = pitch
    transform.rotation.roll = roll
    
    spectator_transform = carla.Transform(back_location, transform.rotation)
    
    spectator.set_transform(spectator_transform)

def spawn_vehicle(world, vehicle_index=0, spawn_index=0, spawn_point=None, pattern='vehicle.*'):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter(pattern)[vehicle_index]
    if spawn_point is None:
        spawn_point = world.get_map().get_spawn_points()[spawn_index]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle

def draw_on_screen(world, transform, content='O', color=carla.Color(0, 255, 0), life_time=20):
    world.debug.draw_string(transform.location, content, color=color, life_time=life_time)

def spawn_camera(world, attach_to=None, transform=carla.Transform(carla.Location(x=0.6, y=0.0, z=1.41), carla.Rotation(pitch=0)), width=640, height=640, show_position=False):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)

    if show_position:
        # Disegna un punto viola (RGB: 255, 0, 255) nella posizione della telecamera
        world.debug.draw_point(transform.location, size=0.2, color=carla.Color(255, 0, 255), life_time=120.0)

        # Stampa la posizione della telecamera
        print(f"Camera spawned at: {transform.location}")

    return camera