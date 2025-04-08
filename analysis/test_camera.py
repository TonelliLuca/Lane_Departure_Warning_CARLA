import carla
import cv2
import numpy as np
import time

def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # Get the Audi A2 blueprint
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.audi.a2')

    # Get a valid spawn point
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points available")
        return
    spawn_point = spawn_points[0]

    # Spawn the vehicle
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        print("Failed to spawn vehicle")
        return
    print(f"Vehicle spawned at: {spawn_point.location}")

    # Camera positioned on the dashboard near the windshield
    # Try this camera position for a proper dashboard view
    camera_transform = carla.Transform(
        carla.Location(x=0.6, y=0.0, z=1.41),  # Move slightly backward and adjust height
    )

    # Get camera blueprint
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '640')

    # Attach camera to the vehicle
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print(f"Camera spawned at: {camera_transform.location}")

    #world_camera_location = camera.get_transform().location
    # Convert camera position to world coordinates
    world_camera_location = vehicle.get_transform().transform(camera_transform.location)

    # Draw a box slightly under the camera position
    box_location = carla.Location(
        x=world_camera_location.x,
        y=world_camera_location.y,
        z=world_camera_location.z - 0.63  # Move 0.2 units down from camera
    )

    world.debug.draw_box(
        box=carla.BoundingBox(box_location, carla.Vector3D(0.13, 0.13, 0.13)),
        rotation=camera.get_transform().rotation,
        thickness=0.05,
        color=carla.Color(255, 0, 255),
        life_time=120.0
    )

    # Draw camera direction (forward vector)
    forward_vector = camera.get_transform().get_forward_vector() * 1.0
    end_point = world_camera_location + forward_vector
    world.debug.draw_arrow(
        world_camera_location,
        end_point,
        thickness=0.1,
        arrow_size=0.1,
        color=carla.Color(255, 0, 0),
        life_time=120.0
    )

    # Add text label
    world.debug.draw_string(
        world_camera_location + carla.Location(z=0.5),
        "Camera",
        color=carla.Color(255, 255, 255),
        life_time=120.0
    )

    # Function to process camera images
    def process_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # Convert to BGRA format
        array = array[:, :, :3]  # Remove alpha channel
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # Convert to RGB
        cv2.imshow("Dashboard Camera View", array)
        cv2.waitKey(1)

    # Listen to the camera and show images
    camera.listen(lambda image: process_image(image))

    try:
        time.sleep(60)  # Keep simulation running for 30 seconds
    finally:
        # Cleanup
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == '__main__':
    main()
