def calculate_turn(center_x_left_camera, center_x_right_camera):
    HFOV = 115  # Horizontal Field of View in degrees
    interocular_distance = 130  # Interocular distance in mm
    frame_width = 640  # Total frame width in pixels for each camera

    center_x_frame_right = frame_width / 2
    center_x_frame_left = frame_width / 2

    center_x_avg = (center_x_left_camera + center_x_right_camera) / 2
    center_x_frame_avg = (center_x_frame_right + center_x_frame_left) / 2

    # Calculate the angle based on pixel inputs
    angle = ((center_x_avg - center_x_frame_avg) / interocular_distance) * (HFOV / 2)

    # Convert angle from degrees to steps (assuming 1600 steps per 360 degrees)
    steps_per_degree = 1600 / 360
    steps = angle * steps_per_degree

    return angle

steps_turned = calculate_turn(220, 320)
print(f"Steps to turn: {steps_turned}")
