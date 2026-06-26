from controller import Robot
import sys
import os
import json
import random
import optparse

try:
    import numpy as np
    from numpy import NaN, nan
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
try:
    import cv2
except ImportError:
    sys.exit("Warning: 'cv2' module not found.")


def clamp(value, value_min, value_max):
    """
    Restricts a given numerical value within a defined lower and upper bound.
    
    Parameters:
        value (float/int): The input number to clamp.
        value_min (float/int): The minimum allowable value.
        value_max (float/int): The maximum allowable value.
    """
    return min(max(value, value_min), value_max)


class DroneController(Robot):
    # --- Flight Dynamics & Control Constants ---
    # Base thrust component required to counteract gravity and maintain a hover.
    K_VERTICAL_THRUST = 68.5  
    # Target height buffer above the literal destination to assist PID stabilization.
    K_VERTICAL_OFFSET = 0.6   
    # Proportional gains for the low-level flight stabilization loops.
    K_VERTICAL_P = 3.0        
    K_ROLL_P = 50.0           
    K_PITCH_P = 30.0          

    # Bounds for directional adjustments applied by high-level navigation.
    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    
    # Distance threshold (in meters) to consider a spatial waypoint successfully reached.
    target_precision = 0.5

    def __init__(self):
        """
        Initializes the drone, configures sensor sample rates, and boots up
        the propulsion systems and onboard payloads.
        """
        Robot.__init__(self)

        # Synchronize controller logic execution cycles with the Webots simulator physics step.
        self.time_step = int(self.getBasicTimeStep())
        self.water_to_drop = 0

        # --- Hardware Device Initialization ---
        self.camera = self.getDevice("camera")
        self.camera.enable(32 * self.time_step) # Throttled frame rate for computer vision performance
        
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        # --- Actuator Layout & Setup ---
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        
        # Tilt gimbal downwards to provide a vertical, top-down perspective of the terrain.
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(1.55)  
        
        # --- Proximity Perception System ---
        self.range_sensor_f = self.getDevice("ds_front")
        self.range_sensor_l = self.getDevice("ds_left")
        self.range_sensor_r = self.getDevice("ds_right")
        
        self.range_sensor_f.enable(self.time_step)
        self.range_sensor_l.enable(self.time_step)
        self.range_sensor_r.enable(self.time_step)

        # Configure motors to use velocity control mode instead of position targets.
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        # --- Navigation & Mission State ---
        self.current_pose = 6 * [0]  # Array representing: [X, Y, Z, yaw, pitch, roll]
        self.target_position = [0, 0, 0]
        self.target_index = 0

        self.world_fire_quadrants = [0, 0]
        self.img_coord_fire = []
        self.WaterDropStatus = False

    def readCamera(self):
        """
        Retrieves raw data from the onboard camera buffer and reformats it 
        into a standard RGB array compatible with standard OpenCV image operations.
        
        Returns:
            np.ndarray: A calibrated, properly oriented 2D image array.
        """
        img = self.camera.getImageArray()
        img = np.asarray(img, dtype=np.uint8)
        
        # Webots native camera output is BGRA; convert to standard RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Compensate for mounting orientation by rotating and flipping the raw matrix.
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(img, 1)

    def set_position(self, pos):
        """
        Updates the local flight state registry with the latest telemetry readings.
        
        Parameters:
            pos (list): Current [X, Y, Z, roll, pitch, yaw] spatial data.
        """
        self.current_pose = pos

    def goToCordinates(self, waypoints, verbose_movement=False, verbose_target=True):
        """
        Computes the telemetry adjustments needed to navigate the drone sequentially 
        through a series of coordinate-based patrol paths.
        
        Parameters:
            waypoints (list): Collection of [X, Y] coordinates defining the flight path.
            verbose_movement (bool): Enables debugging readouts for remaining distance/angles.
            verbose_target (bool): Logs waypoint transition updates to console.
            
        Returns:
            tuple: Calculated (yaw_disturbance, pitch_disturbance) intended for motor inputs.
        """
        # Set initial destination boundary if the navigation loop has just started.
        if self.target_position[0:2] == [0, 0]:
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        # Evaluate if the drone is within acceptable range of the active waypoint.
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):
            # Cycle to the next target location; wrap back to start if loop is complete.
            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ", self.target_position[0:2])

        # Trigonometric calculation to find the heading angle needed to face the target.
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], 
            self.target_position[0] - self.current_pose[0]
        )
        
        # Calculate the angular deflection between current heading and target heading.
        angle_left = self.target_position[2] - self.current_pose[5]
        
        # Normalize the steering angle to keep adjustments within the standard range (-pi to pi].
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if angle_left > np.pi:
            angle_left -= 2 * np.pi

        # Scale steering response based on how aggressively the chassis needs to swing.
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        
        # Use a logarithmic deceleration curve to slow down forward momentum when executing sharp turns.
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1
        )

        if verbose_movement:
            distance_left = np.sqrt(
                ((self.target_position[0] - self.current_pose[0]) ** 2) + 
                ((self.target_position[1] - self.current_pose[1]) ** 2)
            )
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(angle_left, distance_left))
            
        return yaw_disturbance, pitch_disturbance

    def visualServo(self, verbose=True):
        """
        Executes a closed-loop visual positioning adjustment, aligning the drone 
        directly over the center of the tracking targets via image coordinates.
        
        Returns:
            tuple: Dynamic pitch/yaw adjustments to lock position over target center.
        """
        resolutionX, resolutionY = self.camera.getWidth(), self.camera.getHeight()
        x_img, y_img = self.img_coord_fire
        yaw = (self.current_pose[5] + 2 * np.pi) % (2 * np.pi)
        self.world_fire_quadrants = [0, 0]

        # Determine target displacement relative to the camera frame center.
        if abs(x_img - resolutionX / 2) > 20:
            self.world_fire_quadrants[0] = np.sign(x_img - resolutionX / 2)
        if abs(y_img - resolutionY / 2) > 20:
            self.world_fire_quadrants[1] = np.sign(y_img - resolutionY / 2)
            
        # Invert adjustments depending on current yaw orientation to ensure movement directions match reality.
        self.world_fire_quadrants[1] *= np.sign(yaw)
        self.world_fire_quadrants[0] *= -np.sign(yaw)

        # Convert image coordinate displacement offsets into direct attitude inputs.
        yaw_disturbance = self.world_fire_quadrants[0] * clamp(
            abs(x_img - resolutionX / 2), 0, self.MAX_YAW_DISTURBANCE
        )
        pitch_disturbance = self.world_fire_quadrants[1] * clamp(
            abs(y_img - resolutionY / 2), 0, abs(self.MAX_PITCH_DISTURBANCE)
        )

        # Trigger suppression routine once alignment tolerances are met.
        if self.world_fire_quadrants == [0, 0]:
            self.water_to_drop = 15
            if verbose:
                print("Water dropped on fire target: {} at position {}".format(self.target_position[0:2], self.current_pose[0:2]))
            self.img_coord_fire = []

        return yaw_disturbance, pitch_disturbance

    def detectFire(self, verbose=True):
        """
        Processes camera input via color segmentation thresholds to isolate 
        smoke patterns and extract image coordinates.
        
        Returns:
            tuple/None: Centroid (x, y) coordinates of the largest threat area found.
        """
        img = self.readCamera()
        
        # Convert image matrix into HSV color space for consistent color filtering under shifting light conditions.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Lower and upper limits defining the gray-ish signature profiles of smoke plume clouds.
        smoke_lower = np.array([0, 0, 168])
        smoke_upper = np.array([172, 111, 255])

        # Generate binary visibility matrix mask where smoke colors show up as pure white.
        mask_fire = cv2.inRange(hsv, smoke_lower, smoke_upper)

        # Measure what percentage of the image matches the tracking profile.
        fire_ratio = np.round((cv2.countNonZero(mask_fire)) / (img.size / 3) * 100, 2)
        coord_fire = None  

        if fire_ratio > 0.15:  
            # Find boundaries of white pixel blobs within the binary mask.
            contours, _ = cv2.findContours(image=mask_fire, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            contours_poly = [None] * len(contours)
            centers = [None] * len(contours)
            radius = [None] * len(contours)
            radius_max = 0
            
            # Group pixel blobs and find their physical size properties.
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
                
                # Filter out noisy single-pixel calculations by isolating the largest prominent circle region.
                if radius[i] > 3 and radius[i] > radius_max:
                    coord_fire = centers[i]
                    radius_max = radius[i]
                    if verbose:
                        print("fire detected, coordinates {}".format(centers[i]))

            if verbose and coord_fire:  
                # Visual overlay construction for diagnostic image export pipelines.
                drawing = img.copy()
                for i in range(len(contours)):
                    color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                    cv2.drawContours(drawing, contours_poly, i, color)
                    cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
                cv2.imwrite("detectFire.jpg", drawing)

        return coord_fire

    def run(self):
        """
        Core control loop handling parameter parsing, state management execution timing, 
        and lower-level PID mixing matrix operations.
        """
        # Independent scheduling markers for navigation, vision processing, and cooldown clocks.
        t1 = self.getTime()
        t2 = self.getTime()
        t3 = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # --- CLI Argument Configuration Defaults ---
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--patrol_coords", default="11 11, 11 21, 21 21,21 11",
                              help="Specify the patrol coordinates in the format [x1 y1, x2 y2, ...]")
        opt_parser.add_option("--target_altitude", default=42, type=float, 
                              help="target altitude of the robot in meters")
        options, _ = opt_parser.parse_args()

        # Parse command-line coordinates string into structured coordinate matrices.
        point_list = options.patrol_coords.split(',')
        number_of_waypoints = len(point_list)
        waypoints = []
        for i in range(0, number_of_waypoints):
            waypoints.append([])
            waypoints[i].append(float(point_list[i].split()[0]))
            waypoints[i].append(float(point_list[i].split()[1]))

        target_altitude = options.target_altitude

        # Primary physics step loop execution block.
        while self.step(self.time_step) != -1:

            # Update core inertial instrumentation maps.
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            Xpos, Ypos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([Xpos, Ypos, altitude, roll, pitch, yaw])

            # --- Water Actuator Control ---
            if self.water_to_drop > 0:
                self.WaterDropStatus = True
                self.setCustomData(str(self.water_to_drop)) # Communication hook with Webots environment
                self.water_to_drop = 0
            else:
                self.setCustomData(str(0))

            # Trigger navigation and search behaviors only when safe cruise altitudes are established.
            if altitude > target_altitude - 1:
                
                # Navigation Update Loop (10 Hz frequency check).
                if self.getTime() - t1 > 0.1:
                    if self.img_coord_fire:
                        # Fire tracking is verified; initialize precise hover lock routines.
                        yaw_disturbance, pitch_disturbance = self.visualServo()
                    else:
                        # Path clear; proceed along standard survey patterns.
                        yaw_disturbance, pitch_disturbance = self.goToCordinates(waypoints)
                    t1 = self.getTime()
                    
                # CV Inspection Processing Loop (10 Hz frequency check for real-time tracking accuracy).
                if self.getTime() - t2 > 0.1:
                    if not self.WaterDropStatus:
                        self.img_coord_fire = self.detectFire()
                    t2 = self.getTime()

                # Visual suppression blind spot delay handler.
                if not self.WaterDropStatus:
                    t3 = self.getTime()
                if self.getTime() - t3 > 15:  
                    # 15-second tracking lockout prevents dropping payload splash clouds from being re-detected as smoke.
                    self.WaterDropStatus = False

                # --- Distance Sensor Obstacle Avoidance Avoidance Matrix ---
                dist_f = self.range_sensor_f.getValue() 
                dist_l = self.range_sensor_l.getValue() 
                dist_r = self.range_sensor_r.getValue() 
                CRITICAL_PROXIMITY_LIMIT = 300.0 

                if dist_f < CRITICAL_PROXIMITY_LIMIT:
                    pitch_disturbance = 0.5  
                    yaw_disturbance = -0.4 if dist_l < dist_r else 0.4
                elif dist_l < CRITICAL_PROXIMITY_LIMIT:
                    yaw_disturbance = -0.3
                elif dist_r < CRITICAL_PROXIMITY_LIMIT:
                    yaw_disturbance = 0.3

            # --- PID Control Loops & Quadcopter Mixer Matrix ---
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            
            # Non-linear cubic response curve to handle height errors smoothly near equilibrium.
            clamped_difference_altitude = clamp(target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            # Map inputs into physical motor velocity configurations (Mixer Matrix).
            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            # Apply final mixed target parameters into rotational velocities.
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)


# Script entry point setup for the Webots controller engine runner context.
if __name__ == "__main__":
    robot = DroneController()
    robot.run()