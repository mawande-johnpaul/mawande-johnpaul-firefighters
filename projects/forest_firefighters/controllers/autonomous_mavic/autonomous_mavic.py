from controller import Robot
import sys
import os
import json
import random
import optparse
import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("Warning: 'cv2' module not found.")


def limit_value(val, min_val, max_val):
    return min(max(val, min_val), max_val)


# Path configurations for supervisor fire coordinate updates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIVE_FIRE_REGISTRY = os.path.join(BASE_DIR, 'fire_targets.json')


class Mavic (Robot):
    # Core flight dynamics constants (empirically found)
    BASE_VERTICAL_FORCE = 68.5  # Base thrust value required for drone lift
    HEIGHT_STABILIZATION_BIAS = 0.6  # Altitude target tuning offset
    GAIN_ALTITUDE_P = 3.0        # Proportional multiplier for height adjustments
    GAIN_ROLL_P = 50.0           # Proportional multiplier for lateral tilt adjustments
    GAIN_PITCH_P = 30.0          # Proportional multiplier for longitudinal tilt adjustments

    LIMIT_YAW_MODIFIER = 0.4
    LIMIT_PITCH_MODIFIER = -1
    GOAL_PROXIMITY_RADIUS = 0.5  # Proximity window to confirm target arrival (meters)

    def __init__(self):
        Robot.__init__(self)

        self.control_step_ms = int(self.getBasicTimeStep())
        self.payload_volume_to_release = 0  # Remaining fluid payload to discharge

        # Hardware initialization and configuration
        self.payload_cam = self.getDevice("camera")
        self.payload_cam.enable(32 * self.control_step_ms)
        self.spatial_imu = self.getDevice("inertial unit")
        self.spatial_imu.enable(self.control_step_ms)
        self.position_gps = self.getDevice("gps")
        self.position_gps.enable(self.control_step_ms)
        self.rotation_gyro = self.getDevice("gyro")
        self.rotation_gyro.enable(self.control_step_ms)

        self.actuator_fl = self.getDevice("front left propeller")
        self.actuator_fr = self.getDevice("front right propeller")
        self.actuator_rl = self.getDevice("rear left propeller")
        self.actuator_rr = self.getDevice("rear right propeller")
        self.gimbal_pitch = self.getDevice("camera pitch")
        self.gimbal_pitch.setPosition(1.55)  # Establishes vertical point of view
        
        propellers = [self.actuator_fl, self.actuator_fr,
                      self.actuator_rl, self.actuator_rr]
        
        # Proximity perception system for obstacle avoidance
        self.range_sensor_f = self.getDevice("ds_front")
        self.range_sensor_l = self.getDevice("ds_left")
        self.range_sensor_r = self.getDevice("ds_right")
        
        self.range_sensor_f.enable(self.control_step_ms)
        self.range_sensor_l.enable(self.control_step_ms)
        self.range_sensor_r.enable(self.control_step_ms)

        for prop in propellers:
            prop.setPosition(float('inf'))
            prop.setVelocity(1)

        self.telemetry_pose = 6*[0]          # Array format: [X, Y, Z, yaw, pitch, roll]
        self.active_destination = [0, 0, 0]  # Coordinates of current spatial goal
        self.waypoint_sequence_id = 0        # Current tracking index in waypoints list

        self.spatial_fire_quadrants = [0, 0] # Alignment flags relative to the target center
        self.pixel_coords_fire = []          # Screen spatial position of target smoke
        self.discharge_active = False        # Operational lock to prevent re-triggering during a drop

    def evaluate_proximity_hazards(self):
        """
        Processes distance sensor inputs to detect close obstacles and calculate 
        reactive adjustment commands for yaw and pitch to prevent crashes.
        """
        dist_f = self.range_sensor_f.getValue()  # Direct forward clearance metric
        dist_l = self.range_sensor_l.getValue()  # Left lateral clearance metric
        dist_r = self.range_sensor_r.getValue()  # Right lateral clearance metric

        if self.getTime() - getattr(self, 'timestamp_last_sensor_log', 0) > 1.0:
            print(f"[{self.getName()}] SENSORS -> Front: {dist_f:.1f} | Left: {dist_l:.1f} | Right: {dist_r:.1f}")
            self.timestamp_last_sensor_log = self.getTime()

        CRITICAL_PROXIMITY_LIMIT = 300.0 
        yaw_correction = 0.0
        pitch_correction = 0.0
        hazard_present = False

        if dist_f < CRITICAL_PROXIMITY_LIMIT:
            hazard_present = True
            pitch_correction = 0.5  # Pitch backward away from obstacle
            yaw_correction = -0.4 if dist_l < dist_r else 0.4  # Steer toward wider clearance gap
        elif dist_l < CRITICAL_PROXIMITY_LIMIT:
            hazard_present = True
            yaw_correction = -0.3  # Pivot away from left wall
        elif dist_r < CRITICAL_PROXIMITY_LIMIT:
            hazard_present = True
            yaw_correction = 0.3   # Pivot away from right wall

        return hazard_present, yaw_correction, pitch_correction

    def capture_processed_frame(self):
        """
        Extracts raw image buffers from the onboard camera sensor, 
        reshapes and converts it to standard RGB, and handles spatial rotation.
        """
        w, h = self.payload_cam.getWidth(), self.payload_cam.getHeight()
        raw_pixels = self.payload_cam.getImage()  # Extracted raw data stream
        matrix = np.frombuffer(raw_pixels, np.uint8).reshape((h, w, 4))
        matrix = cv2.cvtColor(matrix, cv2.COLOR_BGRA2RGB)
        matrix = cv2.rotate(matrix, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(matrix, 1)

    def update_telemetry_pose(self, absolute_position):
        """
        Updates internal tracking matrix containing current global spatial positioning and rotation state.
        """
        self.telemetry_pose = absolute_position

    def compute_navigation_vectors(self, track_nodes, output_verbose_flow=False, output_verbose_target=True):
        """
        Determines alignment and movement velocity outputs based on differences between 
        the drone's actual coordinates and the current active patrol path nodes.
        """
        if self.active_destination[0:2] == [0, 0]:  
            self.active_destination[0:2] = track_nodes[0]
            if output_verbose_target:
                print("First target: ", self.active_destination[0:2])

        # Verify whether drone has closed the distance gap to target point
        if all([abs(pos_a - pos_b) < self.GOAL_PROXIMITY_RADIUS for (pos_a, pos_b) in zip(self.active_destination, self.telemetry_pose[0:2])]):

            self.waypoint_sequence_id += 1
            if self.waypoint_sequence_id > len(track_nodes)-1:
                self.waypoint_sequence_id = 0
            self.active_destination[0:2] = track_nodes[self.waypoint_sequence_id]
            if output_verbose_target:
                print("Target reached! New target: ", self.active_destination[0:2])

        self.active_destination[2] = np.arctan2(
            self.active_destination[1] - self.telemetry_pose[1], self.active_destination[0] - self.telemetry_pose[0])
        
        angular_deficit = self.active_destination[2] - self.telemetry_pose[5]  # Remainder angle before matching path angle
        angular_deficit = (angular_deficit + 2*np.pi) % (2*np.pi)
        if (angular_deficit > np.pi):
            angular_deficit -= 2*np.pi

        yaw_disturbance = self.LIMIT_YAW_MODIFIER * angular_deficit / (2*np.pi)
        pitch_disturbance = limit_value(
            np.log10(abs(angular_deficit)), self.LIMIT_PITCH_MODIFIER, 0.1)

        if output_verbose_flow:
            range_deficit = np.sqrt(((self.active_destination[0] - self.telemetry_pose[0]) ** 2) + (
                (self.active_destination[1] - self.telemetry_pose[1]) ** 2))  # Absolute distance gap to target point
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angular_deficit, range_deficit))
        return yaw_disturbance, pitch_disturbance

    def get_cords(self):
        """
        Reads global targets map shared by supervisor monitoring system 
        to track locations that are active areas of interest.
        """
        try:
            with open(LIVE_FIRE_REGISTRY, 'r') as file_stream:
                extracted_points = json.load(file_stream)
        except (OSError, ValueError):
            return []
        if not isinstance(extracted_points, list):
            return []
        return extracted_points

    def redirect_towards_hazard(self, locations_array, output_verbose_target=True):
        """
        Overrides typical route behaviors to shift primary focus and calculate 
        vector paths directly toward the closest active high-temperature thermal point.
        """
        closest_point = min(locations_array, key=lambda point: (point[0] - self.telemetry_pose[0]) ** 2
                      + (point[1] - self.telemetry_pose[1]) ** 2)  # Calculated spatial point representing nearest destination

        if self.active_destination[0:2] != list(closest_point):
            self.active_destination[0:2] = list(closest_point)
            if output_verbose_target:
                print("Heading to fire at: ", self.active_destination[0:2])

        self.active_destination[2] = np.arctan2(
            self.active_destination[1] - self.telemetry_pose[1],
            self.active_destination[0] - self.telemetry_pose[0])
        angular_deficit = self.active_destination[2] - self.telemetry_pose[5]
        angular_deficit = (angular_deficit + 2 * np.pi) % (2 * np.pi)
        if angular_deficit > np.pi:
            angular_deficit -= 2 * np.pi

        yaw_disturbance = self.LIMIT_YAW_MODIFIER * angular_deficit / (2 * np.pi)
        pitch_disturbance = limit_value(
            np.log10(abs(angular_deficit)), self.LIMIT_PITCH_MODIFIER, 0.1)
        return yaw_disturbance, pitch_disturbance

    def execute_overhead_alignment(self, output_verbose=True):
        """
        Performs fine adjustments using optical tracking information to center the drone 
        directly over a fire position, triggering deployment once alignment is verified.
        """
        res_x, res_y = self.payload_cam.getWidth(), self.payload_cam.getHeight()
        target_x, target_y = self.pixel_coords_fire  # Extracted image plane coordinates
        heading_yaw = (self.telemetry_pose[5] + 2*np.pi) % (2*np.pi)  # Normalized rotational heading
        self.spatial_fire_quadrants = [0, 0]

        if abs(target_x - res_x / 2) > 20:
            self.spatial_fire_quadrants[0] = np.sign(target_x - res_x / 2)
        if abs(target_y - res_y / 2) > 20:
            self.spatial_fire_quadrants[1] = np.sign(target_y - res_y / 2)
        self.spatial_fire_quadrants[1] *= np.sign(heading_yaw)
        self.spatial_fire_quadrants[0] *= -np.sign(heading_yaw)

        yaw_disturbance = self.spatial_fire_quadrants[0] * limit_value(
            abs(target_x - res_x / 2), 0, self.LIMIT_YAW_MODIFIER)
        pitch_disturbance = self.spatial_fire_quadrants[1] * limit_value(
            abs(target_y - res_y / 2), 0, abs(self.LIMIT_PITCH_MODIFIER))

        if self.spatial_fire_quadrants == [0, 0]:
            self.payload_volume_to_release = 15
            if output_verbose:
                print("Water dropped on fire target: {} at position {}".format(
                    self.active_destination[0:2], self.telemetry_pose[0:2]))
            self.pixel_coords_fire = []

        return yaw_disturbance, pitch_disturbance

    def run_image_hazard_scan(self, output_verbose=True):
        """
        Processes image frames via HSV isolation to distinguish smoke characteristics, 
        identifies targets using area contour definitions, and highlights matches.
        """
        frame = self.capture_processed_frame()
        hsv_layer = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # Converted operational frame matrix

        hsv_floor = np.array([0, 0, 168])   # Lower bound range for smoke filtering
        hsv_roof = np.array([172, 111, 255])  # Upper bound range for smoke filtering

        segmented_mask = cv2.inRange(hsv_layer, hsv_floor, hsv_roof)

        target_center = None
        pixel_density_ratio = np.round(
            (cv2.countNonZero(segmented_mask)) / (frame.size / 3) * 100, 2)  # Percentage matching expected targets
        if pixel_density_ratio > 0.15:

            shape_contours, _ = cv2.findContours(
                image=segmented_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            processed_polys = [None] * len(shape_contours)
            enclosing_centers = [None] * len(shape_contours)
            enclosing_radii = [None] * len(shape_contours)
            peak_radius_found = 0  # Maximum tracker size used for localization filters
            
            for index, contour in enumerate(shape_contours):
                processed_polys[index] = cv2.approxPolyDP(contour, 3, True)
                enclosing_centers[index], enclosing_radii[index] = cv2.minEnclosingCircle(
                    processed_polys[index])
                if enclosing_radii[index] > 3 and enclosing_radii[index] > peak_radius_found:
                    target_center = enclosing_centers[index]
                    peak_radius_found = enclosing_radii[index]
                    if output_verbose:
                        print("fire detected, coordinates {}".format(enclosing_centers[index]))

            if output_verbose:  
                visual_canvas = frame.copy()  # Copied frame used for drawing tracking indicators
                for index in range(len(shape_contours)):
                    random_color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                    cv2.drawContours(visual_canvas, processed_polys, index, random_color)
                    cv2.circle(visual_canvas, (int(enclosing_centers[index][0]), int(
                        enclosing_centers[index][1])), int(enclosing_radii[index]), random_color, 2)
                cv2.imwrite("fire_detection.jpg", visual_canvas)
            return target_center

    def run(self):
        timestamp_nav_update = self.getTime()   # Tracking timeline variable for motion iterations
        timestamp_scan_update = self.getTime()  # Tracking timeline variable for computer vision iterations
        timestamp_discharge_lock = self.getTime()  # Timing anchor to clear payload status drops safely

        roll_adjustment = 0
        pitch_adjustment = 0
        yaw_adjustment = 0

        argument_parser = optparse.OptionParser()
        argument_parser.add_option("--patrol_coords", default="11 11, 11 21, 21 21,21 11",
                              help="Specify the patrol coordinates in the format [x1 y1, x2 y2, ...]")
        argument_parser.add_option("--target_altitude", default=42,
                              type=float, help="target altitude of the robot in meters")
        parsed_options, _ = argument_parser.parse_args()

        raw_node_strings = parsed_options.patrol_coords.split(',')  # Array containing raw text numbers
        total_waypoints_count = len(raw_node_strings)
        configured_waypoints = []
        for index in range(0, total_waypoints_count):
            configured_waypoints.append([])
            configured_waypoints[index].append(float(raw_node_strings[index].split()[0]))
            configured_waypoints[index].append(float(raw_node_strings[index].split()[1]))

        cruise_altitude = parsed_options.target_altitude  # Main target flight height assignment

        while self.step(self.control_step_ms) != -1:

            # Retrieve telemetry data from sensors
            telemetry_roll, telemetry_pitch, telemetry_yaw = self.spatial_imu.getRollPitchYaw()
            gps_x, gps_y, current_alt = self.position_gps.getValues()
            accel_roll, accel_pitch, _ = self.rotation_gyro.getValues()
            self.update_telemetry_pose([gps_x, gps_y, current_alt, telemetry_roll, telemetry_pitch, telemetry_yaw])

            # Trigger physical fluid release mechanism if queued
            if self.payload_volume_to_release > 0:
                self.discharge_active = True
                self.setCustomData(str(self.payload_volume_to_release))
                self.payload_volume_to_release = 0
            else:
                self.setCustomData(str(0))

            if current_alt > cruise_altitude - 1:
                # Flight Navigation Decision Tree
                if self.getTime() - timestamp_nav_update > 0.1:
                    if self.pixel_coords_fire:
                        # Target is visually acquired: perform overhead alignment
                        yaw_adjustment, pitch_adjustment = self.execute_overhead_alignment()
                    else:
                        live_fires = self.get_cords()
                        if live_fires:
                            # Intercept target locations recorded by supervisor map
                            yaw_adjustment, pitch_adjustment = self.redirect_towards_hazard(live_fires)
                        else:
                            # Default back to normal structural tracking route loops
                            yaw_adjustment, pitch_adjustment = self.compute_navigation_vectors(configured_waypoints)
                    timestamp_nav_update = self.getTime()

                # Operational Image Sweep Run
                if self.getTime() - timestamp_scan_update > 1:
                    if not self.discharge_active:
                        self.pixel_coords_fire = self.run_image_hazard_scan()
                    timestamp_scan_update = self.getTime()

                if not self.discharge_active:
                    timestamp_discharge_lock = self.getTime()
                if self.getTime() - timestamp_discharge_lock > 15:  # Mask drop window to avoid self-detecting water vapor
                    self.discharge_active = False

                # Proximity sensor collision check override
                is_obstructed, reactive_yaw, reactive_pitch = self.evaluate_proximity_hazards()
                if is_obstructed:
                    yaw_adjustment = reactive_yaw
                    pitch_adjustment = reactive_pitch

            # Mixing control loops to calculate dynamic system forces
            mix_roll = self.GAIN_ROLL_P * \
                limit_value(telemetry_roll, -1, 1) + accel_roll + roll_adjustment
            mix_pitch = self.GAIN_PITCH_P * \
                limit_value(telemetry_pitch, -1, 1) + accel_pitch + pitch_adjustment
            mix_yaw = yaw_adjustment
            
            clamped_altitude_delta = limit_value(
                cruise_altitude - current_alt + self.HEIGHT_STABILIZATION_BIAS, -1, 1)
            mix_vertical = self.GAIN_ALTITUDE_P * pow(clamped_altitude_delta, 3.0)

            # Consolidating actuator metrics
            force_fl = self.BASE_VERTICAL_FORCE + mix_vertical - mix_yaw + mix_pitch - mix_roll
            force_fr = self.BASE_VERTICAL_FORCE + mix_vertical + mix_yaw + mix_pitch + mix_roll
            force_rl = self.BASE_VERTICAL_FORCE + mix_vertical + mix_yaw - mix_pitch - mix_roll
            force_rr = self.BASE_VERTICAL_FORCE + mix_vertical - mix_yaw - mix_pitch + mix_roll

            self.actuator_fl.setVelocity(force_fl)
            self.actuator_fr.setVelocity(-force_fr)
            self.actuator_rl.setVelocity(-force_rl)
            self.actuator_rr.setVelocity(force_rr)


uav_instance = Mavic()
uav_instance.run()