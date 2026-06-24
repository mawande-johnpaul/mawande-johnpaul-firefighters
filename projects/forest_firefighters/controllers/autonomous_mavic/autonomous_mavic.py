from controller import Robot
import sys
import random
import optparse
import re

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
try:
    import cv2
except ImportError:
    sys.exit("Warning: 'cv2' module not found.")


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class Mavic (Robot):
    K_VERTICAL_THRUST = 68.5  
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        
    K_ROLL_P = 50.0           
    K_PITCH_P = 30.0          

    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -0.6
    target_precision = 1.0

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        self.water_to_drop = 0

        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(1.55)  
        
        for motor in [self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6*[0]  
        self.target_position = [0, 0, 0]
        self.target_index = 0

        # Tracking memory states
        self.tracking_fire = False
        self.WaterDropStatus = False
        self.last_seen_time = 0.0
        self.tracking_patience = 2.0  # Seconds to hold target after losing visual contact

    def get_image_from_camera(self):
        width, height = self.camera.getWidth(), self.camera.getHeight()
        raw_bytes = self.camera.getImage()
        img = np.frombuffer(raw_bytes, np.uint8).reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(img, 1)

    def set_position(self, pos):
        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=True):
        if self.target_position[0:2] == [0, 0]:  
            self.target_position[0:2] = waypoints[0]

        if not self.tracking_fire:
            if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):
                self.target_index += 1
                if self.target_index > len(waypoints)-1:
                    self.target_index = 0
                self.target_position[0:2] = waypoints[self.target_index]
                if verbose_target:
                    print(f"[{self.getName()}] Waypoint reached! Heading to: {self.target_position[0:2]}")

        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        
        angle_left = self.target_position[2] - self.current_pose[5]
        angle_left = (angle_left + 2*np.pi) % (2*np.pi)
        if (angle_left > np.pi):
            angle_left -= 2*np.pi

        yaw_disturbance = self.MAX_YAW_DISTURBANCE*angle_left/(2*np.pi)
        pitch_disturbance = clamp(np.log10(max(abs(angle_left), 1e-6)), self.MAX_PITCH_DISTURBANCE, 0.1)

        # Ease off both forward thrust and heading correction on final approach.
        # Bearing-to-target gets wildly sensitive once you're close -- a tiny
        # sideways drift swings the desired heading by a huge angle -- so without
        # damping both terms toward zero, the drone keeps nudging forward *and*
        # spinning to chase that noisy bearing, which looks like it orbiting or
        # dancing back and forth instead of settling over the target.
        distance_to_target = np.hypot(self.target_position[0] - self.current_pose[0],
                                       self.target_position[1] - self.current_pose[1])
        slowdown_radius = 6.0
        ease = clamp(distance_to_target / slowdown_radius, 0.0, 1.0)
        pitch_disturbance *= ease
        yaw_disturbance *= ease

        if self.tracking_fire and all([abs(x1 - x2) < 1.2 for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):
            self.water_to_drop = 15

        return yaw_disturbance, pitch_disturbance

    def fire_detection(self, verbose=True):
        img = self.get_image_from_camera()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        smoke_lower = np.array([0, 0, 168])
        smoke_upper = np.array([172, 111, 255])
        mask_fire = cv2.inRange(hsv, smoke_lower, smoke_upper)
        fire_ratio = np.round((cv2.countNonZero(mask_fire))/(img.size/3)*100, 2)

        coord_fire = None  
        if fire_ratio > 0.15:  
            contours, _ = cv2.findContours(image=mask_fire, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            contours_poly = [None]*len(contours)
            centers = [None]*len(contours)
            radius = [None]*len(contours)
            radius_max = 0
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
                if radius[i] > 3 and radius[i] > radius_max:
                    coord_fire = centers[i]
                    radius_max = radius[i]

        return coord_fire

    def run(self):
        t1 = self.getTime()
        t2 = self.getTime()
        t3 = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--patrol_coords", default="11 11, 11 21, 21 21,21 11",
                              help="Specify the patrol coordinates in the format [x1 y1, x2 y2, ...]")
        opt_parser.add_option("--target_altitude", default=42,
                              type=float, help="target altitude of the robot in meters")
        options, _ = opt_parser.parse_args()

        point_list = options.patrol_coords.split(',')
        number_of_waypoints = len(point_list)
        waypoints = []
        for i in range(0, number_of_waypoints):
            waypoints.append([])
            waypoints[i].append(float(point_list[i].split()[0]))
            waypoints[i].append(float(point_list[i].split()[1]))

        try:
            drone_digits = re.findall(r'\d+', self.getName())
            drone_id = int(drone_digits[0]) if drone_digits else 0
        except Exception:
            drone_id = 0
        
        target_altitude = max(37.0, options.target_altitude - (drone_id * 3.0))

        while self.step(self.time_step) != -1:
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            Xpos, Ypos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([Xpos, Ypos, altitude, roll, pitch, yaw])

            if self.water_to_drop > 0:
                self.WaterDropStatus = True
                self.setCustomData(str(self.water_to_drop))
                self.water_to_drop = 0
                self.tracking_fire = False  
            else:
                self.setCustomData(str(0))

            if altitude > target_altitude - 1:
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance = self.move_to_target(waypoints)
                    t1 = self.getTime()

                if self.getTime() - t2 > 0.15:
                    if not self.WaterDropStatus:
                        img_coord = self.fire_detection()
                        
                        if img_coord is not None:
                            # Update visual memory confirmation timestamp
                            self.last_seen_time = self.getTime()
                            
                            x_img, y_img = img_coord
                            res_x, res_y = self.camera.getWidth(), self.camera.getHeight()
                            
                            if not self.tracking_fire:
                                print(f"[{self.getName()}] FIRE DETECTED! Lock acquired.")
                                self.tracking_fire = True
                            
                            fov_factor = 1.05  
                            dx_body = -((y_img - res_y / 2) / res_y) * altitude * fov_factor
                            dy_body = ((x_img - res_x / 2) / res_x) * altitude * fov_factor
                            
                            fire_world_x = Xpos + (dx_body * np.cos(yaw) - dy_body * np.sin(yaw))
                            fire_world_y = Ypos + (dx_body * np.sin(yaw) + dy_body * np.cos(yaw))
                            
                            self.target_position[0:2] = [fire_world_x, fire_world_y]
                        else:
                            # FIX: If fire goes out of frame, hold last known target position until patience expires
                            if self.tracking_fire:
                                if self.getTime() - self.last_seen_time > self.tracking_patience:
                                    print(f"[{self.getName()}] Fire lost permanently. Resuming patrol to index: {self.target_index}")
                                    self.tracking_fire = False
                                    self.target_position[0:2] = waypoints[self.target_index]
                                # else: pass (Keep moving to the last estimated fire coordinate)
                                
                    t2 = self.getTime()

                if not self.WaterDropStatus:
                    t3 = self.getTime()
                if self.getTime() - t3 > 15:  
                    self.WaterDropStatus = False

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)


if __name__ == "__main__":
    robot = Mavic()
    robot.run()