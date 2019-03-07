
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704



class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, 
                wheel_base, wheel_radius, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        Kp = 0.3
        Ki = 0.1
        Kd = 0.0
        min_throttle = 0.0 # minimum throttle value
        max_throttle = 0.2 # maximum throttle value
        self.throttle_controller = PID(Kp, Ki, Kd, min_throttle, max_throttle)

        tau = 0.5 # 1/(2*pi*tau) = cutoff frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.steer_ratio = steer_ratio
        # self.max_lat_accel = max_lat_accel
        # self.max_steer_angle = max_steer_angle

        self.last_time = rospy.get_time()
        self.last_vel = None


    

    def control(self, current_vel, dbw_status, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_status:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        
        current_vel = self.vel_lpf.filt(current_vel)

        # rospy.logwarn("Angular vel: {}".format(angular_vel))
        # rospy.logwarn("Target velocity: {}".format(linear_vel))
        # rospy.logwarn("Current velocity: {}".format(current_vel))
        # rospy.logwarn("Filtered velocity: {}".format(self.vel_lpf.get()))

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0
            brake = 400
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
