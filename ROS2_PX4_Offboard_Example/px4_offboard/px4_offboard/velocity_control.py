#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Braden Wagstaff"
__contact__ = "braden@arkelectron.com"

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition
from geometry_msgs.msg import Twist, Vector3, Pose
from math import pi
from std_msgs.msg import Bool


class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        csv_file = "/home/kangle/ros2_px4_offboard_example_ws/src/ROS2_PX4_Offboard_Example/px4_offboard/px4_offboard/traj_files/trajectory_1.csv"

        self.load_csv(csv_file)

        #Create subscriptions
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        
        self.offboard_velocity_sub = self.create_subscription(
            Twist,
            '/offboard_velocity_cmd',
            self.offboard_velocity_callback,
            qos_profile)
        
        self.offboard_position_sub = self.create_subscription(
            Pose,
            '/offboard_position_cmd',
            self.offboard_position_callback,
            qos_profile)
        
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile)
        
        self.my_bool_sub = self.create_subscription(
            Bool,
            '/arm_message',
            self.arm_message_callback,
            qos_profile)
        
        self.my_bool_sub = self.create_subscription(
            Bool,
            '/confirm_message',
            self.confirm_message_callback,
            qos_profile)
        
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)


        #Create publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_velocity = self.create_publisher(Twist, '/fmu/in/setpoint_velocity/cmd_vel_unstamped', qos_profile)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)

        
        #creates callback function for the arm timer
        # period is arbitrary, just should be more than 2Hz
        arm_timer_period = .1 # seconds
        self.arm_timer_ = self.create_timer(arm_timer_period, self.arm_timer_callback)

        # creates callback function for the command loop
        # period is arbitrary, just should be more than 2Hz. Because live controls rely on this, a higher frequency is recommended
        # commands in cmdloop_callback won't be executed if the vehicle is not in offboard mode
        timer_period = 0.02  # seconds
        self.timer_period = timer_period
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.velocity = Vector3()
        self.position = Vector3()
        self.position_mode = True
        self.yaw = 0.0  #yaw value we send as command
        self.trueYaw = 0.0  #current yaw value of drone
        self.offboardMode = False
        self.flightCheck = False
        self.myCnt = 0
        self.minor_steps = 0
        self.max_minor_steps = int(10.0 / timer_period)
        self.arm_message = False
        self.failsafe = False
        self.confirm = False
        self.test = False
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])

        self.target_position = Vector3()
        self.target_position.x = 0.0
        self.target_position.y = 0.0
        self.target_position.z = 0.0
        self.target_velocity = Vector3()
        self.target_velocity.x = float('nan')
        self.target_velocity.y = float('nan')
        self.target_velocity.z = float('nan')
        self.phase_one = True
        self.counter = 0
        ########################################### CONTROLS EFFECTIVE SPEED OF DRONE - SET TO VERY VERY SLOW
        self.time_period = 0.2
        ########################################### CONTROLS EFFECTIVE SPEED OF DRONE - SET TO VERY VERY SLOW
        self.time_steps = int(self.time_period / timer_period)
        self.current_time_steps = 0

        self.prev_position = Vector3()
        self.prev_position.x = 0.0
        self.prev_position.y = 0.0
        self.prev_position.z = 2.0

        #states with corresponding callback functions that run once when state switches
        self.states = {
            "IDLE": self.state_init,
            "ARMING": self.state_arming,
            "TAKEOFF": self.state_takeoff,
            "LOITER": self.state_loiter,
            "OFFBOARD": self.state_offboard
        }

        self.traj_states = [
            "WAITING",
            "MOVE_TO_START",
            "TRAJ_1",
            "TRAJ_2",
            "DONE"
        ]
        
        self.current_traj_state = "WAITING"

        self.current_state = "IDLE"
        self.last_state = self.current_state

        self.step_counter = 0
        self.csv_index = 0
        self.starting_position_steps = 1000
    
    def load_csv(self, csv_file):
        try:
            self.data = pd.read_csv(csv_file)
            self.get_logger().info(f"CSV data loaded successfully:\n{self.data.head()}")
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV file: {e}")


    def arm_message_callback(self, msg):
        self.arm_message = msg.data
        self.get_logger().info(f"Arm Message: {self.arm_message}")
    
    def confirm_message_callback(self, msg):
        self.confirm = msg.data
        self.get_logger().info(f"Confirm Message: {self.confirm}")

    #callback function that arms, takes off, and switches to offboard mode
    #implements a finite state machine
    def arm_timer_callback(self):
        self.get_logger().info(f"Current State: {self.current_state}, Flight Check: {self.flightCheck}, Arm State: {self.arm_state}, Failsafe: {self.failsafe}, Nav State: {self.nav_state}")

        match self.current_state:
            case "IDLE":
                if(self.flightCheck and self.arm_message == True):
                    self.current_state = "ARMING"
                    self.get_logger().info(f"Arming")

            case "ARMING":
                if(not(self.flightCheck)):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Arming, Flight Check Failed")
                elif(self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self.myCnt > 10):
                    self.current_state = "TAKEOFF"
                    self.get_logger().info(f"Arming, Takeoff")
                self.arm() #send arm command

            case "TAKEOFF":
                if(not(self.flightCheck)):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Takeoff, Flight Check Failed")
                elif(self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF):
                    self.current_state = "LOITER"
                    self.get_logger().info(f"Takeoff, Loiter")
                self.arm() #send arm command
                self.take_off() #send takeoff command

            # waits in this state while taking off, and the 
            # moment VehicleStatus switches to Loiter state it will switch to offboard
            case "LOITER": 
                self.offboardMode = False
                if(not(self.flightCheck)):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Loiter, Flight Check Failed")
                elif(self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER) or self.confirm:
                    self.confirm = False
                    self.current_state = "OFFBOARD"
                    self.get_logger().info(f"Loiter, Offboard")
                self.arm()

            case "OFFBOARD":
                if(not(self.flightCheck) or self.arm_state == VehicleStatus.ARMING_STATE_DISARMED or self.failsafe == True):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Offboard, Flight Check Failed")
                self.state_offboard()

                if self.nav_state == VehicleStatus.NAVIGATION_STATE_POSCTL:
                    self.current_state = "LOITER"
                    self.get_logger().info(f"Posctl, Returning to IDLE")

        if(self.arm_state != VehicleStatus.ARMING_STATE_ARMED):
            self.arm_message = False

        if (self.last_state != self.current_state):
            self.last_state = self.current_state
            self.get_logger().info(self.current_state)

        self.myCnt += 1

    def state_init(self):
        self.myCnt = 0

    def state_arming(self):
        self.myCnt = 0
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command send")

    def state_takeoff(self):
        self.myCnt = 0
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1 = 1.0, param7=2.0) # param7 is altitude in meters
        self.get_logger().info("Takeoff command send")

    def state_loiter(self): 
        self.myCnt = 0
        self.get_logger().info("Loiter Status")

    def state_offboard(self):
        self.myCnt = 0
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
        self.offboardMode = True


    

        

    # Arms the vehicle
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command send")

    # Takes off the vehicle to a user specified altitude (meters)
    def take_off(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1 = 1.0, param7=2.0) # param7 is altitude in meters
        self.get_logger().info("Takeoff command send")

    #publishes command to /fmu/in/vehicle_command
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7    # altitude value in takeoff command
        msg.command = command  # command ID
        msg.target_system = 1  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher_.publish(msg)

    #receives and sets vehicle status values 
    def vehicle_status_callback(self, msg):

        if (msg.nav_state != self.nav_state):
            self.get_logger().info(f"NAV_STATUS: {msg.nav_state}")
        
        if (msg.arming_state != self.arm_state):
            self.get_logger().info(f"ARM STATUS: {msg.arming_state}")

        if (msg.failsafe != self.failsafe):
            self.get_logger().info(f"FAILSAFE: {msg.failsafe}")
        
        if (msg.pre_flight_checks_pass != self.flightCheck):
            self.get_logger().info(f"FlightCheck: {msg.pre_flight_checks_pass}")

        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.failsafe = msg.failsafe
        self.flightCheck = msg.pre_flight_checks_pass

    def offboard_position_callback(self, msg):
        self.position_mode = True
        self.position.x = - msg.position.y
        self.position.y = msg.position.x
        self.position.z = - msg.position.z


    #receives Twist commands from Teleop and converts NED -> FLU
    def offboard_velocity_callback(self, msg):
        self.position_mode = False
        #implements NED -> FLU Transformation
        self.velocity.x = -msg.linear.y
        self.velocity.y = msg.linear.x
        self.velocity.z = -msg.linear.z
        self.yaw = msg.angular.z

        # X (FLU) is -Y (NED)
        self.velocity.x = -msg.linear.y

        # Y (FLU) is X (NED)
        self.velocity.y = msg.linear.x

        # Z (FLU) is -Z (NED)
        self.velocity.z = -msg.linear.z

        # A conversion for angular z is done in the attitude_callback function(it's the '-' in front of self.trueYaw)
        self.yaw = msg.angular.z

    #receives current trajectory values from drone and grabs the yaw value of the orientation
    def attitude_callback(self, msg):
        orientation_q = msg.q

        #trueYaw is the drones current yaw value
        self.trueYaw = -(np.arctan2(2.0*(orientation_q[3]*orientation_q[0] + orientation_q[1]*orientation_q[2]), 
                                  1.0 - 2.0*(orientation_q[0]*orientation_q[0] + orientation_q[1]*orientation_q[1])))
    
    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation 
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz
    
    def has_reached_position(self, target_position, threshold=0.4):
        
        current_position = self.vehicle_local_position
        distance = np.sqrt(
            (current_position[0] - target_position.x) ** 2 +
            (current_position[1] - target_position.y) ** 2 +
            (-current_position[2] - target_position.z) ** 2
        )
        self.get_logger().info(f"Aim {target_position}, current: {current_position}, distance: {distance}, time: {self.time_steps}")
        return distance < threshold
    
    def set_target_pos(self, position, update_velocity=True):
        
        self.target_position.x = float(position['x'] - 1.9)    # Puts bar at 0
        self.target_position.y = float(position['y'])    # Puts bar at 0
        self.target_position.z = - float(position['z']) + 1.0 # Puts the bar at 2.7
        self.get_logger().info(f"Setting target position {self.target_position} | {position['x']} | {position['y']}")
        self.current_time_steps = 0

        if update_velocity:
            self.update_target_velocity()

    
    def update_target_velocity(self):

        time = (self.time_steps - self.current_time_steps) * self.timer_period

        distance_x = self.target_position.x - self.vehicle_local_position[0]
        distance_y = self.target_position.y - self.vehicle_local_position[1]
        distance_z = self.target_position.z + self.vehicle_local_position[2]

        if time is not None and time >= (self.time_period / 3):
            self.target_velocity.x = distance_x / time
            self.target_velocity.y = distance_y / time
            self.target_velocity.z = distance_z / time
        else:
            self.target_velocity.x = float('nan')
            self.target_velocity.y = float('nan')
            self.target_velocity.z = float('nan')
        
        return time

        
    #publishes offboard control modes and velocity as trajectory setpoints
    def cmdloop_callback(self):
        if(self.offboardMode == True):
            # Publish offboard control modes
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            offboard_msg.position = True
            offboard_msg.velocity = True
            offboard_msg.acceleration = False
            self.publisher_offboard_mode.publish(offboard_msg)            

            # Compute velocity in the world frame
            cos_yaw = np.cos(self.trueYaw)
            sin_yaw = np.sin(self.trueYaw)
            velocity_world_x = (self.velocity.x * cos_yaw - self.velocity.y * sin_yaw)
            velocity_world_y = (self.velocity.x * sin_yaw + self.velocity.y * cos_yaw)

            # Create and publish TrajectorySetpoint message with NaN values for position and acceleration
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            trajectory_msg.acceleration[0] = float('nan')
            trajectory_msg.acceleration[1] = float('nan')
            trajectory_msg.acceleration[2] = float('nan')
            trajectory_msg.yaw = float('nan')
            trajectory_msg.yawspeed = float('nan')

            if self.current_traj_state == "WAITING":
                if self.confirm:
                    self.current_traj_state = "MOVE_TO_START"
                    self.confirm = False
                    self.set_target_pos(self.data.iloc[0], update_velocity=False)
                return
            

            elif self.current_traj_state == "MOVE_TO_START":
                if self.confirm: # Confirmed at starting position TODO: Has reached target position
                    self.current_traj_state = "TRAJ_1"
                    self.confirm = False
            
            elif self.current_traj_state == "TRAJ_1":
                time = self.update_target_velocity()
                self.current_time_steps += 1
                if time < 0:
                    self.get_logger().error(f"Out of time to reach next position time:{time}")
                if self.has_reached_position(self.target_position):
                    if self.test:
                        if self.confirm:
                            self.confirm = False
                            self.csv_index += 1
                            if self.csv_index >= len(self.data):
                                self.current_traj_state = "DONE"
                                return
                            self.set_target_pos(self.data.iloc[self.csv_index])
                            self.phase_one = bool(self.data.iloc[self.csv_index]['h'] == False)

                    elif self.phase_one:
                        self.csv_index += 1
                        self.set_target_pos(self.data.iloc[self.csv_index])
                        self.phase_one = bool(self.data.iloc[self.csv_index]['h'] == False)

                    elif self.confirm:
                        self.current_traj_state = "TRAJ_2"
                        self.confirm = False
            
            elif self.current_traj_state == "TRAJ_2":
                time = self.update_target_velocity()
                self.current_time_steps += 1
                if time < 0:
                    self.get_logger().error(f"Out of time to reach next position time:{time}")

                if self.has_reached_position(self.target_position):
                    self.csv_index += 1

                    if self.csv_index >= len(self.data):
                        self.current_traj_state = "DONE"
                    else:
                        self.set_target_pos(self.data.iloc[self.csv_index])
            
            elif self.current_traj_state == "DONE":
                self.get_logger().info("Completed the trajectory!")
                return
            
            else:
                self.get_logger().info("Ended the trajectory!")
            
            trajectory_msg.position[0] = self.target_position.x
            trajectory_msg.position[1] = self.target_position.y
            trajectory_msg.position[2] = self.target_position.z
            trajectory_msg.velocity[0] = self.target_velocity.x
            trajectory_msg.velocity[1] = self.target_velocity.y
            trajectory_msg.velocity[2] = self.target_velocity.z

            self.get_logger().info(f"MEssage Sending: {trajectory_msg}")

            self.publisher_trajectory.publish(trajectory_msg)
        else:
            self.get_logger().info("Not in offboard mode")



def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()