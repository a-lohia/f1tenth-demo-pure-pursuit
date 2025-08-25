#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
from transforms3d.euler import quat2euler

import numpy as np
import os, csv
from time import gmtime, strftime
from nav_msgs.msg import Odometry

class Waypoint(Node):
    def __init__(self):
        super().__init__('waypoint_node')

        POSE_TOPIC = '/pf/pose/odom'
        print(os.getcwd())

        self.fp = strftime('/home/team5/f1tenth_ws/src/pure_pursuit/waypoints/waypoints.csv')

        pose_listener = self.create_subscription(Odometry, POSE_TOPIC, self.pose_callback, 10)
        timer = self.create_timer(0.1, self.timer_callback)

        self.euler = None
        self.speed = None
        self.x = None
        self.y = None

    def pose_callback(self, pose_msg):
        self.x = pose_msg.pose.pose.position.x
        self.y = pose_msg.pose.pose.position.y

        quaternion = np.array([pose_msg.pose.pose.orientation.x, 
                            pose_msg.pose.pose.orientation.y, 
                            pose_msg.pose.pose.orientation.z, 
                            pose_msg.pose.pose.orientation.w])

        self.euler = quat2euler(quaternion)
        self.speed = np.linalg.norm(np.array([pose_msg.twist.twist.linear.x, 
                                         pose_msg.twist.twist.linear.y, 
                                         pose_msg.twist.twist.linear.z]),2)
    
    def timer_callback(self):
        if (self.euler is not None):
            with open(self.fp, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow([self.x, self.y, self.euler[2], self.speed])


def main(args=None):
    rclpy.init(args=args)
    print("Waypoint Listener Initialized")
    waypoint_node = Waypoint()
    rclpy.spin(waypoint_node)

    waypoint_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
