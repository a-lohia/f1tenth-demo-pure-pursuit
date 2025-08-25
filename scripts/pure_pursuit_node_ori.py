#!/usr/bin/env python3
# ros imports
import rclpy
from rclpy.node import Node
import tf2_ros, transforms3d

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# standard imports
import numpy as np
import csv
import time

# msg imports
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import MarkerArray, Marker
from builtin_interfaces.msg import Duration
# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        DRIVE_TOPIC = "/drive"
        POSE_TOPIC = "/pf/pose/odom"
        MARKER_TOPIC = "/visualization_marker_array"
        GOAL_TOPIC = "/visualization_marker"

        # Pure Pursuit Parameters
        self.lookahead = .75 # 1 meter for now
        self.wheelbase = .3302
        self.velocity = 1.5

        # Waypoint Setup
        self.marker_id = 0
        self.waypoints = []
        self.current_waypoint = None
        self.goal_point = None

        # tf2
        self.from_frame = 'map'
        self.to_frame = 'laser'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # create ROS subscribers and publishers
        self.drive = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.lidar = self.create_subscription(Odometry, POSE_TOPIC, self.pose_callback, 10)
        self.markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.test = self.create_publisher(MarkerArray, "/v_m_test", 10)
        self.goal = self.create_publisher(Marker, GOAL_TOPIC, 10)
        self.goal_tf = self.create_publisher(Marker, GOAL_TOPIC+"1", 10)

        time.sleep(2)

        self.marker_init()
        print('published markers')

    def create_marker(self, row, color, m_id=-1, scale=.2, inf=True):
        # row format ['x', 'y', '3rd euler angle', 'speed']
        msg = Marker()
        msg.header.frame_id = "map"
        # msg.header.stamp = self.get_clock().now()
        if not inf:
            msg.lifetime = Duration()
            msg.lifetime.nanosec = int(1e9/4)
        else:
            pass
            # msg.header.stamp.sec = 0
            # msg.header.stamp.nanosec = 0
        msg.ns = "my_namespace"
        msg.id = self.marker_id if m_id==-1 else m_id
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
        msg.pose.position.x = float(row[0])
        msg.pose.position.y = float(row[1])
        msg.pose.position.z = .2

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        msg.scale.x = scale
        msg.scale.y = scale
        msg.scale.z = scale

        r, g, b = color
        msg.color.r = r
        msg.color.g = g
        msg.color.b = b
        msg.color.a = 1.0

        self.marker_id += 1

        return msg

    def marker_init(self):
        with open("/home/team6/f1tenth_ws/src/pure_pursuit/waypoints/fp2.csv", 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            marker_id = 0
            arr = MarkerArray()
            for row in reader:
                self.waypoints.append([float(row[0]), float(row[1])])
                msg = self.create_marker(row, (1.0, 0.0, 0.0))
                arr.markers.append(msg)
            self.markers.publish(arr)
        self.waypoints = np.stack(self.waypoints)
    
    def pose_callback(self, pose_msg):

        quat2npy = lambda o: np.array([o.w, o.x, o.y, o.z])
        xyz2npy = lambda v: np.array([v.x, v.y, v.z])
        
        x, y, z = xyz2npy(pose_msg.pose.pose.position)

        pose = np.array([x, y])

        if self.current_waypoint is None:
            for i in range(self.waypoints.shape[0]):  # TODO chnage this to find closest point and start from there
                i_x, i_y = self.waypoints[i]
                if (np.linalg.norm([i_x - x, i_y-y]) - self.lookahead) < 0:
                    continue
                else:
                    self.current_waypoint = i
                    break
            self.goal_point = np.array([self.waypoints[self.current_waypoint][0], self.waypoints[self.current_waypoint][1]])    
        
        # Update the goal point if less than self.lookahead
        if np.linalg.norm(self.goal_point - pose) < self.lookahead:
            while (np.linalg.norm(self.goal_point - pose) < self.lookahead):
                # if self.current_waypoint < self.waypoints.shape[0]-1:
                self.current_waypoint += 1
                self.current_waypoint %= self.waypoints.shape[0]
                # elif self.current_waypoint == self.waypoints.shape[0]:
                #    self.current_waypoint = 0  # loop around if at the end
                self.goal_point = np.array([self.waypoints[self.current_waypoint][0], self.waypoints[self.current_waypoint][1]])

        # Show the goal point in RViz
        self.goal.publish(self.create_marker([self.goal_point[0], self.goal_point[1], 0, 0], (1.0, 1.0, 1.0), scale=.5, inf=False))

        # TODO: transform goal point to vehicle frame of reference

        try:
            t = self.tf_buffer.lookup_transform(
                self.to_frame,
                self.from_frame,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.to_frame} to {self.from_frame}: {ex}')
            return

        rot_mat = transforms3d.quaternions.quat2mat(quat2npy(t.transform.rotation))

        goal_hom = np.array([self.goal_point[0], self.goal_point[1], 0, 1])
        transform = np.zeros(shape=(4,4))
        transform[:3, :3] = rot_mat
        transform[:3, 3] = xyz2npy(t.transform.translation)
        transform[3, 3] = 1

        transformed = transform @ goal_hom
        x_n, y_n, _, _ = transformed

        self.goal_tf.publish(self.create_marker([x_n, y_n, 0, 0], (.0, 1.0, .0), scale=.5, inf=False))

        # TODO: calculate curvature/steering angle
        sign = 1 if y_n > 0 else -1
        gamma = sign * 2 * np.abs(y_n)/ (np.linalg.norm([x_n, y_n])**2)

        # TODO: publish drive message, don't forget to limit the steering angle.
        steering_angle = np.arctan(self.wheelbase * gamma)
        if steering_angle > (np.pi/4): steering_angle = np.pi /4
        if steering_angle < -1 * (np.pi/4): steering_angle = -1 * np.pi / 4

        # print(f"transformed goal point ({x_n, y_n}) s_angle {steering_angle}")

        out = AckermannDriveStamped()
        out.drive.steering_angle = steering_angle
        out.drive.speed = self.velocity
        # print(x_n, y_n, steering_angle)
        self.drive.publish(out)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
