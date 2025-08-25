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
from geometry_msgs.msg import PointStamped
# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        DRIVE_TOPIC = "/drive"
        POSE_TOPIC = "/pf/pose/odom"
        MARKER_TOPIC = "/pure_pursuit/visualization_marker_array"
        GOAL_TOPIC = "/pure_pursuit/visualization_marker"

        # Pure Pursuit Parameters
        self.lookahead = .75
        
        self.wheelbase = .3302

        self.declare_parameter('C', 1.2)
        # self.declare_parameter('C', 1.3)
        self.declare_parameter('max_speed', 4.5)
        self.declare_parameter('sec_lookahead', 1.75)
        self.declare_parameter('lookahead', 1.25)
        # self.declare_parameter('sec_lookahead', .6)
        # self.declare_parameter('lookahead', 1.)
        self.declare_parameter('fp', "/home/team5/f1tenth_ws/src/pure_pursuit/waypoints/speed_traj_3_a3.0.csv")

        # Waypoint Setup
        self.marker_id = 0
        self.waypoints = []
        self.current_waypoint_idx = None
        self.goal_point = None
        self.first = True

        # tf2
        self.from_frame = 'map'
        self.to_frame = 'laser'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # create ROS subscribers and publishers
        self.drive = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.lidar = self.create_subscription(Odometry, POSE_TOPIC, self.pose_callback, 1)
        self.markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.test = self.create_publisher(MarkerArray, "/v_m_test", 10)
        # self.goal = self.create_publisher(Marker, GOAL_TOPIC, 10)
        self.goal = self.create_publisher(MarkerArray, GOAL_TOPIC, 10)
        self.goal_tf = self.create_publisher(Marker, GOAL_TOPIC+"1", 10)
        
        # sbuscribe to rrt
        self.rrt_waypoint = None
        self.rrt_sub = self.create_subscription(PointStamped, '/rrt/waypoint', self.rrt_callback, 10)

        time.sleep(2)

        self.marker_init()
        self.steer_to_speed(0)
        print('published markers')

    def create_marker(self, row, color, m_id=-1, scale=.2, inf=True, msg_type=2):
        # row format ['x', 'y', '3rd euler angle', 'speed']
        msg = Marker()
        msg.header.frame_id = "map"
        # msg.header.stamp = self.get_clock().now()
        if not inf:
            msg.lifetime = Duration()
            msg.lifetime.nanosec = int(1e7)
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

        msg.type = msg_type

        r, g, b = color
        msg.color.r = r
        msg.color.g = g
        msg.color.b = b
        msg.color.a = 1.0

        self.marker_id += 1

        return msg

    def marker_init(self):
        counter = 0
        with open(self.get_parameter('fp').value, 'r') as file:
        # with open("/home/team5/f1tenth_ws/src/pure_pursuit/waypoints/penn_day_1_centerline.csv", 'r') as file:
            reader = csv.reader(file, delimiter=',')
            marker_id = 0
            arr = MarkerArray()
            for row in reader:
                # counter += 1
                # if (counter % 15 != 0): 
                #     continue
                self.waypoints.append([float(row[0]), float(row[1]), float(row[2]), None]) #, float(row[3])])
                msg = self.create_marker(row, (.0, 1.0, 1.0), scale = .1)
                arr.markers.append(msg)
            self.markers.publish(arr)
        self.waypoints = np.stack(self.waypoints)
        
    def steer_to_speed(self, theta, C = 1, max_speed = 4.0):
        if theta == 0:
            return max_speed
        return min(C / np.sqrt(np.tan(abs(theta))), max_speed)

    def find_first_goal(self, cur_pose):
        closest = 0
        closest_dist = 100
        lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        # Find goal point
        for i in range(self.waypoints.shape[0]):
            point = np.array(self.waypoints[i])
            if  np.linalg.norm(cur_pose - point[:2]) < lookahead:
                closest_dist = np.linalg.norm(cur_pose - point[:2])
                closest = i
                break

        self.current_waypoint_idx = closest
        self.second_waypoint_idx = closest
        self.goal_point = self.waypoints[self.current_waypoint_idx, :3]
        self.second_goal_point = self.waypoints[self.current_waypoint_idx, :3]

    
    def find_closest_goal(self, cur_pose):
        closest = 0
        closest_dist = 100
        lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        # Find goal point
        for i in range(self.waypoints.shape[0]):
            point = np.array(self.waypoints[i])
            if  np.linalg.norm(cur_pose - point[:2]) < lookahead:
                closest_dist = np.linalg.norm(cur_pose - point[:2])
                closest = i
                break
        
        return self.waypoints[closest, :3]
        

    def update_waypoint(self, pose): 
        # Update the goal point if less than self.lookahead
        lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        if np.linalg.norm(self.goal_point[:2] - pose) < lookahead:
            while (np.linalg.norm(self.goal_point[:2] - pose) < lookahead):
                self.current_waypoint_idx += 1
                self.current_waypoint_idx %= self.waypoints.shape[0]
                self.goal_point = self.waypoints[self.current_waypoint_idx, :3]

    def update_second_waypoint(self, pose): 
        second_lookahead = self.get_parameter('sec_lookahead').get_parameter_value().double_value
        if np.linalg.norm(self.second_goal_point[:2] - pose) < second_lookahead:
            while (np.linalg.norm(self.second_goal_point[:2] - pose) < second_lookahead):
                self.second_waypoint_idx += 1
                self.second_waypoint_idx %= self.waypoints.shape[0]
                self.second_goal_point = self.waypoints[self.second_waypoint_idx, :3]

    def goal_to_steer(self, goal_point): 
        
        quat2npy = lambda o: np.array([o.w, o.x, o.y, o.z])
        xyz2npy = lambda v: np.array([v.x, v.y, v.z])
        
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

        goal_hom = np.array([goal_point[0], goal_point[1], 0, 1])
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
        if steering_angle > (2*np.pi/3): steering_angle = 2*np.pi /3
        if steering_angle < -1 * (2*np.pi/3): steering_angle = -1 * 2*np.pi / 3
        return steering_angle


    def pose_callback(self, pose_msg):
        C = self.get_parameter('C').get_parameter_value().double_value
        max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        quat2npy = lambda o: np.array([o.w, o.x, o.y, o.z])
        xyz2npy = lambda v: np.array([v.x, v.y, v.z])
        
        x, y, z = xyz2npy(pose_msg.pose.pose.position)

        pose = np.array([x, y])

        if self.first:
            self.find_first_goal(pose)
            self.first = False

        if (np.linalg.norm(self.goal_point[:2] - pose) > 1.5):
            self.find_first_goal(pose)

        self.update_waypoint(pose)
        self.update_second_waypoint(pose)

        # Use RRT waypoint if available, otherwise follow original logic
        goal_arr = MarkerArray()

        if (self.rrt_waypoint is None) or (self.rrt_waypoint[2] < 0):
            goal_to_use = self.goal_point
            goal_arr.markers.append(self.create_marker([goal_to_use[0], goal_to_use[1], 0, 0], (0.0, 1.0, 0.0), scale=.25, inf=False, msg_type=1))
        else:
            goal_to_use = self.rrt_waypoint
            goal_arr.markers.append(self.create_marker([goal_to_use[0], goal_to_use[1], 0, 0], (0.0, 0.0, 1.0), scale=.25, inf=False, msg_type=1))

        # Show the goal point in RViz
        goal_arr.markers.append(self.create_marker([self.second_goal_point[0], self.second_goal_point[1], 0, 0], (1.0, 0.0, 0.0), scale=.25, inf=False, msg_type=1))
        self.goal.publish(goal_arr)

        # TODO: transform goal point to vehicle frame of reference
        steering_angle = self.goal_to_steer(goal_to_use)
        if steering_angle is None: 
            return 
        velocity_angle = self.goal_to_steer(self.second_goal_point)
        # print(f"transformed goal point ({x_n, y_n}) s_angle {steering_angle}")

        out = AckermannDriveStamped()
        out.drive.steering_angle = steering_angle
        if (self.rrt_waypoint is None) or (self.rrt_waypoint[2] < 0):
            out.drive.speed = self.steer_to_speed(velocity_angle, C=C, max_speed=max_speed)
            # out.drive.speed = min(self.goal_point[2], self.get_parameter('max_speed').get_parameter_value().double_value)
            # out.drive.speed = min(self.find_closest_goal(pose)[2], self.get_parameter('max_speed').get_parameter_value().double_value)
        else:
            out.drive.speed = 1.5
            
        print(out.drive.speed)
        # print(x_n, y_n, steering_angle)
        self.drive.publish(out)
        
    def rrt_callback(self, msg):
        # Store as np.array for easy use later
        self.rrt_waypoint = np.array([msg.point.x, msg.point.y, msg.point.z])

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()