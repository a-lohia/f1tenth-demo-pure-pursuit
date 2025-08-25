#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import csv
import numpy as np
from scipy import interpolate 
from scipy.interpolate import CubicSpline
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PointStamped
import os


class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')

        self.subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_callback,
            10
        )

        self.publisher = self.create_publisher(MarkerArray, 'waypoints_marker_array', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # one-time timer


        self.csv_path = '/home/team5/f1tenth_ws/src/pure_pursuit/waypoints/waypoints_modified.csv'  # Change to full path or use ROS param
        # self.csv_path = '/home/team5/f1tenth_ws/src/pure_pursuit/waypoints/waypoints.csv'  # Change to full path or use ROS param
        self.waypoints = self.load_waypoints()

    def load_waypoints(self):
        waypoints = []
        with open(self.csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    x, y = float(row[0]), float(row[1])
                    waypoints.append((x, y))
        waypoints = np.array(waypoints)
        self.get_logger().info(f'Total waypoints loaded: {waypoints.shape}')
        return waypoints

    def linear_interpolation(self, p1, p2, num_points):
        x1, y1 = p1
        x2, y2 = p2
        x_values = np.linspace(x1, x2, num_points)
        y_values = np.linspace(y1, y2, num_points)
        return np.column_stack((x_values, y_values))
    
    def clicked_callback(self, msg): 
        spacing = None
        clicked_point = np.array([msg.point.x, msg.point.y])

        nearest_index, dist = self.find_nearest_waypoint(clicked_point[0], clicked_point[1])
        offset = 4

        pre_idx = (nearest_index - offset) % len(self.waypoints)
        next_idx = (nearest_index + offset) % len(self.waypoints)
        
        pre_point = self.waypoints[pre_idx]
        next_point = self.waypoints[next_idx]

        # Perform linear interpolation
        interp1 = self.linear_interpolation(pre_point, clicked_point, offset + 1)
        interp2 = self.linear_interpolation(clicked_point, next_point, offset + 1)

        # Combine the interpolated points, excluding the duplicate clicked_point
        new_points = np.vstack((interp1[:-1], interp2))

        print(f'pre: {pre_idx}, next: {next_idx}')
        print(f'Pre Point: {pre_point}, Clicked Point: {clicked_point}, Next Point: {next_point}')
        print(f'Interpolated Points: \n{new_points}')

        # Replace the waypoints from pre_idx to next_idx with new_points
        if pre_idx < next_idx:
            self.waypoints = np.vstack((self.waypoints[:pre_idx], new_points, self.waypoints[next_idx + 1:]))
        else:
            # Handle wrap-around case
            self.waypoints = np.vstack((new_points, self.waypoints[next_idx + 1:pre_idx]))

        # Resample the trajectory
        self.waypoints = self.resample_loop_trajectory(self.waypoints, spacing=spacing)



    def find_nearest_waypoint(self, x, y):
        waypoints = np.array(self.waypoints)
        distances = np.linalg.norm(waypoints - np.array([x, y]), axis=1)
        nearest_index = np.argmin(distances)
        return nearest_index, distances[nearest_index]
    

    def resample_loop_trajectory(self, traj, spacing=None):
        # unpack and copy
        xs = traj[:, 0].copy()
        ys = traj[:, 1].copy()

        # 1) remove consecutive duplicates
        #    compute dx,dy and keep only where they move
        deltas = np.vstack((np.diff(xs), np.diff(ys))).T
        nonzero = np.any(np.abs(deltas) > 1e-6, axis=1)
        mask = np.concatenate(([True], nonzero))
        xs = xs[mask]
        ys = ys[mask]

        # 2) close the loop if needed
        if not np.allclose([xs[0], ys[0]], [xs[-1], ys[-1]]):
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])

        # 3) compute cumulative arc‐length
        ds = np.hypot(np.diff(xs), np.diff(ys))
        cum = np.insert(np.cumsum(ds), 0, 0)
        total_length = cum[-1]

        # 4) pick your spacing
        if spacing is None:
            spacing = total_length / len(ds)

        s_new = np.arange(0, total_length, spacing)

        # 5) build the periodic splines
        cs_x = CubicSpline(cum, xs, bc_type='periodic')
        cs_y = CubicSpline(cum, ys, bc_type='periodic')

        x_new = cs_x(s_new)
        y_new = cs_y(s_new)

        return np.column_stack((x_new, y_new))
    #     x, y = traj[:, 0], traj[:, 1]

    #     # Close the loop if not already closed
    #     if not np.allclose(traj[0], traj[-1]):
    #         x = np.append(x, x[0])
    #         y = np.append(y, y[0])

    #     # Compute arc length (cumulative distance)
    #     distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    #     cumulative = np.insert(np.cumsum(distances), 0, 0)
    #     total_length = cumulative[-1]

    #     # Estimate spacing if not given
    #     if spacing is None:
    #         spacing = total_length / len(distances)

    #     # Fit periodic spline
    #     tck, _ = interpolate.splprep([x, y], u=cumulative, s=0, per=True)

    #     # Compute new sampling distances
    #     new_distances = np.arange(0, total_length, spacing)

    #     # Evaluate the spline
    #     x_new, y_new = interpolate.splev(new_distances, tck)

    #     return np.stack([x_new, y_new], axis=1)

    

    def timer_callback(self):
        marker_array = MarkerArray()

        for idx, (x, y) in enumerate(self.waypoints):
            marker = Marker()
            marker.header = Header(frame_id='map')
            marker.ns = 'waypoints'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # green

            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)
        # self.get_logger().info(f'Published {len(marker_array.markers)} markers.')

    def save_csv(self):
        print("saving_points")
        # self.waypoints = np.column_stack(self.waypoints, np.zeros_like(self.waypoints.shape[0]))
        path_new = os.path.join(os.path.dirname(self.csv_path), "waypoints_modified.csv")
        np.savetxt(path_new, self.waypoints, delimiter=',')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
