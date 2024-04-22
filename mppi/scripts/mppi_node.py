#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan




# TODO CHECK: include needed ROS msg type headers and libraries
import math
from jax_mpc.mppi import MPPI
from mppi_env import MPPIEnv
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import time

class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
        
    def new_key(self):
        self.rng, key = jax.random.split(self.rng)
        return key

jRNG = oneLineJaxRNG(1337)


class MPPIPlanner(Node):
    def __init__(self):
        super().__init__('mppi_node')
        self.waypoint_path = "/home/bosky2001/Downloads/levine_raceline.csv"

        self.waypoints = self.load_waypoints(self.waypoint_path)

        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.drive_msg_ = AckermannDriveStamped()


        self.ref_goal_points_ = self.create_publisher(MarkerArray, 'ref_goal_points', 1)
        self.ref_trajectory_ = self.create_publisher(Marker,'ref_trajectory', 1)
        self.opt_trajectory_ = self.create_publisher(Marker,'opt_trajectory', 1)
        self.sampled_trajectory_ = self.create_publisher(Marker,'sampled_trajectory', 1)

        # MPPI params
        self.n_steps = 12
        self.n_samples = 128
        self.jRNG = jRNG
        self.DT = 0.05
        self.is_real = False
        pose_topic = "/pf/viz/inferred_pose" if self.is_real else "/ego_racecar/odom"
        self.pose_sub_ = self.create_subscription(PoseStamped if self.is_real else Odometry, pose_topic, self.pose_callback, 1)
        # self.pose_sub_ = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 1)
        self.mppi_env = MPPIEnv(self.waypoints, self.n_steps, mode = 'ks', DT= self.DT)
        self.mppi = MPPI(n_iterations = 1, n_steps = self.n_steps,
                         n_samples = self.n_samples, a_noise = 1.0, scan = False)
        
        self.a_opt = None
        self.a_cov = None
        self.mppi_state = None
        

        self.norm_param = np.array([0.45, 3.5])
        self.init_state()
        self.ref_goal_points_data = self.viz_ref_points()
    
    def load_waypoints(self, path):
        points = np.loadtxt(path, delimiter=';',skiprows=3, dtype=np.float64)
        #  points = np.loadtxt(path, delimiter=';', skiprows=3)
        points[:, 3] += 0.5*math.pi
        # CONFIGURE DLK
        # self.config.dlk = points[1, 0] - points[0, 0]
        return points
    
    def init_state(self):
        self.mppi_state =  self.mppi.init_state(self.mppi_env.a_shape, self.jRNG.new_key() )
        self.a_opt = self.mppi_state[0]
    
    def pose_callback(self, pose_msg):
        start = time.time()
        self.a_opt = jnp.concatenate([self.a_opt[1:, :],
                    jnp.expand_dims(jnp.zeros((2,)),
                                    axis=0)])  # [n_steps, dim_a]
        
        # da = jax.random.normal(
        #     self.jRNG.new_key(),
        #     shape=(self.n_samples, self.n_steps, self.mppi_env.a_shape)
        # ) 
        a_opt = self.a_opt.copy()
        da = jax.random.truncated_normal(
            self.jRNG.new_key(),
            -jnp.ones_like(a_opt) - a_opt,
            jnp.ones_like(a_opt) - a_opt,
            shape=(self.n_samples, self.n_steps, 2)
        )

        x_state = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        y_state = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y
        curr_orien = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        # x_state = pose_msg.pose.pose.position.x
        # y_state = pose_msg.pose.pose.position.y
        # curr_orien = pose_msg.pose.pose.orientation
        # print(x_state, y_state)
        vel_state = self.drive_msg_.drive.speed
        steer_angle = self.drive_msg_.drive.steering_angle

        
        q = [curr_orien.x, curr_orien.y, curr_orien.z, curr_orien.w]
        yaw_state = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        # print("current yaw", yawp)
        state = np.array([x_state, y_state, steer_angle, vel_state, yaw_state])
        # print(da.shape)

        ref_traj,_ = self.mppi_env.get_refernece_traj(state, target_speed = 3,  vind = 5, speed_factor= 1)
        # print(ref_traj.shape) #[n_steps + 1, 7]

        self.mppi_state, sampled_traj, s_opt, _, _,_ = self.mppi.update(self.mppi_state, self.mppi_env, state.copy(), self.jRNG.new_key(), da)

        a_opt = self.mppi_state[0]
        control = a_opt[0]
        scaled_control = np.multiply(self.norm_param, control)
        # print(sampled_traj[0].shape) [n_samples, n_steps, 5]
        # print(control)
        # print(scaled_control)
        # TODO: check the mppi outputs( its in steerv, accl), convert to vel and steering angle control ig and check mpc node what they do
        
        steerv = scaled_control[0]
        accl = scaled_control[1]
        cmd_steer_angle = self.drive_msg_.drive.steering_angle + steerv*self.DT
        cmd_drive = self.drive_msg_.drive.speed + accl*self.DT

        cmd_steer_angle = np.clip(cmd_steer_angle, -0.4189, 0.4189)
        cmd_drive = np.clip(cmd_drive, 0, 6)

        self.drive_msg_.drive.speed = cmd_drive
        self.drive_msg_.drive.steering_angle = cmd_steer_angle
        # self.drive_msg_.drive.steering_angle_velocity = steerv
        # self.drive_msg_.drive.acceleration = accl

        self.drive_pub_.publish(self.drive_msg_)
        print("drive commands are steer{} and vel{}".format(cmd_steer_angle, cmd_drive))
        print(f"Compute time is {time.time() - start}")

        self.viz_rej_traj(ref_traj)
        self.viz_opt_traj(s_opt)

        # self.viz_sampled_traj(sampled_traj[0])
        self.ref_goal_points_.publish(self.ref_goal_points_data)
        

    #  Visualization MPPI
    def viz_ref_points(self):
        ref_points = MarkerArray()

        for i in range(self.waypoints.shape[0]):
            message = Marker()
            message.header.frame_id="map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.action = Marker.ADD
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.2
            message.scale.y=0.2
            message.scale.z=0.2
            message.color.a=1.0
            message.color.r=1.0
            message.color.b=0.0
            message.color.g=0.0
            message.pose.position.x=float(self.waypoints[i,1])
            message.pose.position.y=float(self.waypoints[i,2])
            message.pose.position.z=0.0
            ref_points.markers.append(message)
        return ref_points
    
    def viz_rej_traj(self, ref_traj):

        traj = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        traj.header.frame_id = 'map'
        traj.color.r = 0.0
        traj.color.g = 0.0
        traj.color.b = 1.0
        traj.color.a = 1.0
        traj.id = 1
        for i in range(ref_traj.shape[1]):
            x, y = ref_traj[i, :2]
            # print(f'Publishing ref traj x={x}, y={y}')
            traj.points.append(Point(x=x, y=y, z=0.0))
        self.ref_trajectory_.publish(traj)

    def viz_opt_traj(self, opt_traj):

        traj = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        traj.header.frame_id = 'map'
        traj.color.r = 1.0
        traj.color.g = 0.0
        traj.color.b = 1.0
        traj.color.a = 1.0
        traj.id = 1
        for i in range(opt_traj.shape[0]):
            x, y = opt_traj[i,:2]
            # print(f'Publishing ref traj x={x}, y={y}')
            traj.points.append(Point(x=float(x), y=float(y), z=0.0))
        self.opt_trajectory_.publish(traj)

    
    def viz_sampled_traj(self, sampled_traj):

        traj = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        traj.header.frame_id = 'map'
        traj.color.r = 1.0
        traj.color.g = 0.5
        traj.color.b = 0.5
        traj.color.a = 0.2
        traj.id = 1
        for i in range(sampled_traj.shape[0]):
            for t in range(sampled_traj.shape[1]):
                x, y = sampled_traj[i,t,:2]
                # print(f'Publishing ref traj x={x}, y={y}')
                traj.points.append(Point(x=float(x), y=float(y), z=0.0))
        self.sampled_trajectory_.publish(traj)


def main(args=None):

    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPPIPlanner()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()