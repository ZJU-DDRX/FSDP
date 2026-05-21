#!/usr/bin/env python3
import time
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from f110_msgs.msg import Wpnt, WpntArray, Obstacle, ObstacleArray, OTWpntArray, OpponentTrajectory, OppWpnt
from dynamic_reconfigure.msg import Config
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray, Float32
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from frenet_converter.frenet_converter import FrenetConverter
from std_msgs.msg import Bool
from copy import deepcopy
from common.differential_flatness import PolynomialPath
from common.qp_fit import QPFit
from common.converter_utils import InitialRefSpline
from geometry_msgs.msg import Point
from mpc.mpc_tracking_controller_ca import MPC_Tracking_Controller, CodeTimer
from mpc.mpc_builder_loader import load_mpc_builder
from tf.transformations import euler_from_quaternion
from scipy.interpolate import CubicSpline
import os
from datetime import datetime
try:
    mpc_builder = load_mpc_builder()
    rospy.loginfo("[SQP_Node] Successfully imported C++ mpc_builder for Spline Fitting.")
except ImportError:
    rospy.logerr("[SQP_Node] Failed to import mpc_builder! Make sure it is compiled.")




class SQPAvoidanceNode:
    """
    This class implements a ROS node that creates a overtaking trajectory around osbtacles and opponents.

    It subscribes to the following topics:
        - `/perception/obstacles`: Subscribes to the obstacle array.
        - `/collision_prediction/obstacles`: Subscribes to the predicted obstacle array (ROCs).
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to the global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.
        - `/global_waypoints_updated`: Subscribes to the updated global waypoints.
        - `/local_waypoints`: Subscribes to the local waypoints.
        - `/opponent_waypoints`: Subscribes to the opponent waypoints.
        - `/ot_section_check`: Subscribes to the overtaking section check.
        - `/dynamic_sqp_tuner_node/parameter_updates`: Subscribes to the dynamic reconfigure updates.

    The node publishes the following topics:
        - `/planner/avoidance/markers_sqp`: Publishes the markers for the avoidance trajectory.
        - `/planner/avoidance/otwpnts`: Publishes the overtaking waypoints.
        - `/planner/avoidance/merger`: Publishes the merger region of the overtaking trajectory.
        - `/planner/pspliner_sqp/latency`: Publishes the latency of the SQP solver. (Only if measure is set to True)
    """

    def __init__(self):
        # Initialize node
        rospy.init_node('sqp_avoidance_node')
        self.rate = rospy.Rate(20)

        # Params
        self.frenet_state = Odometry()
        self.cart_state = Odometry()
        self.local_wpnts = None
        self.lookahead = 15
        self.past_avoidance_d = []
        # Scaled waypoints params
        self.scaled_wpnts = None
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None
        # Updated waypoints params
        self.wpnts_updated = None
        self.max_s_updated = None
        self.max_idx_updated = None
        # Obstalces params
        self.obs = ObstacleArray()
        self.obs_perception = ObstacleArray()
        self.obs_predict = ObstacleArray()
        # Opponent waypoint params
        self.opponent_waypoints = OpponentTrajectory()
        self.max_opp_idx = None
        self.opponent_wpnts_sm = None
        # OT params
        self.last_ot_side = ""
        self.ot_section_check = False
        # Solver params
        self.min_radius = 0.05  # wheelbase / np.tan(max_steering)
        self.max_kappa = 1/self.min_radius
        self.width_car = 0.28
        self.avoidance_resolution = 20
        self.back_to_raceline_before = 5
        self.back_to_raceline_after = 5
        self.obs_traj_tresh = 2
        self.merge_speed_factor = 1.5  # Dynamic merge distance = max(5, v * factor)

        # Dynamic sovler params
        self.down_sampled_delta_s = None
        self.global_traj_kappas = None

        # ROS Parameters
        self.opponent_traj_topic = '/opponent_trajectory'
        self.measure = rospy.get_param("/measure", False)

        # Dynamic reconf params
        self.avoid_static_obs = True

        self.converter = None
        self.global_waypoints = None

        self.initial_ref_spline = None

        self.qp_fit = QPFit()
        self.qp_fit_poly_x = None
        self.qp_fit_poly_y = None
        self.half_width = 0.28 / 2.0

        # Subscribers
        rospy.Subscriber("/perception/obstacles", ObstacleArray, self.obs_perception_cb)
        rospy.Subscriber("/collision_prediction/obstacles", ObstacleArray, self.obs_prediction_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.state_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.scaled_wpnts_cb)
        rospy.Subscriber("/local_waypoints", WpntArray, self.local_wpnts_cb)
        rospy.Subscriber("/dynamic_sqp_tuner_node/parameter_updates", Config, self.dyn_param_cb)
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb)
        rospy.Subscriber("/global_waypoints_updated", WpntArray, self.updated_wpnts_cb)
        rospy.Subscriber(self.opponent_traj_topic, OpponentTrajectory, self.opponent_trajectory_cb)
        rospy.Subscriber("/ot_section_check", Bool, self.ot_sections_check_cb)
        rospy.Subscriber("/car_state/odom", Odometry, self.cart_state_cb)
        # Publishers
        self.mrks_pub = rospy.Publisher("/planner/avoidance/markers_sqp", MarkerArray, queue_size=10)
        self.df_mrks_pub = rospy.Publisher("/planner/avoidance/df_markers", MarkerArray, queue_size=10)
        self.mpc_mrks_pub = rospy.Publisher("/planner/avoidance/mpc_markers", MarkerArray, queue_size=10)
        self.ob_line_pub = rospy.Publisher('/planner/avoidance/ob_line_marker', MarkerArray, queue_size=10)
        self.raw_mrks_pub = rospy.Publisher("/planner/avoidance/raw_traj_markers", MarkerArray, queue_size=10)
        self.evasion_pub = rospy.Publisher("/planner/avoidance/otwpnts", OTWpntArray, queue_size=10)
        self.merger_pub = rospy.Publisher("/planner/avoidance/merger", Float32MultiArray, queue_size=10)
        if self.measure:
            self.measure_pub = rospy.Publisher("/planner/pspliner_sqp/latency", Float32, queue_size=10)

        self.converter = self.initialize_converter()
        self.initial_ref_spline = self.initialize_spline()
        self.mpc_controller = MPC_Tracking_Controller()
        rospy.wait_for_message("/global_waypoints", WpntArray)
        self.mpc_controller.update_vehicle_Lmodel(self.s_ref, self.kapparef)
        rospy.loginfo("[FSDP] initial vehicle linear model!")

    ### Callbacks ###
    def obs_perception_cb(self, data: ObstacleArray):
        self.obs_perception = data
        self.obs_perception.obstacles = [obs for obs in data.obstacles if obs.is_static == True]
        if self.avoid_static_obs == True:
            self.obs.header = data.header
            self.obs.obstacles = self.obs_perception.obstacles + self.obs_predict.obstacles

    def obs_prediction_cb(self, data: ObstacleArray):
        self.obs_predict = data
        self.obs = self.obs_predict
        if self.avoid_static_obs == True:
            self.obs.obstacles = self.obs.obstacles + self.obs_perception.obstacles

    def state_cb(self, data: Odometry):
        self.frenet_state = data

    def cart_state_cb(self, data: Odometry):
        self.cart_state = data
    
    def gb_cb(self, data: WpntArray):
        self.global_waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts]) 
        self.kapparef = np.array([x.kappa_radpm for x in data.wpnts])
        self.s_ref = np.array([x.s_m for x in data.wpnts])
        self.d_left_ref = np.array([x.d_left for x in data.wpnts])
        self.d_right_ref = np.array([x.d_right for x in data.wpnts]) 

    # Everything is refered to the SCALED global waypoints
    def scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])
        self.scaled_wpnts_msg = data
        v_max = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
        if self.scaled_vmax != v_max:
            self.scaled_vmax = v_max
            self.scaled_max_idx = data.wpnts[-1].id
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

    def updated_wpnts_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m
        self.max_idx_updated = self.wpnts_updated[-1].id

    def local_wpnts_cb(self, data: WpntArray):
        self.local_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        self.max_opp_idx = len(data.oppwpnts)-1
        self.opponent_wpnts_sm = np.array([wpnt.s_m for wpnt in data.oppwpnts])

    def ot_sections_check_cb(self, data: Bool):
        self.ot_section_check = data.data

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: Config):
        self.evasion_dist = rospy.get_param("dynamic_sqp_tuner_node/evasion_dist", 0.65)
        self.obs_traj_tresh = rospy.get_param("dynamic_sqp_tuner_node/obs_traj_tresh", 1.5)
        self.spline_bound_mindist = rospy.get_param("dynamic_sqp_tuner_node/spline_bound_mindist", 0.2)
        self.lookahead = rospy.get_param("dynamic_sqp_tuner_node/lookahead_dist", 15)
        self.avoidance_resolution = rospy.get_param("dynamic_sqp_tuner_node/avoidance_resolution", 20)
        self.back_to_raceline_before = rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_before", 5)
        self.back_to_raceline_after = rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_after", 5)
        self.avoid_static_obs = rospy.get_param("dynamic_sqp_tuner_node/avoid_static_obs", True)
        self.merge_speed_factor = rospy.get_param("dynamic_sqp_tuner_node/merge_speed_factor", 1.5)

        print(
            f"[Planner] Dynamic reconf triggered new spline params: \n"
            f" Evasion apex distance: {self.evasion_dist} [m],\n"
            f" Obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" Spline boundary mindist: {self.spline_bound_mindist} [m]\n"
            f" Lookahead distance: {self.lookahead} [m]\n"
            f" Avoid static obstacles: {self.avoid_static_obs}\n"
            f" Avoidance resolution: {self.avoidance_resolution}\n"
            f" Back to raceline before: {self.back_to_raceline_before} [m]\n"
            f" Back to raceline after: {self.back_to_raceline_after} [m]\n"
            f" Merge speed factor: {self.merge_speed_factor}\n"
        )

    def loop(self):
        # Wait for critical Messages and services
        rospy.loginfo("[OBS Spliner] Waiting for messages and services...")
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/dynamic_sqp_tuner_node/parameter_updates", Config)
        rospy.wait_for_message("/local_waypoints", WpntArray)
        rospy.loginfo("[OBS Spliner] Ready!")

        # Frame counter for performance reporting
        frame_count = 0

        while not rospy.is_shutdown():
            # Measure the total loop time
            with CodeTimer("0. Total Loop Time", verbose=False):
                start_time = time.perf_counter()
                obs = deepcopy(self.obs)
                mrks = MarkerArray()
                frenet_state = self.frenet_state
                ct_state = self.cart_state
                self.current_d = frenet_state.pose.pose.position.y
                self.cur_s = frenet_state.pose.pose.position.x
                
                # Obstacle pre-processing
                obs.obstacles = sorted(obs.obstacles, key=lambda obs: obs.s_start)
                considered_obs = []
                for obs in obs.obstacles:               
                    if abs(obs.d_center) < self.obs_traj_tresh and (obs.s_start - self.cur_s) % self.scaled_max_s < self.lookahead:
                        considered_obs.append(obs)
                
                # If there is an obstacle and we are in OT section
                if len(considered_obs) > 0 and self.ot_section_check == True:
                    # print('#################begin overtake###################')
                    
                    # [Timer 1] Measure Fit Curve (Geometric Planning)
                    # This includes GenRawTraj (C++) and QPFit (C++)
                    with CodeTimer("1. Fit Curve (Total)", verbose=False):
                        evasion_x, evasion_y, evasion_s, evasion_d, evasion_v = self.fit_curve(considered_obs, frenet_state.pose.pose.position.x)
                    
                    mpc_x = []
                    mpc_y = []
                    mpc_s = []
                    mpc_d = []
                    mpc_v = []

                    if len(evasion_x) != 0:
                        wheel_base = 0.32
                        dt = 0.05
                        
                        # [Timer 2] Measure Differential Flatness (Analytical Conversion)
                        with CodeTimer("2. Diff Flatness", verbose=False):
                            poly_traj = PolynomialPath(wheel_base, dt, evasion_x, evasion_y, evasion_v, self.max_s_updated, self.converter, self.initial_ref_spline)
                            frenet_states_list, frenet_inputs_list = poly_traj.get_df_frenet(evasion_s, evasion_d, self.qp_fit_poly_x, self.qp_fit_poly_y)
                        
                        ########## MPC Tracking ##########
                        # Note: The timer for MPC Build & Solve is inside mpc_tracking_controller_ca.py
                        x0 = np.array([frenet_states_list[0, 0], frenet_states_list[0, 1], frenet_states_list[0, 3]])
                        mpc_x, mpc_y, mpc_s, mpc_d, mpc_v = self.mpc_tracking(x0, frenet_states_list, frenet_inputs_list)

                    # Publish MPC trajectory
                    mpc_wpnts_msg = OTWpntArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
                    mpc_wpnts = []
                    mpc_wpnts = [Wpnt(id=len(mpc_wpnts), s_m=s, d_m=d, x_m=x, y_m=y, vx_mps=v) for x, y, s, d, v in zip(mpc_x, mpc_y, mpc_s, mpc_d, evasion_v)]
                    mpc_wpnts_msg.wpnts = mpc_wpnts

                    self.evasion_pub.publish(mpc_wpnts_msg)
                    self.visualize_mpc(mpc_s, mpc_d, mpc_x, mpc_y, mpc_v)

                    if len(mpc_s) > 0:
                        self.merger_pub.publish(Float32MultiArray(data=[considered_obs[-1].s_end % self.scaled_max_s, mpc_s[-1] % self.scaled_max_s]))
                    frame_count += 1
                    if frame_count % 30 == 0:
                        CodeTimer.print_summary(clear_history=True)
                # IF there is no point in overtaking anymore delete all markers
                else:
                    mrks = MarkerArray()
                    del_mrk = Marker(header=rospy.Header(stamp=rospy.Time.now()))
                    del_mrk.action = Marker.DELETEALL
                    mrks.markers = []
                    mrks.markers.append(del_mrk)
                    self.mrks_pub.publish(mrks)
                    self.raw_mrks_pub.publish(mrks)
            
                # Publish latency
                if self.measure:
                    self.measure_pub.publish(Float32(data=time.perf_counter() - start_time))
            
            # [Report] Print performance summary every 100 frames
            

            self.rate.sleep()

    def mpc_tracking(self, x0, frenet_states_list, frenet_inputs_list):
        """ mpc tracking controller """
        # [s, d, delta_phi]
        raw_s = frenet_states_list[:, 0]
        raw_d = frenet_states_list[:, 1]
        for i in range(len(raw_s) - 1):
            if raw_s[i] > raw_s[i + 1]:
                raw_s[i + 1:] = [s + self.scaled_max_s for s in raw_s[i + 1:]]
                break
        
        # Detect merge-back region (where trajectory returns to raceline)
        # Criterion: |d| is decreasing towards 0 in the second half of trajectory
        is_merge_region = np.zeros(len(raw_d), dtype=bool)
        if len(raw_d) > 10:
            mid_point = len(raw_d) // 2
            d_gradient = np.gradient(np.abs(raw_d[mid_point:]))
            # Mark points where |d| is decreasing (negative gradient) and |d| < threshold
            merge_threshold = 0.3  # Consider merge when |d| < 0.3m
            for i in range(mid_point, len(raw_d)):
                if np.abs(raw_d[i]) < merge_threshold and (i > mid_point and np.abs(raw_d[i]) < np.abs(raw_d[i-1])):
                    is_merge_region[i] = True
        
        x_ref = np.column_stack((
            raw_s, 
            frenet_states_list[:, 1], 
            frenet_states_list[:, 3]
        ))

        # [v, delta]
        u_ref = np.column_stack((
            frenet_states_list[:, 2], 
            frenet_inputs_list[:, 1]
        ))

        #print bounds time
        start_time = time.time()
        # cal left boundary and right boundary for raw_s (vectorized nearest-index lookup)
        s_mod = np.asarray(raw_s, dtype=float) % self.scaled_max_s
        idx_r = np.searchsorted(self.s_ref, s_mod)
        idx_r = np.clip(idx_r, 1, len(self.s_ref) - 1)
        idx_l = idx_r - 1
        bound_raw_s_idx = np.where(
            np.abs(self.s_ref[idx_l] - s_mod) <= np.abs(self.s_ref[idx_r] - s_mod),
            idx_l, idx_r
        )
        left_bound = self.d_left_ref[bound_raw_s_idx] - 0.5 * self.width_car
        right_bound = np.minimum(-self.d_right_ref[bound_raw_s_idx] + 0.5 * self.width_car, left_bound - 2*(self.half_width+0.1)) # avoid right_bound > left_bound
        s_bound = raw_s % self.scaled_max_s
        self.get_obs_pre_resp()

        # Extract s values from observed responses (N observations)
        obs_s = self.obs_pre_resp[:, 0]  # Observed s values (first column)
        obs_left = self.obs_pre_resp[:, 1]  # Left boundary values from observations
        obs_right = self.obs_pre_resp[:, 2]  # Right boundary values from observations
        for i in range(len(obs_s) - 1):
            if obs_s[i] > obs_s[i + 1]:
                obs_s[i + 1:] = [s + self.scaled_max_s for s in obs_s[i + 1:]]
                break

        # Find the range of obs_s (ensure ref_s is within this range)
        min_s = np.min(obs_s)
        max_s = np.max(obs_s)

        # Create a mask to only interpolate ref_s values within the range of obs_s
        valid_mask = (raw_s >= min_s) & (raw_s <= max_s)

        left_bound[valid_mask] = np.interp(raw_s[valid_mask], obs_s, obs_left)  # Interpolate left bound
        right_bound[valid_mask] = np.interp(raw_s[valid_mask], obs_s, obs_right)  # Interpolate right bound

        overtake_left_resp = self.converter.get_cartesian(s_bound, left_bound)
        overtake_right_resp = self.converter.get_cartesian(s_bound, right_bound)
        start_point = overtake_left_resp.T
        end_point = overtake_right_resp.T

        self.visualize_ob_lines(start_point, end_point)
        end_time = time.time()-start_time
        #
        # print(f"bounds time: {end_time*1000:.3f}ms")

        if len(x_ref) >= self.mpc_controller.N:
            # Relax margin in merge region for smoother transition
            if np.any(is_merge_region[:self.mpc_controller.N]):
                original_margin = self.mpc_controller.margin
                self.mpc_controller.margin = 0.02  # Reduce margin in merge region
                res_u, res_x = self.mpc_controller.solve_qp(x0, x_ref, u_ref, left_bound, right_bound)
                self.mpc_controller.margin = original_margin  # Restore original margin
            else:
                res_u, res_x = self.mpc_controller.solve_qp(x0, x_ref, u_ref, left_bound, right_bound)
        else:
            res_u = None
            res_x = None

        mpc_x = []
        mpc_y = []
        mpc_s = []
        mpc_d = []
        mpc_v = []
        
        if res_u is not None:
            res_x = np.array(res_x).reshape(-1, 3)
            res_u = np.array(res_u).reshape(-1, 2)

            # output of mpc
            origin_s = res_x[:, 0]
            origin_d = res_x[:, 1]
            origin_v = res_u[:, 0]

            # interpolate the output of mpc
            ds = 0.1
            mpc_s = np.arange(origin_s[0], origin_s[-1], ds) 
            mpc_d = np.interp(mpc_s, origin_s, origin_d)
            mpc_v = np.interp(mpc_s, origin_s, origin_v)

            # spline_d = CubicSpline(origin_s, origin_d)
            # spline_v = CubicSpline(origin_s, origin_v)
            # mpc_d = spline_d(mpc_s)
            # mpc_v = spline_v(mpc_s)

            mpc_s = mpc_s % self.scaled_max_s
            resp = self.converter.get_cartesian(mpc_s, mpc_d)
            mpc_x = resp[0, :]
            mpc_y = resp[1, :]

        return mpc_x, mpc_y, mpc_s, mpc_d, mpc_v

   
    
    def fit_curve(self, considered_obs: list, cur_s: float):
        danger_flag = False
        
        # --------------------------------------------------------
        # 1. Decision Making & ROI Selection
        # --------------------------------------------------------
        
        # Get the initial guess of the overtaking side (see spliner)
        initial_guess_object = self.group_objects(considered_obs)
        
        # Get the total number of waypoints for correct wrapping
        num_points = len(self.scaled_wpnts)

        # Find start and end indices of the obstacle in the global path
        # Note: Using [:, 0] to ensure we search only in the 's' column
        initial_guess_object_start_idx = np.abs(self.scaled_wpnts[:, 0] - initial_guess_object.s_start).argmin()
        initial_guess_object_end_idx = np.abs(self.scaled_wpnts[:, 0] - initial_guess_object.s_end).argmin()
        
        # Calculate the number of points covering the Region of Collision (ROC)
        # Handle wrap-around case (e.g., obstacle crosses the finish line)
        if initial_guess_object_end_idx < initial_guess_object_start_idx:
            n_points_roc = (initial_guess_object_end_idx + num_points) - initial_guess_object_start_idx
        else:
            n_points_roc = initial_guess_object_end_idx - initial_guess_object_start_idx

        # Get array of indexes of the global waypoints overlapping with the ROC
        # Using modulo operator to handle circular track indices safely
        gb_idxs = np.array([(initial_guess_object_start_idx + i) % num_points for i in range(n_points_roc)])
        
        # If the ROC is too short, we take the next 20 waypoints to ensure stability
        if len(gb_idxs) < 20:
            # Re-calculate start index approximation to ensure integer type
            start_idx_approx = int(initial_guess_object.s_center / self.scaled_delta_s)
            gb_idxs = np.array([(start_idx_approx + i) % num_points for i in range(20)])

        # Determine overtaking side (left/right) based on available space
        side, initial_apex = self._more_space(initial_guess_object, self.scaled_wpnts_msg.wpnts, gb_idxs)
        
        # Analyze curvature to adjust strategy
        kappas = np.array([self.scaled_wpnts_msg.wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs])
        max_kappa = np.max(np.abs(kappas))
        outside = "left" if np.sum(kappas) < 0 else "right"

        # Enlongate the ROC if our initial guess suggests that we are overtaking on the outside
        # This provides more margin for the maneuver on the longer path
        if side == outside:
            for i in range(len(considered_obs)):
                considered_obs[i].s_end = considered_obs[i].s_end + (considered_obs[i].s_end - considered_obs[i].s_start)%self.max_s_updated * max_kappa * (self.width_car + self.evasion_dist)

        # --------------------------------------------------------
        # 2. Discretization & Sampling
        # --------------------------------------------------------

        min_s_obs_start = self.scaled_max_s
        max_s_obs_end = 0
        for obs in considered_obs:
            if obs.s_start < min_s_obs_start:
                min_s_obs_start = obs.s_start
            if obs.s_end > max_s_obs_end:
                max_s_obs_end = obs.s_end
            # Check if it is a really wide obstacle (Safety Flag)
            if obs.d_left > 3 or obs.d_right < -3:
                danger_flag = True

        # Define the longitudinal range for the avoidance maneuver
        # Dynamic merge distance based on current velocity
        current_velocity = max(self.frenet_state.twist.twist.linear.x, 1.0)  # Avoid zero velocity
        dynamic_merge_dist = max(self.back_to_raceline_after, current_velocity * self.merge_speed_factor)
        
        start_avoidance = max((min_s_obs_start - self.back_to_raceline_before), cur_s)
        end_avoidance = max_s_obs_end + dynamic_merge_dist

        # Generate downsampled s-coordinates for the maneuver
        s_avoidance = np.linspace(start_avoidance, end_avoidance, self.avoidance_resolution)
        self.down_sampled_delta_s = s_avoidance[1] - s_avoidance[0]
        
        # Find corresponding global waypoints for boundary extraction (vectorized)
        s_mod = s_avoidance % self.scaled_max_s
        scaled_wpnts_s = self.scaled_wpnts[:, 0]
        idx_r = np.searchsorted(scaled_wpnts_s, s_mod)
        idx_r = np.clip(idx_r, 1, len(scaled_wpnts_s) - 1)
        idx_l = idx_r - 1
        scaled_wpnts_indices = np.where(
            np.abs(scaled_wpnts_s[idx_l] - s_mod) <= np.abs(scaled_wpnts_s[idx_r] - s_mod),
            idx_l, idx_r
        )
        corresponding_scaled_wpnts = [self.scaled_wpnts_msg.wpnts[i] for i in scaled_wpnts_indices]
        
        # Pre-extract wpnt arrays for faster access
        wpnts_d_left = np.array([w.d_left for w in corresponding_scaled_wpnts])
        wpnts_d_right = np.array([w.d_right for w in corresponding_scaled_wpnts])
        wpnts_x = np.array([w.x_m for w in corresponding_scaled_wpnts])
        wpnts_y = np.array([w.y_m for w in corresponding_scaled_wpnts])
        wpnts_v = np.array([w.vx_mps for w in corresponding_scaled_wpnts])
        
        # Calculate Global Trajectory Curvature (vectorized)
        x_prime = np.diff(wpnts_x)
        x_prime = np.where(x_prime == 0, 1e-6, x_prime)
        y_prime = np.diff(wpnts_y)
        y_prime = np.where(y_prime == 0, 1e-6, y_prime)
        x_prime_prime = np.diff(x_prime)
        y_prime_prime = np.diff(y_prime)
        x_prime = x_prime[:-1]
        y_prime = y_prime[:-1]
        self.global_traj_kappas = (x_prime*y_prime_prime - y_prime*x_prime_prime) / ((x_prime**2 + y_prime**2)**(3/2))
       
        # --------------------------------------------------------
        # 3. Obstacle Projection
        # --------------------------------------------------------
        
        # Map obstacles onto the sampled s-avoidance grid (optimized with list pre-allocation)
        obs_indices_list = []
        obs_center_list = []
        obs_dist_list = []
        inflation_idx = 2
        n_avoidance = len(s_avoidance)

        for obs in considered_obs:
            obs_idx_start = np.searchsorted(s_avoidance, obs.s_start)
            obs_idx_start = min(obs_idx_start, n_avoidance - 1)
            obs_idx_end = np.searchsorted(s_avoidance, obs.s_end)
            obs_idx_end = min(obs_idx_end, n_avoidance - 1)
            obs_idx_start = max(0, obs_idx_start - inflation_idx)
            obs_idx_end = min(n_avoidance - 1, obs_idx_end + inflation_idx)
            
            if obs_idx_start < n_avoidance - 2:
                if obs.is_static or obs_idx_end == obs_idx_start:
                    if obs_idx_end == obs_idx_start:
                        obs_idx_end = obs_idx_start + 1
                    n_pts = obs_idx_end - obs_idx_start + 1
                    obs_indices_list.append(np.arange(obs_idx_start, obs_idx_end + 1))
                    obs_center_list.append(np.full(n_pts, (obs.d_left + obs.d_right) * 0.5))
                    obs_dist_list.append(np.full(n_pts, (obs.d_left - obs.d_right) * 0.5 + self.width_car + self.evasion_dist))
                else:
                    indices = np.arange(obs_idx_start, obs_idx_end + 1)
                    obs_indices_list.append(indices)
                    # Vectorized opponent waypoint lookup
                    s_query = s_avoidance[indices] % self.max_opp_idx
                    opp_idx_r = np.searchsorted(self.opponent_wpnts_sm, s_query)
                    opp_idx_r = np.clip(opp_idx_r, 1, len(self.opponent_wpnts_sm) - 1)
                    opp_idx_l = opp_idx_r - 1
                    opp_wpnts_idx = np.where(
                        np.abs(self.opponent_wpnts_sm[opp_idx_l] - s_query) <= np.abs(self.opponent_wpnts_sm[opp_idx_r] - s_query),
                        opp_idx_l, opp_idx_r
                    )
                    d_opp = np.array([self.opponent_waypoints[i].d_m for i in opp_wpnts_idx])
                    obs_center_list.append(d_opp)
                    obs_dist_list.append(np.full(len(indices), self.width_car + self.evasion_dist))

        self.obs_downsampled_indices = np.concatenate(obs_indices_list).astype(int) if obs_indices_list else np.array([], dtype=int)
        self.obs_downsampled_center_d = np.concatenate(obs_center_list) if obs_center_list else np.array([])
        self.obs_downsampled_min_dist = np.concatenate(obs_dist_list) if obs_dist_list else np.array([])
        self.s_avoidance = s_avoidance
        self.side = side

        # --------------------------------------------------------
        # 4. Raw Trajectory Generation (C++ Optimized)
        # --------------------------------------------------------

        raw_evasion_v = wpnts_v
        ref_d_left = wpnts_d_left
        ref_d_right = wpnts_d_right
        raw_evasion_s = s_avoidance[:]
        
        # Generate the initial piecewise linear guess
        # Note: Optimized via C++ extension (mpc_builder) if integrated
        raw_evasion_d, poly_x, poly_y, t_data, is_traj_valid = self.gen_raw_traj(
            raw_evasion_s, raw_evasion_v, ref_d_left, ref_d_right, side
        )

        # --------------------------------------------------------
        # 5. Trajectory Fitting (C++ / QP Optimized)
        # --------------------------------------------------------

        # Update minimum radius constraints based on speed
        clipped_speed = np.clip(self.frenet_state.twist.twist.linear.x, 1, 6.5)
        radius_speed = min([clipped_speed, self.wpnts_updated[(scaled_wpnts_indices[0])%self.max_idx_updated].vx_mps])
        self.min_radius = np.interp(radius_speed, [1, 6, 7], [0.2, 2, 4])
        self.max_kappa = 1/self.min_radius

        if len(self.past_avoidance_d) == 0:
            initial_guess = np.full(len(s_avoidance), initial_apex)
        elif len(self.past_avoidance_d) > 0:
            initial_guess = self.past_avoidance_d
        else:
            if self.last_ot_side == "left":
                initial_guess = np.full(len(s_avoidance), 2)
            else:
                initial_guess = np.full(len(s_avoidance), -2)
            
        # Note: The original Python SQP solver call is commented out in favor of the new pipeline
        # result = self.solve_sqp(initial_guess, bounds)

        if is_traj_valid == True:
            # Prepare data for fitting
            x_data, y_data, t_data = self.get_fit_variable(raw_evasion_s, raw_evasion_d, raw_evasion_v)
            
            # Solve QP for Quintic Spline Coefficients
            # (This step is now accelerated via C++ Closed-form solution)
            poly_x, poly_y = self.get_qp_fit_coeffs(x_data, y_data, t_data, raw_evasion_s, raw_evasion_v)
            
            # Validate the fitted trajectory against boundaries
            closedform_evasion_valid = self.check_fit_traj_valid(poly_x, poly_y, t_data, ref_d_left if side == 'left' else ref_d_right)
            
            if closedform_evasion_valid == False:
                # Return empty if validation fails
                return [], [], [], [], []
                
            # --------------------------------------------------------
            # 6. Interpolation & Output Generation
            # --------------------------------------------------------

            # Resample to global resolution
            n_global_avoidance_points = int((end_avoidance - start_avoidance) / self.scaled_delta_s)
            s_array = np.linspace(start_avoidance, end_avoidance, n_global_avoidance_points)
            
            # Interpolate lateral deviation d
            evasion_d = np.interp(s_array, s_avoidance, raw_evasion_d)
            
            # Handle wrapping for s
            evasion_s = np.mod(s_array, self.scaled_max_s)
            
            # Convert to Cartesian coordinates
            resp = self.converter.get_cartesian(evasion_s, evasion_d)
            evasion_x = resp[0, :]
            evasion_y = resp[1, :]
            
            # Interpolate velocity (use pre-extracted wpnts_v)
            evasion_v = np.interp(s_array, s_avoidance, wpnts_v)
            
            # Post-overtake speed reduction: detect upcoming corners in merge region
            merge_start_idx = int(len(evasion_s) * 0.7)  # Last 30% is merge region
            if merge_start_idx < len(evasion_s):
                # Get upcoming curvature after merge
                s_merge = evasion_s[merge_start_idx:]
                idx_merge = np.searchsorted(self.s_ref, s_merge)
                idx_merge = np.clip(idx_merge, 0, len(self.kapparef) - 1)
                kappas_ahead = self.kapparef[idx_merge]
                max_kappa_ahead = np.max(np.abs(kappas_ahead)) if len(kappas_ahead) > 0 else 0
                
                # If high curvature ahead (sharp corner), reduce speed in merge region
                if max_kappa_ahead > 0.15:  # Threshold for "sharp corner"
                    speed_reduction_factor = np.clip(1.0 - (max_kappa_ahead - 0.15) * 2.0, 0.6, 1.0)
                    # Apply smooth speed reduction in merge region
                    blend = np.linspace(1.0, speed_reduction_factor, len(evasion_s) - merge_start_idx)
                    evasion_v[merge_start_idx:] *= blend
            
            # Update internal state
            self.past_avoidance_d = raw_evasion_d[:]
            self.qp_fit_poly_x = poly_x[:]
            self.qp_fit_poly_y = poly_y[:]
            
            if np.mean(evasion_d) > 0:
                self.last_ot_side = "left"
            else:
                self.last_ot_side = "right"

        else:
            # Return empty lists if trajectory generation failed
            evasion_x = []
            evasion_y = []
            evasion_s = []
            evasion_d = []
            evasion_v = []
            self.past_avoidance_d = []
            self.qp_fit_poly_x = np.array([])
            self.qp_fit_poly_y = np.array([])
        
        # Visualize the result in Rviz
        self.visualize_sqp(evasion_s, evasion_d, evasion_x, evasion_y, evasion_v) 

        return evasion_x, evasion_y, evasion_s, evasion_d, evasion_v
    
    def get_obs_pre_resp(self):
        # Vectorized nearest s_ref index for obstacle downsampled points
        s_obs = self.s_avoidance[self.obs_downsampled_indices].astype(float) % self.scaled_max_s
        idx_r = np.searchsorted(self.s_ref, s_obs)
        idx_r = np.clip(idx_r, 1, len(self.s_ref) - 1)
        idx_l = idx_r - 1
        opp_wpnts_glbidx = np.where(
            np.abs(self.s_ref[idx_l] - s_obs) <= np.abs(self.s_ref[idx_r] - s_obs),
            idx_l, idx_r
        )

        track_left_bound = self.d_left_ref[opp_wpnts_glbidx]
        track_right_bound = -self.d_right_ref[opp_wpnts_glbidx]
        obs_center = self.obs_downsampled_center_d
        safe_dist = self.obs_downsampled_min_dist

        if self.side == 'left':
            ub = track_left_bound
            lb = np.where(obs_center + safe_dist > ub, ub - 0.05, obs_center + safe_dist)
        else:
            lb = track_right_bound
            ub = np.where(obs_center - safe_dist < lb, lb + 0.05, obs_center - safe_dist)

        first_sequence = ub.tolist()
        second_sequence = lb.tolist()
        s_sequence = self.s_avoidance[self.obs_downsampled_indices].copy()

        # Points ahead of ROC for safer overtaking (prepend without insert(0))
        ahead_num = 5
        safe_point_start_idx = max(self.obs_downsampled_indices[0] - ahead_num, 0)
        if safe_point_start_idx > 0:
            start_idx = self.obs_downsampled_indices[0] - ahead_num
            end_idx = self.obs_downsampled_indices[0] - 1
            index_range = np.arange(start_idx, end_idx + 1)
            s_ahead = self.s_avoidance[index_range].astype(float) % self.scaled_max_s
            idx_r = np.searchsorted(self.s_ref, s_ahead)
            idx_r = np.clip(idx_r, 1, len(self.s_ref) - 1)
            idx_l = idx_r - 1
            larger_wpnts_glbidx = np.where(
                np.abs(self.s_ref[idx_l] - s_ahead) <= np.abs(self.s_ref[idx_r] - s_ahead),
                idx_l, idx_r
            )
            # Order for prepend: obs_start-1, obs_start-2, ..., obs_start-5 -> use larger_wpnts_glbidx[::-1]
            rev_idx = larger_wpnts_glbidx[::-1]
            new_s = np.array([self.s_avoidance[self.obs_downsampled_indices[0] - (i + 1)] for i in range(ahead_num)], dtype=float)
            if self.side == 'left':
                new_first = self.d_left_ref[rev_idx].tolist()
                limit0 = second_sequence[0]
                new_second = np.maximum.accumulate(
                    np.concatenate([[limit0], -self.d_right_ref[rev_idx]])
                )[1:].tolist()
            else:
                new_second = (-self.d_right_ref[rev_idx]).tolist()
                limit0 = first_sequence[0]
                new_first = np.minimum.accumulate(
                    np.concatenate([[limit0], self.d_left_ref[rev_idx]])
                )[1:].tolist()
            first_sequence = new_first + first_sequence
            second_sequence = new_second + second_sequence
            s_sequence = np.concatenate([new_s, s_sequence])

        self.obs_pre_resp = np.column_stack((s_sequence, first_sequence, second_sequence))

    
    def plot_sample(self, corresponding_scaled_wpnts, poly_x, poly_y, t_data, raw_evasion_s, raw_evasion_d, ref_d_left, ref_d_right, side, is_traj_valid):
        x_global_points = np.array([wpnt.x_m for wpnt in corresponding_scaled_wpnts])
        y_global_points = np.array([wpnt.y_m for wpnt in corresponding_scaled_wpnts])

        raw_resp = self.converter.get_cartesian(raw_evasion_s, raw_evasion_d)
        raw_evasion_x = raw_resp[0, :]
        raw_evasion_y = raw_resp[1, :]

        left_resp = self.converter.get_cartesian(raw_evasion_s, ref_d_left)
        left_x = left_resp[0, :]
        left_y = left_resp[1, :]

        right_resp = self.converter.get_cartesian(raw_evasion_s, -ref_d_right)
        right_x = right_resp[0, :]
        right_y = right_resp[1, :]


        obs_s = np.array(raw_evasion_s)[self.obs_downsampled_indices]
        obs_resp = self.converter.get_cartesian(obs_s, self.obs_downsampled_center_d)
        obs_x = obs_resp[0, :]
        obs_y = obs_resp[1, :]

        
        if side == 'left':
            safe_d = self.obs_downsampled_min_dist
        else:
            safe_d = -self.obs_downsampled_min_dist
        safe_resp = self.converter.get_cartesian(obs_s, safe_d)
        safe_x = safe_resp[0, :]
        safe_y = safe_resp[1, :]

        save_dir = '~/eth_race/sample'
        save_dir = os.path.expanduser(save_dir)
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(save_dir, f"sample_{side}_{timestamp}_valid_{is_traj_valid}.png")

        # Initialize the plot
        plt.figure()
        plt.title("Sample Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        current_pos = self.converter.get_cartesian(self.frenet_state.pose.pose.position.x, self.frenet_state.pose.pose.position.y)
        print(f'current_pos size:{current_pos.shape}')
        current_x = current_pos[0]
        current_y = current_pos[1]
        plt.scatter(current_x, current_y, color='r', s=200, marker='*')
        plt.plot(x_global_points, y_global_points, color='k', linestyle='--')
        plt.plot(raw_evasion_x, raw_evasion_y, color="r", linestyle="-")
        plt.plot(left_x, left_y, color="k", linestyle="-")
        plt.plot(right_x, right_y, color="k", linestyle="-")

        plt.plot(safe_x, safe_y, color="g", linestyle="--")
        plt.scatter(obs_x, obs_y, color='b', s=100, marker='o')

        x_fit = self.qp_fit.poly(t_data, poly_x)  # x(t)
        y_fit = self.qp_fit.poly(t_data, poly_y)  # y(t)
        plt.plot(x_fit, y_fit, 'purple')
        # Save the plot
        plt.savefig(file_path)
        plt.close()  # Close the figure to free memory

    # def gen_raw_traj(self, raw_evasion_s, raw_evasion_v, ref_d_left, ref_d_right, side):
    #     # Convert inputs to C++ compatible format
    #     s_vec = raw_evasion_s.astype(np.float64)
    #     ref_l = ref_d_left.astype(np.float64)
    #     ref_r = ref_d_right.astype(np.float64)
    #     obs_idx = self.obs_downsampled_indices.astype(np.int32)
    #     obs_center = self.obs_downsampled_center_d.astype(np.float64)
    #     obs_min = self.obs_downsampled_min_dist.astype(np.float64)

    #     current_d = self.frenet_state.pose.pose.position.y
    #     if self.cur_s % self.scaled_max_s < raw_evasion_s[0] % self.scaled_max_s:
    #          current_d = 0.0

    #     # Call C++
    #     raw_d, is_valid = mpc_builder.gen_raw_traj_cpp(
    #         s_vec, current_d, ref_l, ref_r, 
    #         obs_idx, obs_center, obs_min, side
    #     )

    #     if not is_valid:
    #         return np.array([]), np.array([]), np.array([]), np.array([]), False
        
    #     return raw_d, np.array([]), np.array([]), np.array([]), True   
    def gen_raw_traj(self, raw_evasion_s, raw_evasion_v, ref_d_left, ref_d_right, side):
        if self.obs_downsampled_indices.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), False
        
        # Initialize raw_evasion_d as a zero list with the same length as raw_evasion_s
        raw_evasion_d = [0] * len(raw_evasion_s)
        # Get the indices of the obstacles
        idx_start_in_es = self.obs_downsampled_indices[0]  # The first obstacle point
        idx_start_in_es = max(idx_start_in_es - 2, 0)
        idx_mid_in_es = self.obs_downsampled_indices[len(self.obs_downsampled_indices) // 2]  # The middle obstacle point
        idx_end_in_es = self.obs_downsampled_indices[-1]  # The last obstacle point
        idx_end_in_es = min(idx_end_in_es + 6, len(raw_evasion_s) - 1)
        idx_start_in_obs = 0
        idx_mid_in_obs = len(self.obs_downsampled_indices) // 2
        idx_end_in_obs = -1
        safe_dist = 0.2
        lon_dist = 1.5

        raw_evasion_valid = False
        closedform_evasion_valid = False
        frenet_state = self.frenet_state

        current_s = frenet_state.pose.pose.position.x
        current_d = frenet_state.pose.pose.position.y
        if current_s % self.scaled_max_s < raw_evasion_s[0] % self.scaled_max_s:
            # the current pos of ego car is not the planning start
            current_d = 0
        # the ego car and the planning start are not on the same side
        # and the longitudinal distance between the ego car and the planning start is too small. 
        if current_d < 0:
            if side == "left":
                if np.fabs(raw_evasion_s[idx_start_in_es] - raw_evasion_s[0]) < lon_dist:
                    # not suitable for overtaking
                    return np.array([]), np.array([]), np.array([]), np.array([]), False
        if current_d > 0:
            if side == "right":
                if np.fabs(raw_evasion_s[idx_start_in_es] - raw_evasion_s[0]) < lon_dist:
                    # not suitable for overtaking
                    return np.array([]), np.array([]), np.array([]), np.array([]), False

        # Calculate the lateral offset absolute value for each of the points
        # Curvature compensation for outside overtaking to avoid cutting corners
        kappas_start_to_mid = self.global_traj_kappas[0:max(idx_mid_in_es - 2, 1)] if len(self.global_traj_kappas) > 0 else np.array([0])
        kappas_merge = self.global_traj_kappas[max(idx_end_in_es - 2, 0):] if len(self.global_traj_kappas) > idx_end_in_es else np.array([0])
        avg_kappa = np.mean(kappas_start_to_mid)  # Keep sign for direction
        avg_kappa_merge = np.mean(kappas_merge) if len(kappas_merge) > 0 else 0
        outside = "left" if avg_kappa < 0 else "right"
        is_outside_overtake = (side == outside)
        curvature_gain =2.5#tuning parameter for gain for extreme curves
        
        if side == "left":
            mid_d = max(min(ref_d_left[idx_mid_in_es] - safe_dist, self.obs_downsampled_min_dist[idx_mid_in_obs] + safe_dist), self.obs_downsampled_min_dist[idx_mid_in_obs])
            start_d = max(min(ref_d_left[idx_start_in_es] - safe_dist, self.obs_downsampled_center_d[idx_start_in_obs] + safe_dist), self.obs_downsampled_min_dist[idx_start_in_obs])
            end_d = max(min(ref_d_left[idx_end_in_es] - safe_dist, self.obs_downsampled_center_d[idx_end_in_obs] + safe_dist), self.obs_downsampled_min_dist[idx_end_in_obs])
            
            # Outside overtake curvature compensation: increase start_d to avoid cutting
            if is_outside_overtake and idx_start_in_es > 0:
                lon_dist_to_mid = (idx_mid_in_es - idx_start_in_es) * self.down_sampled_delta_s
                curvature_comp = curvature_gain * abs(avg_kappa) * lon_dist_to_mid * mid_d
                start_d = min(start_d + curvature_comp, ref_d_left[idx_start_in_es] - safe_dist)
            
            # Ensure that the lateral offsets at idx_start and idx_end are less than mid_d
            if idx_start_in_es == 0:
                start_d = current_d
            elif 0 < idx_start_in_es < 4:
                start_d = current_d + (idx_start_in_es / idx_mid_in_es) * (mid_d - current_d)
            else:
                start_d = min(start_d, mid_d)
            end_d = min(end_d, mid_d)
            print(f'[Left Overtake] start_d:{start_d:.3f}, mid_d:{mid_d:.3f}, end_d:{end_d:.3f}, outside:{is_outside_overtake}')
            print(f'[Merge Info] idx_end:{idx_end_in_es}, merge_points:{len(raw_evasion_s) - idx_end_in_es}, merge_dist:{(len(raw_evasion_s) - idx_end_in_es)*self.scaled_delta_s:.2f}m')

            # For the segments: 0 to idx_start, idx_start to idx_mid, idx_mid to idx_end, idx_end to len
            if idx_start_in_es == 0:
                raw_evasion_d[0] = current_d
                start_d = current_d
            else:
                raw_evasion_d[0:idx_start_in_es] = np.linspace(current_d, start_d, num=idx_start_in_es)
            raw_evasion_d[idx_start_in_es:idx_mid_in_es] = np.linspace(start_d, mid_d, num=idx_mid_in_es - idx_start_in_es)
            raw_evasion_d[idx_mid_in_es:idx_end_in_es] = np.linspace(mid_d, end_d, num=idx_end_in_es - idx_mid_in_es)
            
            # Curvature-aware merge-back: adjust based on track curvature direction
            n_merge = len(raw_evasion_s) - idx_end_in_es
            if n_merge > 0:
                # Check if merge direction conflicts with track curvature (convex/concave mismatch)
                merge_curvature_conflict = (end_d > 0 and avg_kappa_merge > 0) or (end_d < 0 and avg_kappa_merge < 0)
                if merge_curvature_conflict:
                    # Use slower S-curve for smoother transition when curvature conflicts
                    t = np.linspace(0, 1, n_merge)
                    s_curve = 3 * t**2 - 2 * t**3  # Smooth S-curve: 0->1
                    raw_evasion_d[idx_end_in_es:] = end_d * (1 - s_curve)
                else:
                    t = np.linspace(0, np.pi/2, n_merge)
                    raw_evasion_d[idx_end_in_es:] = end_d * np.cos(t)

            # check the raw d is not out of boundary
            raw_evasion_valid = self.check_interp_traj_valid(raw_evasion_d, ref_d_left)
            if raw_evasion_valid == False:
                return raw_evasion_d, np.array([]), np.array([]), np.array([]), False
            else:
                x_data, y_data, t_data = self.get_fit_variable(raw_evasion_s, raw_evasion_d, raw_evasion_v)
                start_qp_fit_time = time.time()
                poly_x, poly_y = self.get_qp_fit_coeffs(x_data, y_data, t_data, raw_evasion_s, raw_evasion_v)
                end_qp_fit_time = time.time()
                qp_fit_time = end_qp_fit_time - start_qp_fit_time
                print(f"qp_fit_time: {qp_fit_time*1000:.3f}ms")
                if len(poly_x) == 0 or len(poly_y) == 0:
                    return raw_evasion_d, np.array([]), np.array([]), np.array([]), False
                start_check_closed_time = time.time()
                closedform_evasion_valid = self.check_fit_traj_valid(poly_x, poly_y, t_data, ref_d_left)
                end_check_closed_time = time.time()
                check_closed_time = end_check_closed_time - start_check_closed_time
                print(f"check fit traj time: {check_closed_time*1000:.3f}ms")
                if closedform_evasion_valid == False:
                    # return np.array([]), np.array([]), np.array([]), False
                    return raw_evasion_d, poly_x, poly_y, t_data, False
        elif side == "right":
            mid_d = -max(min(ref_d_right[idx_mid_in_es] - safe_dist, self.obs_downsampled_min_dist[idx_mid_in_obs] + safe_dist), self.obs_downsampled_min_dist[idx_mid_in_obs])
            start_d = -max(min(ref_d_right[idx_start_in_es] - safe_dist, self.obs_downsampled_center_d[idx_start_in_obs] + safe_dist), self.obs_downsampled_min_dist[idx_start_in_obs])
            end_d = -max(min(ref_d_right[idx_end_in_es] - safe_dist, self.obs_downsampled_center_d[idx_end_in_obs] + safe_dist), self.obs_downsampled_min_dist[idx_end_in_obs])

            # Outside overtake curvature compensation: decrease start_d (more negative) to avoid cutting
            if is_outside_overtake and idx_start_in_es > 0:
                lon_dist_to_mid = (idx_mid_in_es - idx_start_in_es) * self.down_sampled_delta_s
                curvature_comp = curvature_gain * abs(avg_kappa) * lon_dist_to_mid * abs(mid_d)
                start_d = max(start_d - curvature_comp, -(ref_d_right[idx_start_in_es] - safe_dist))

            # Ensure that the lateral offsets at idx_start and idx_end are less than mid_d
            if idx_start_in_es == 0:
                start_d = current_d
            elif 0 < idx_start_in_es < 4:
                start_d = current_d + (idx_start_in_es / idx_mid_in_es) * (mid_d - current_d)
            else:
                start_d = max(start_d, mid_d)
            end_d = max(end_d, mid_d)
            print(f'[Right Overtake] start_d:{start_d:.3f}, mid_d:{mid_d:.3f}, end_d:{end_d:.3f}, outside:{is_outside_overtake}')
            print(f'[Merge Info] idx_end:{idx_end_in_es}, merge_points:{len(raw_evasion_s) - idx_end_in_es}, merge_dist:{(len(raw_evasion_s) - idx_end_in_es)*self.scaled_delta_s:.2f}m')

            # For the segments: 0 to idx_start, idx_start to idx_mid, idx_mid to idx_end, idx_end to len
            raw_evasion_d[0:idx_start_in_es] = np.linspace(current_d, start_d, num=idx_start_in_es)
            raw_evasion_d[idx_start_in_es:idx_mid_in_es] = np.linspace(start_d, mid_d, num=idx_mid_in_es - idx_start_in_es)
            raw_evasion_d[idx_mid_in_es:idx_end_in_es] = np.linspace(mid_d, end_d, num=idx_end_in_es - idx_mid_in_es)
            
            # Curvature-aware merge-back (right side)
            n_merge = len(raw_evasion_s) - idx_end_in_es
            if n_merge > 0:
                merge_curvature_conflict = (end_d > 0 and avg_kappa_merge > 0) or (end_d < 0 and avg_kappa_merge < 0)
                if merge_curvature_conflict:
                    t = np.linspace(0, 1, n_merge)
                    s_curve = 3 * t**2 - 2 * t**3
                    raw_evasion_d[idx_end_in_es:] = end_d * (1 - s_curve)
                else:
                    t = np.linspace(0, np.pi/2, n_merge)
                    raw_evasion_d[idx_end_in_es:] = end_d * np.cos(t)

            # check the raw d is not out of boundary
            raw_evasion_valid = self.check_interp_traj_valid(raw_evasion_d, ref_d_right)
            if raw_evasion_valid == False:
                return raw_evasion_d, np.array([]), np.array([]), np.array([]), False
            else:
                x_data, y_data, t_data = self.get_fit_variable(raw_evasion_s, raw_evasion_d, raw_evasion_v)
                start_qp_fit_time = time.time()
                poly_x, poly_y = self.get_qp_fit_coeffs(x_data, y_data, t_data, raw_evasion_s, raw_evasion_v)
                end_qp_fit_time = time.time()
                qp_fit_time = end_qp_fit_time - start_qp_fit_time
                print(f"qp_fit_time: {qp_fit_time*1000:.3f}ms")
                if len(poly_x) == 0 or len(poly_y) == 0:
                    return raw_evasion_d, np.array([]), np.array([]), np.array([]), False
                start_check_closed_time = time.time()
                closedform_evasion_valid = self.check_fit_traj_valid(poly_x, poly_y, t_data, ref_d_right)
                end_check_closed_time = time.time()
                check_closed_time = end_check_closed_time - start_check_closed_time
                print(f"check fit traj time: {check_closed_time*1000:.3f}ms")
                if closedform_evasion_valid == False:
                    # return np.array([]), np.array([]), np.array([]), False
                    return raw_evasion_d, poly_x, poly_y, t_data, False
        
        return raw_evasion_d, poly_x, poly_y, t_data, True
    def get_qp_fit_coeffs(self, x_data, y_data, t_data, raw_evasion_s, raw_evasion_v):
        phi_start = self.initial_ref_spline.get_phi(raw_evasion_s[0])
        phi_end = self.initial_ref_spline.get_phi(raw_evasion_s[-1])
        v_start_x = raw_evasion_v[0]*np.cos(phi_start)
        v_start_y = raw_evasion_v[0]*np.sin(phi_start)
        v_end_x = raw_evasion_v[-1]*np.cos(phi_end)
        v_end_y = raw_evasion_v[-1]*np.sin(phi_end)

        # Call C++ Closed-form Solver
        try:
            poly_x, poly_y = mpc_builder.solve_quintic_spline(
                t_data.astype(np.float64), 
                x_data.astype(np.float64), 
                y_data.astype(np.float64),
                v_start_x, v_start_y, v_end_x, v_end_y
            )
        except Exception as e:
            rospy.logerr(f"C++ Fit failed: {e}")
            return np.array([]), np.array([])

        return poly_x, poly_y
    def get_fit_variable(self, raw_evasion_s, raw_evasion_d, raw_evasion_v):
        waypts_resp = self.converter.get_cartesian(raw_evasion_s, raw_evasion_d)
        x_data = waypts_resp[0, :]
        y_data = waypts_resp[1, :] 
        
        # Call C++
        t_data = mpc_builder.calculate_path_time_cpp(
            x_data.astype(np.float64), 
            y_data.astype(np.float64), 
            raw_evasion_v.astype(np.float64)
        )
        return x_data, y_data, t_data
    
    def check_interp_traj_valid(self, raw_evasion_d, ref_d):
        safe_dist = 0.1
        return np.all(np.abs(raw_evasion_d) <= (np.abs(ref_d)- safe_dist))

    def check_fit_traj_valid(self, poly_x, poly_y, t_data, ref_d):
        safe_dist = 0.1
        x_fit = self.qp_fit.poly(t_data, poly_x)
        y_fit = self.qp_fit.poly(t_data, poly_y)
        resp = self.converter.get_frenet(x_fit, y_fit)
        s, d = resp[0, :], resp[1, :]
        return np.all(np.abs(d) <= (np.abs(ref_d)- safe_dist))
    
    ### Optimal Trajectory Solver ###
    def objective_function(self, d):
        return np.sum((d) ** 2) * 10  + np.sum(np.diff(np.diff(d))**2) * 100 + (np.diff(d)[0] ** 2) * 1000

    ## Constraint functions ##
    def start_on_raceline_constraint(self, d): # And end on raceline
        return np.array([0.02 - abs(d[0] - self.current_d), 0.02 - abs(d[-2]), 0.02 - abs(d[-1])])        

    def obstacle_constraint(self, d):
        distance_to_obstacle = np.abs(d[self.obs_downsampled_indices] - self.obs_downsampled_center_d)
        violation = distance_to_obstacle - self.obs_downsampled_min_dist
        return violation
    
    # Prevents points from jumping trhough obstacles due to resoultion isses
    def consecutive_points_constraint(self, d):
        # Extract the relevant points
        points = d[self.obs_downsampled_indices]
    
        # Check the condition for each pair of consecutive points
        violations = []
        for i in range(len(points) - 1):
            if not ((points[i] > self.obs_downsampled_center_d[i] and points[i+1] > self.obs_downsampled_center_d[i+1]) or
                    (points[i] < self.obs_downsampled_center_d[i] and points[i+1] < self.obs_downsampled_center_d[i+1])):
                violations.append(-1)  # Add a violation as a negative value if the condition is not met
            else:
                violations.append(1)
        return violations

    def turning_radius_constraint(self, d):
        # Calculate curvature at each point using numerical differentiation
        # k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        # x' = self.down_sampled_delta_s, x'' = 0
        
        y_prime = np.diff(d)
        y_prime = np.where(y_prime == 0, 1e-6, y_prime) # Avoid division by zero
        y_prime_prime = np.diff(y_prime)
        y_prime = y_prime[:-1] # Make it the same length as y_prime_prime
        
        kappa = (self.down_sampled_delta_s * y_prime_prime) / ((self.down_sampled_delta_s ** 2) ** (3/2))
        # np.diff losses last two points so we delete them from self.global_traj_kappas
        total_kappa = self.global_traj_kappas - kappa
        violation = self.max_kappa - abs(total_kappa)
        return violation
    
    # The arctan of of (d[1]-d[0])/ delta_s_sample_points < than 45 degrees
    def first_point_constraint(self, d):
        return np.array([self.down_sampled_delta_s - abs(d[1]-d[0])])

    def combined_equality_constraints(self, d):
        return self.start_on_raceline_constraint(d)

    def combined_inequality_constraints(self, d):
        return np.concatenate([self.obstacle_constraint(d), self.consecutive_points_constraint(d), self.turning_radius_constraint(d), self.first_point_constraint(d)]) #

    def solve_sqp(self, d_array, track_boundaries):
        result = minimize(
        self.objective_function, d_array, method='SLSQP', jac='10-point',
        bounds=track_boundaries,
        constraints=[
            {'type': 'eq', 'fun': self.combined_equality_constraints},
            {'type': 'ineq', 'fun': self.combined_inequality_constraints}
            ],
        options={'ftol': 1e-1, 'maxiter': 20, 'disp': False},
        )
        return result

    def group_objects(self, obstacles: list):
        # Group obstacles that are close to each other
        initial_guess_object = obstacles[0]
        for obs in obstacles:
            if obs.d_left > initial_guess_object.d_left:
                initial_guess_object.d_left = obs.d_left
            if obs.d_right < initial_guess_object.d_right:
                initial_guess_object.d_right = obs.d_right
            if obs.s_start < initial_guess_object.s_start:
                initial_guess_object.s_start = obs.s_start
            if obs.s_end > initial_guess_object.s_end:
                initial_guess_object.s_end = obs.s_end
        initial_guess_object.s_center = (initial_guess_object.s_start + initial_guess_object.s_end) / 2
        return initial_guess_object

    def _more_space(self, obstacle: Obstacle, gb_wpnts, gb_idxs):
        left_boundary_mean = np.mean([gb_wpnts[gb_idx].d_left for gb_idx in gb_idxs])
        right_boundary_mean = np.mean([gb_wpnts[gb_idx].d_right for gb_idx in gb_idxs])
        left_gap = abs(left_boundary_mean - obstacle.d_left)
        right_gap = abs(right_boundary_mean + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obstacle.d_right - self.evasion_dist
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obstacle.d_left + self.evasion_dist
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist

            if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
                # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right
    
    ### Visualize SQP Rviz###
    def visualize_sqp(self, evasion_s, evasion_d, evasion_x, evasion_y, evasion_v):
        mrks = MarkerArray()
        if len(evasion_s) == 0:
            pass
        else:
            resp = self.converter.get_cartesian(evasion_s, evasion_d)
            for i in range(len(evasion_s)):
                mrk = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.1
                mrk.scale.y = 0.1
                mrk.scale.z = evasion_v[i] / self.scaled_vmax
                mrk.color.a = 1.0
                mrk.color.g = 0.13
                mrk.color.r = 0.63
                mrk.color.b = 0.94

                mrk.id = i
                mrk.pose.position.x = evasion_x[i]
                mrk.pose.position.y = evasion_y[i]
                mrk.pose.position.z = evasion_v[i] / self.scaled_vmax / 2
                mrk.pose.orientation.w = 1
                mrks.markers.append(mrk)
            self.mrks_pub.publish(mrks)

    def visualize_df(self, evasion_s, evasion_d, evasion_x, evasion_y, evasion_v):
        mrks = MarkerArray()
        if len(evasion_s) == 0:
            pass
        else:
            resp = self.converter.get_cartesian(evasion_s, evasion_d)
            for i in range(len(evasion_s)):
                mrk = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.1
                mrk.scale.y = 0.1
                mrk.scale.z = evasion_v[i] / self.scaled_vmax
                mrk.color.a = 1.0
                mrk.color.g = 0.53
                mrk.color.r = 0.83
                mrk.color.b = 0.34

                mrk.id = i
                mrk.pose.position.x = evasion_x[i]
                mrk.pose.position.y = evasion_y[i]
                mrk.pose.position.z = evasion_v[i] / self.scaled_vmax / 2
                mrk.pose.orientation.w = 1
                mrks.markers.append(mrk)
            self.df_mrks_pub.publish(mrks)

    def visualize_mpc(self, evasion_s, evasion_d, evasion_x, evasion_y, evasion_v):
        mrks = MarkerArray()
        if len(evasion_s) == 0:
            pass
        else:
            for i in range(len(evasion_s)):
                mrk = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.1
                mrk.scale.y = 0.1
                mrk.scale.z = evasion_v[i] / self.scaled_vmax
                mrk.color.a = 1.0
                mrk.color.g = 0.13
                mrk.color.r = 0.93
                mrk.color.b = 0.14

                mrk.id = i
                mrk.pose.position.x = evasion_x[i]
                mrk.pose.position.y = evasion_y[i]
                mrk.pose.position.z = evasion_v[i] / self.scaled_vmax / 2
                mrk.pose.orientation.w = 1
                mrks.markers.append(mrk)
            self.mpc_mrks_pub.publish(mrks)
    
    def visualize_ob_lines(self, start_points, end_points, color=[1.0, 0.0, 0.0], step=5):
        """
        Creates an RViz MarkerArray for lines connecting multiple points, with even sampling.
        
        :param step: The step size for sampling points (default: 10).
        """
        marker_array = MarkerArray()

        if len(start_points) == 0:
            return

        # Sample the points with the specified step size for visualization
        sampled_start_points = start_points[::step]  # Select points with a step interval
        sampled_end_points = end_points[::step]      # Select points with the same step interval

        for i in range(len(sampled_start_points)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "line_visualization"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0  # Alpha (transparency)

            marker.scale.x = 0.05  # Line width
            marker.pose.orientation.w = 1.0  # No rotation

            # Create points for the start and end of the line
            p1 = Point(x=float(sampled_start_points[i, 0]), y=float(sampled_start_points[i, 1]), z=0.0)
            p2 = Point(x=float(sampled_end_points[i, 0]), y=float(sampled_end_points[i, 1]), z=0.0)
            marker.points.append(p1)
            marker.points.append(p2)

            # Add the marker to the array
            marker_array.markers.append(marker)

        # Publish the marker array
        self.ob_line_pub.publish(marker_array)


        
    
    def visualize_raw_traj(self, evasion_s, evasion_d, evasion_x, evasion_y, evasion_v):
        mrks = MarkerArray()
        if len(evasion_s) == 0:
            pass
        else:
            mrk = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
            mrk.type = mrk.LINE_STRIP
            mrk.scale.x = 0.05
            mrk.color.a = 1.0
            mrk.color.r = 1.0

            for i in range(len(evasion_s)):
                point = Point()
                point.x = evasion_x[i]
                point.y = evasion_y[i]
                point.z = 0
                mrk.points.append(point)

            mrks.markers.append(mrk)
            self.raw_mrks_pub.publish(mrks)


    def initialize_converter(self) -> bool:
            """
            Initialize the FrenetConverter object"""
            rospy.wait_for_message("/global_waypoints", WpntArray)

            # Initialize the FrenetConverter object
            converter = FrenetConverter(self.global_waypoints[:, 0], self.global_waypoints[:, 1])
            rospy.loginfo("[Spliner] initialized FrenetConverter object")

            return converter
    
    def initialize_spline(self):
        """
        Initialize the spline Converter object"""

        initial_ref_spline = InitialRefSpline()

        return initial_ref_spline

if __name__ == "__main__":
    SQPAvoidance = SQPAvoidanceNode()
    SQPAvoidance.loop()
