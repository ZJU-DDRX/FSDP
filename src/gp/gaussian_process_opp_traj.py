#!/usr/bin/env python3

import numpy as np
import rospy
import rospkg
from f110_msgs.msg import ObstacleArray, OpponentTrajectory, OppWpnt, WpntArray, ProjOppTraj
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
import time
from scipy.optimize import fmin_l_bfgs_b
from frenet_converter.frenet_converter import FrenetConverter
from gp.Sparse_GP_function import sparse_gp_model_train, sparse_gp_model_predict, sparse_gp_plot
from gp.Sparse_GP_data_selection import data_selection
import warnings
import os
import pandas as pd

from ccma import CCMA

class GaussianProcessOppTraj(object):
    def __init__(self):
        #Node
        rospy.init_node('gaussian_process_opp_traj', anonymous=True)
        self.rate = rospy.Rate(10)  
        self.opp_traj = OpponentTrajectory()
        self.opp_traj_gp = OpponentTrajectory()
        self.opp_positions_in_map_frenet = [] #testing with Rosbag

        #Subscribers
        rospy.Subscriber('/proj_opponent_trajectory', ProjOppTraj, self.proj_opp_traj_cb)
        rospy.Subscriber('/global_waypoints', WpntArray, self.glb_wpnts_cb) # global waypoints

        #Publishers
        self.opp_traj_gp_pub = rospy.Publisher('/opponent_trajectory', OpponentTrajectory, queue_size=10)
        self.opp_traj_marker_pub = rospy.Publisher('/opponent_traj_markerarray', MarkerArray, queue_size=10)

        #Frenet Converter
        rospy.wait_for_message("/global_waypoints", WpntArray)
        self.converter = self.initialize_converter()

        #define data selection mode: Time, All, Kmeans, New
        self.data_selection_mode = "Kmeans" 
        self.debug = True
        self.showsigma = False
        self.sorted_detection_list_fullold = []
        self.dataset_vsmax = 7
        self.dataset_vsmin = 0
        self.dataset_dmax = 5
        self.dataset_dmin = -5
        self.Kmeans_target_size = 400

        #GPpytorch warmup
        # First, define the sparse GP optimizer iteration numbers and inducing numbers
        # self.sgp_training_iterations = 50
        # self.inducing_points_number = 30
        self.sgp_training_iterations = 15
        self.inducing_points_number = 10
        warnings.filterwarnings("ignore", category=UserWarning, module='gpytorch')
        self.gpytorch_warmup()

    #Callback
    def gpytorch_warmup(self):
        warmup_train_x = np.random.rand(100, 2)  # 200 samples, 2-dimensional input
        warmup_train_y = np.sin(warmup_train_x[:, 0]) + np.cos(warmup_train_x[:, 1])  # 1-dimensional output
        warmup_train_y = (warmup_train_y - warmup_train_y.min()) / (warmup_train_y.max() - warmup_train_y.min())
        _, _=sparse_gp_model_train(warmup_train_x, warmup_train_y, self.inducing_points_number, self.sgp_training_iterations, predic_warmup=True, train_viz_disable=True)
        rospy.loginfo("GPpytorch warmup is completed.")

    def proj_opp_traj_cb(self,data: ProjOppTraj):
        self.proj_opp_traj = data

    def glb_wpnts_cb(self, data):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.glb_wpnts = data
        self.track_length = data.wpnts[-1].s_m


    #Functions
    def initialize_converter(self) -> bool:
        
        """Initialize the FrenetConverter object"""

        rospy.wait_for_message("/global_waypoints", WpntArray)

        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
        rospy.loginfo("[Tracking] initialized FrenetConverter object")

        return converter    
        
    #Main Loop
    def get_gp_opp_traj(self):
        # Define the constant kernels with a lower bound for the constant_value parameter
        constant_kernel1_d = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
        constant_kernel2_d = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))
        constant_kernel1_vs = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
        constant_kernel2_vs = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))

        #Define the Gaussian Process kernel
        self.kernel_vs = constant_kernel1_vs * RBF(length_scale=1.0) + constant_kernel2_vs * WhiteKernel(noise_level=1)
        self.kernel_d = constant_kernel1_d * Matern(length_scale=1.0, nu=3/2) + constant_kernel2_d * WhiteKernel(noise_level=1)
        
        
        first_half_lap = True
        self.global_wpnts = rospy.wait_for_message("/global_waypoints", WpntArray)
        self.max_velocity = max([wnpt.vx_mps for wnpt in self.global_wpnts.wpnts])
        ego_s_original = [wnpt.s_m for wnpt in self.global_wpnts.wpnts]
        #pop last point of original ego_s since it is double
        ego_s_original.pop()
        ego_s_doublelap = ego_s_original.copy()
        for i in range(len(ego_s_original)):
            ego_s_doublelap.append(ego_s_original[i]+self.track_length)
            
        #create a oppwpmt lap with velocity 100
        oppwpnts_list = self.make_initial_opponent_trajectory_msg(ego_s_original=ego_s_original)
        proj_opp_traj = ProjOppTraj()
        sorted_detection_list = []
        while not rospy.is_shutdown():
            proj_opp_traj = rospy.wait_for_message('/proj_opponent_trajectory', ProjOppTraj)
            self.lap_count = proj_opp_traj.lapcount
            opp_on_traj = proj_opp_traj.opp_is_on_trajectory
            nr_of_points = proj_opp_traj.nrofpoints
            if opp_on_traj == True and len(proj_opp_traj.detections) != 0:
                if self.debug:
                    print("Lap count:", self.lap_count)
                if self.lap_count <=1:
                    sorted_detection_list_first_lap, around_origin = self.create_sorted_detection_list(proj_opponent_detections=proj_opp_traj.detections, sorted_detection_list=[], ego_s_original=ego_s_original)
                    for i in range(len(sorted_detection_list_first_lap)):
                        sorted_detection_list.append(sorted_detection_list_first_lap[i])
                    
                    opponent_s_sorted, opponent_d_sorted,opponent_vs_sorted, opponent_vd_sorted = self.get_sorted_s_d_vs_vd_lists(sorted_detection_list=sorted_detection_list_first_lap)

                    if first_half_lap:
                        first_half_lap = False
                        ego_s_sorted_halflap = []
                        for i in range(len(ego_s_doublelap)):
                            if ego_s_doublelap[i] > opponent_s_sorted[0] and ego_s_doublelap[i] < opponent_s_sorted[0]+(self.track_length/2):
                                ego_s_sorted_halflap.append(ego_s_doublelap[i])
                        last_ego_s = ego_s_sorted_halflap[-1]
                        first_ego_s = ego_s_sorted_halflap[0]        

                    else:
                        #make a new ego_s_sorted_halflap with the last_ego_s as a statring point and first_ego_s as an end point
                        ego_s_sorted_halflap = []
                        last_ego_s = last_ego_s%self.track_length
                        if last_ego_s > first_ego_s:
                            first_ego_s = first_ego_s+self.track_length

                        for i in range(len(ego_s_doublelap)):
                            if ego_s_doublelap[i] >= last_ego_s and ego_s_doublelap[i] <= first_ego_s:
                                ego_s_sorted_halflap.append(ego_s_doublelap[i])
                        last_ego_s = ego_s_sorted_halflap[-1]
                        first_ego_s = ego_s_sorted_halflap[0] 


                    oppwpnts_list , opp_traj_marker_array = self.get_opponnent_wpnts(whole_lap=False, ego_s_sorted=ego_s_sorted_halflap, opponent_s_sorted=opponent_s_sorted, 
                                                                                    opponent_d_sorted=opponent_d_sorted, opponent_vs_sorted=opponent_vs_sorted, 
                                                                                    opponent_vd_sorted=opponent_vd_sorted, arond_origin=around_origin, oppwpnts_list=oppwpnts_list)
                    
                    opp_traj_gp_msg = self.make_opponent_trajectory_msg(oppwpnts_list=oppwpnts_list, lap_count=self.lap_count, raw_oppenent_traj_msg=proj_opp_traj)
                    #Publish
                    self.opp_traj_gp_pub.publish(opp_traj_gp_msg)
                    self.opp_traj_marker_pub.publish(opp_traj_marker_array) 
                    
                    if around_origin:#reset s value of sorted_detection_list and sort again
                        for i in range(len(sorted_detection_list)):
                            sorted_detection_list[i].s = sorted_detection_list[i].s%self.track_length
                        sorted_detection_list.sort(key=lambda x: x.s)

                    if self.lap_count == 1:
                        self.lap_count = 1.1
                        sorted_detection_list, around_origin = self.create_sorted_detection_list(proj_opponent_detections=proj_opp_traj.detections, sorted_detection_list=sorted_detection_list, ego_s_original=ego_s_original)
                        opponent_s_sorted, opponent_d_sorted,opponent_vs_sorted, opponent_vd_sorted = self.get_sorted_s_d_vs_vd_lists(sorted_detection_list=sorted_detection_list)
                        self.sorted_detection_list_fullold = sorted_detection_list.copy()
                        
                        oppwpnts_list , opp_traj_marker_array = self.get_opponnent_wpnts(whole_lap=True, ego_s_sorted=ego_s_original, opponent_s_sorted=opponent_s_sorted,
                                                                                        opponent_d_sorted=opponent_d_sorted, opponent_vs_sorted=opponent_vs_sorted,
                                                                                        opponent_vd_sorted=opponent_vd_sorted, arond_origin=False, oppwpnts_list=oppwpnts_list)
                        #Publish
                        self.opp_traj_gp_pub.publish(opp_traj_gp_msg)
                        self.opp_traj_marker_pub.publish(opp_traj_marker_array)

                else: #adding additional points to the trajectory
                    
                    if self.data_selection_mode == "Kmeans":

                        sorted_detection_list, around_origin= self.create_sorted_detection_list(proj_opponent_detections=proj_opp_traj.detections, sorted_detection_list=sorted_detection_list, ego_s_original=ego_s_original)
                        opponent_s_sorted, opponent_d_sorted,opponent_vs_sorted, opponent_vd_sorted = self.get_sorted_s_d_vs_vd_lists(sorted_detection_list=sorted_detection_list)

                        temp = sorted_detection_list.copy()
                        opponent_s_sorted_old, opponent_d_sorted_old,opponent_vs_sorted_old, opponent_vd_sorted_old = self.get_sorted_s_d_vs_vd_lists(sorted_detection_list=self.sorted_detection_list_fullold)
                            
                        #fyh:20250228
                        # oppwpnts_list , opp_traj_marker_array = self.get_opponnent_wpnts(whole_lap=False, ego_s_sorted=ego_s_sorted_halflap, opponent_s_sorted=opponent_s_sorted, 
                        #                                                             opponent_d_sorted=opponent_d_sorted, opponent_vs_sorted=opponent_vs_sorted, 
                        #                                                             opponent_vd_sorted=opponent_vd_sorted, arond_origin=around_origin, oppwpnts_list=oppwpnts_list)    
                        oppwpnts_list , opp_traj_marker_array = self.get_opponnent_wpnts_Kmeans(whole_lap=True, ego_s_sorted=ego_s_original, 
                                                                                    opponent_s_sorted=opponent_s_sorted, opponent_s_sorted_old=opponent_s_sorted_old,
                                                                                    opponent_d_sorted=opponent_d_sorted, opponent_d_sorted_old=opponent_d_sorted_old,
                                                                                    opponent_vs_sorted=opponent_vs_sorted, opponent_vs_sorted_old=opponent_vs_sorted_old,
                                                                                    opponent_vd_sorted=opponent_vd_sorted, opponent_vd_sorted_old=opponent_vd_sorted_old,
                                                                                    arond_origin=False, oppwpnts_list=oppwpnts_list)
                        
                        self.sorted_detection_list_fullold.extend(temp)

                        self.sorted_detection_list_fullold.sort(key=lambda x: x.s)

                    elif self.data_selection_mode == "Time" or self.data_selection_mode == "All":     

                        sorted_detection_list, around_origin = self.create_sorted_detection_list(proj_opponent_detections=proj_opp_traj.detections, sorted_detection_list=sorted_detection_list, ego_s_original=ego_s_original)
                        opponent_s_sorted, opponent_d_sorted,opponent_vs_sorted, opponent_vd_sorted = self.get_sorted_s_d_vs_vd_lists(sorted_detection_list=sorted_detection_list)

                        oppwpnts_list , opp_traj_marker_array = self.get_opponnent_wpnts(whole_lap=True, ego_s_sorted=ego_s_original, opponent_s_sorted=opponent_s_sorted,
                                                                                    opponent_d_sorted=opponent_d_sorted, opponent_vs_sorted=opponent_vs_sorted,
                                                                                    opponent_vd_sorted=opponent_vd_sorted, arond_origin=False, oppwpnts_list=oppwpnts_list)

                    opp_traj_gp_msg = self.make_opponent_trajectory_msg(oppwpnts_list=oppwpnts_list, lap_count=self.lap_count, raw_oppenent_traj_msg=proj_opp_traj)
                    #Publish
                    self.opp_traj_gp_pub.publish(opp_traj_gp_msg)
                    self.opp_traj_marker_pub.publish(opp_traj_marker_array)
                    

    #Helper Functions
    def create_sorted_detection_list(self,proj_opponent_detections:list, sorted_detection_list:list, ego_s_original:list):

        """Sort the opponent trajectory based on the s position and return the sorted lists"""
        around_origin = False
        if self.lap_count <=1:
            for i in range(len(proj_opponent_detections)-1):
                if proj_opponent_detections[i].s > proj_opponent_detections[i+1].s:
                    around_origin = True
                if around_origin:
                   proj_opponent_detections[i+1].s = proj_opponent_detections[i+1].s+self.track_length
        else:
            for i in range(len(proj_opponent_detections)-1):
               proj_opponent_detections[i+1].s = proj_opponent_detections[i+1].s%self.track_length
        for i in range(len(proj_opponent_detections)):
            sorted_detection_list.append(proj_opponent_detections[i])
        sorted_detection_list.sort(key=lambda x: x.s)
        printed_s_list = [sorted_detection_list[i].s for i in range(len(sorted_detection_list))]

        if self.data_selection_mode == "Time" or self.data_selection_mode == "Kmeans":

            # delta_s = 2*(self.track_length/len(ego_s_original)) #2*distance between waypoints
            delta_s = 0.25*(self.track_length/len(ego_s_original)) #2*distance between waypoints
            if (len(sorted_detection_list) > 200) and (self.lap_count >= 1):

                # Print the length before processing
                # print(f"Length before processing: {len(sorted_detection_list)}")

                # Initialize a list to store the count of detections in each segment
                segment_counts = []

                # If detections are too close together, remove the older ones
                sorted_detection_list_new = []
                last_s = sorted_detection_list[-1].s
                i = 0
                for x in range(int(self.track_length / delta_s)):
                    if last_s > (x + 1) * delta_s:
                        helper_list = []
                        while (sorted_detection_list[i].s < x * delta_s):
                            helper_list.append(sorted_detection_list[i])
                            i = i + 1
                        if (len(helper_list) > 0):
                            helper_list.sort(key=lambda x: x.time)
                            sorted_detection_list_new.append(helper_list[-1])

                        # Record the number of detections in this segment
                        segment_counts.append(len(helper_list))

                sorted_detection_list = sorted_detection_list_new

                # # Print the length after processing
                # print(f"Length after processing: {len(sorted_detection_list)}")

                # # Print the counts of detections in each segment
                # print(f"Number of detections in each segment: {segment_counts}")

        if self.data_selection_mode == "All" :
            pass
        
        return sorted_detection_list, around_origin

    

    def get_sorted_s_d_vs_vd_lists(self, sorted_detection_list:list): 

        """Sort the opponent trajectory based on the s position and return the sorted lists"""

        opponent_s_sorted = [detection.s for detection in sorted_detection_list]
        opponent_d_sorted = [detection.d for detection in sorted_detection_list]
        opponent_vs_sorted = [detection.vs for detection in sorted_detection_list]
        opponent_vd_sorted = [detection.vd for detection in sorted_detection_list]

        # print("sort S、D")
        # print("now lap count is:", self.lap_count)
        # print(f"The range of S is from {np.array(opponent_s_sorted).min()} to {np.array(opponent_s_sorted).max()}.")
        # print("opponent_s_sorted first few values:", opponent_s_sorted[:5])
        # print("opponent_s_sorted last few values:", opponent_s_sorted[-5:])
        # print("sort finished")

        return opponent_s_sorted, opponent_d_sorted,opponent_vs_sorted, opponent_vd_sorted
    
    def get_opponnent_wpnts(self,whole_lap: bool, ego_s_sorted: list, opponent_s_sorted: list, opponent_d_sorted: list, opponent_vs_sorted: list, opponent_vd_sorted: list, arond_origin: bool, oppwpnts_list: list):

        """Resample the opponent trajectory based on the ego vehicle's s position and return the resampled opponent trajectory (aso return the resampled opponent trajectory as a marker array)"""
        ego_s_sorted_copy = ego_s_sorted.copy()

        #testing with Rosbag
        #split self.opp_positions_in_map_frenet in a s/s/vs/vd list 
        # opp_s = [position[0] for position in self.opp_positions_in_map_frenet]
        # opp_d = [position[1] for position in self.opp_positions_in_map_frenet]
        # opp_vs = [position[2] for position in self.opp_positions_in_map_frenet]
        # opp_vd = [position[3] for position in self.opp_positions_in_map_frenet]
        if whole_lap:
            
            #Predict the d coordinate with CCMA in case of a whole lap
            #stretch the s and d list to ensure that the CCMA works smoothly around the origin
            opp_s_copy = opponent_s_sorted.copy()
            opp_d_copy = opponent_d_sorted.copy()
            opp_s_copy.insert(0,opp_s_copy[-1])
            opp_d_copy.insert(0,opp_d_copy[-1])
            opp_s_copy.append(opp_s_copy[1])
            opp_d_copy.append(opp_d_copy[1])

            #convert to cartesian
            noisy_xy_points=self.converter.get_cartesian(opp_s_copy, opp_d_copy)
            noisy_xy_points = noisy_xy_points.transpose()
            #smooth the trajectory with CCMA
            ccma = CCMA(w_ma=5, w_cc=3)
            smoothed_xy_points = ccma.filter(noisy_xy_points)

            #convert back to frenet
            smoothed_sd_points = self.converter.get_frenet(smoothed_xy_points[:, 0], smoothed_xy_points[:, 1])
            #sort the points based on s
            smoothed_s_points = smoothed_sd_points[0]
            smoothed_d_points = smoothed_sd_points[1]

            smoothed_s_points, smoothed_d_points = zip(*sorted(zip(smoothed_s_points, smoothed_d_points)))

            #interpolate the smoothed trajectory on the same s points as the ego vehicles trajectory
            
            d_pred_CCMA = np.interp(ego_s_sorted, smoothed_s_points, smoothed_d_points)


            #Preparing the data for the Gaussian Process
            #prepend the last points of the opponent trajectory to the beginning of the list
            n=-1
            nr_of_points_pre = 0
            for i in range(len(opponent_s_sorted)):
                if abs(opponent_s_sorted[n]-self.track_length) < 3: #go 3 m in negative direction
                    opponent_s_sorted.insert(0,opponent_s_sorted[n]-self.track_length)
                    n=n-1
                    nr_of_points_pre = nr_of_points_pre+1
            n=-1
            for i in range(nr_of_points_pre):
                opponent_d_sorted.insert(0,opponent_d_sorted[n])
                n=n-1
            n=-1
            for i in range(nr_of_points_pre):
                opponent_vs_sorted.insert(0,opponent_vs_sorted[n])
                n=n-1
            n=-1
            for i in range(nr_of_points_pre):
                opponent_vd_sorted.insert(0,opponent_vd_sorted[n])
                n=n-1
            n=-1
            #prepend last points of the ego_s_sorted to the beginning of the list as a negative value
            nr_of_points_ego_s = 0
            for i in range(len(ego_s_sorted_copy)):
                if abs(ego_s_sorted_copy[n]-self.track_length) < 3:
                    ego_s_sorted_copy.insert(0,ego_s_sorted_copy[n]-self.track_length)
                    n=n-1
                    nr_of_points_ego_s = nr_of_points_ego_s+1
            #append the first points of the opponent trajectory to the end of the list
            n=0
            nr_of_points_app = 0
            for i in range(len(opponent_s_sorted)):
                if abs(opponent_s_sorted[nr_of_points_pre+n]) < 3:
                    opponent_s_sorted.append(opponent_s_sorted[nr_of_points_pre+n]+self.track_length)
                    n=n+1
                    nr_of_points_app = nr_of_points_app+1
            n=0
            for i in range(nr_of_points_app):
                opponent_d_sorted.append(opponent_d_sorted[nr_of_points_pre+n])
                n=n+1
            n=0
            for i in range(nr_of_points_app):
                opponent_vs_sorted.append(opponent_vs_sorted[nr_of_points_pre+n])
                n=n+1
            n=0
            for i in range(nr_of_points_app):
                opponent_vd_sorted.append(opponent_vd_sorted[nr_of_points_pre+n])
                n=n+1
            n=0

            #append first 3m of the ego_s_sorted to the end of the list as a value bigger than the track_length
            for i in range(len(ego_s_sorted_copy)):
                if abs(ego_s_sorted_copy[nr_of_points_ego_s+n]) < 3:
                    ego_s_sorted_copy.append(ego_s_sorted_copy[nr_of_points_ego_s+n]+self.track_length)
                    n=n+1
            train_s = np.array([opponent_s_sorted[i] for i in range(len(opponent_s_sorted))])
            train_d = np.array([opponent_d_sorted[i] for i in range(len(opponent_d_sorted))])
            train_vs = np.array([opponent_vs_sorted[i] for i in range(len(opponent_vs_sorted))])
            train_vd = np.array([opponent_vd_sorted[i] for i in range(len(opponent_vd_sorted))])
        else:
            train_s = np.array([opponent_s_sorted[i] for i in range(len(opponent_s_sorted))])
            train_d = np.array([opponent_d_sorted[i] for i in range(len(opponent_d_sorted))])
            train_vs = np.array([opponent_vs_sorted[i] for i in range(len(opponent_vs_sorted))])
            train_vd = np.array([opponent_vd_sorted[i] for i in range(len(opponent_vd_sorted))])

        opponent_s_sorted_reshape = train_s.reshape(-1, 1)
        opponent_d_sorted_reshape = train_d.reshape(-1, 1)
        opponent_vs_sorted_reshape = train_vs.reshape(-1, 1)
        opponent_vd_sorted_reshape = train_vd.reshape(-1, 1)

        #Define a range of s values for prediction
        ego_s_sorted_nparray = np.array(ego_s_sorted_copy)
        s_pred = ego_s_sorted_nparray.reshape(-1, 1)

        #Fit the Gaussian Process Regressor to the data
        # def optimizer_wrapper(obj_func, initial_theta, bounds):
        #     solution, function_value, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
        #     return solution, function_value
        
        #Fit Vs
        # Method 1: Using GaussianProcessRegressor
        # start_time_gpmethod_train_vs = time.time()
        # gpr_vs = GaussianProcessRegressor(kernel=self.kernel_vs, optimizer=optimizer_wrapper)
        # gpr_vs.fit(opponent_s_sorted_reshape, opponent_vs_sorted_reshape)
        # end_time_gpmethod_train_vs = time.time()

        # start_time_gpmethod_predict_vs = time.time()
        # vs_pred, sigma_vs = gpr_vs.predict(s_pred, return_std=True)
        # end_time_gpmethod_predict_vs = time.time()

        # Method 2: Using SparseGPRegression (GPyTorch)
        start_time_gpytorch_train_vs = time.time()
        model_vs, likelihood_vs = sparse_gp_model_train(
            opponent_s_sorted_reshape,  
            opponent_vs_sorted_reshape,  
            self.inducing_points_number,
            self.sgp_training_iterations,
            train_viz_disable=True
        )
        end_time_gpytorch_train_vs = time.time()

        start_time_gpytorch_predict_vs = time.time()
        vs_pred, sigma_vs, inducing_points_value_vs = sparse_gp_model_predict(model_vs, likelihood_vs, s_pred)
        end_time_gpytorch_predict_vs = time.time()

        if self.debug:
            print("\nGP Vs Data Size:")
            # print("Raw data, first 5 s values:", opponent_s_sorted_reshape[:5])
            # print("Raw data, last 5 s values:", opponent_s_sorted_reshape[-5:])
            # print("Raw data, maximum s value:", max(opponent_s_sorted_reshape))
            # print("Raw data, minimum s value:", min(opponent_s_sorted_reshape))
            print(f"  Training data size : {opponent_s_sorted_reshape.shape}")
            
            # print("\nVS Calculation Times:")
            # print(f"Method 1 (GaussianProcessRegressor):")
            # print(f"  Training time: {end_time_gpmethod_train_vs - start_time_gpmethod_train_vs:.4f} s")
            # print(f"  Prediction time: {end_time_gpmethod_predict_vs - start_time_gpmethod_predict_vs:.4f} s")

            # print(f"Method 2 (SparseGPRegression):")
            vs_training_time = end_time_gpytorch_train_vs - start_time_gpytorch_train_vs
            print(f"  Training time: {vs_training_time:.4f} s")

            vs_predict_time = end_time_gpytorch_predict_vs - start_time_gpytorch_predict_vs
            print(f"  Prediction time: {vs_predict_time:.4f} s")


        if self.data_selection_mode == "Kmeans" and self.lap_count == 1.1:
            self.model_vs_old = model_vs
            self.likelihood_vs_old = likelihood_vs
            if self.debug:
                print("  Save vs model")

        
        #if not already predicted with CCMA fit D with GP
        if not whole_lap or (self.lap_count == 1.1 and self.data_selection_mode == "Kmeans"):
            # Method 1: Using GaussianProcessRegressor
            # start_time_gpmethod_train_d = time.time()
            # gpr_d = GaussianProcessRegressor(kernel=self.kernel_d, optimizer=optimizer_wrapper)
            # gpr_d.fit(opponent_s_sorted_reshape, opponent_d_sorted_reshape)
            # end_time_gpmethod_train_d = time.time()

            # start_time_gpmethod_predict_d = time.time()
            # d_pred_GP, sigma_d = gpr_d.predict(s_pred, return_std=True)
            # end_time_gpmethod_predict_d = time.time()

            # Method 2: Using SparseGPRegression (GPyTorch)
            start_time_gpytorch_train_d = time.time()
            model_d, likelihood_d = sparse_gp_model_train(
                opponent_s_sorted_reshape,  
                opponent_d_sorted_reshape,  
                self.inducing_points_number,
                self.sgp_training_iterations,
                train_viz_disable=True
            )
            end_time_gpytorch_train_d = time.time()

            start_time_gpytorch_predict_d = time.time()
            d_pred_GP, sigma_d, inducing_points_value_d = sparse_gp_model_predict(model_d, likelihood_d, s_pred)
            end_time_gpytorch_predict_d = time.time()

            # print(f"The range of S is from {opponent_s_sorted_reshape.min()} to {opponent_s_sorted_reshape.max()}.")

            if self.debug:
                # # Print data size
                print("\nGP D Data Size:")
                print(f"  Training data size : {opponent_s_sorted_reshape.shape}")
                # print(f"The range of S is from {opponent_s_sorted_reshape.min()} to {opponent_s_sorted_reshape.max()}.")

                # # Print calculation times
                # print("\nD Calculation Times:")
                # print(f"Method 1 (GaussianProcessRegressor):")
                # print(f"  Training time: {end_time_gpmethod_train_d - start_time_gpmethod_train_d:.4f} s")
                # print(f"  Prediction time: {end_time_gpmethod_predict_d - start_time_gpmethod_predict_d:.4f} s")

                # print(f"Method 2 (SparseGPRegression):")
                d_training_time = end_time_gpytorch_train_d - start_time_gpytorch_train_d
                print(f"  Training time: {d_training_time:.4f} s")
                d_predict_time = end_time_gpytorch_predict_d - start_time_gpytorch_predict_d
                print(f"  Prediction time: {d_predict_time:.4f} s")

            if self.lap_count == 1.1 and self.data_selection_mode == "Kmeans":
                self.model_d_old = model_d
                self.likelihood_d_old = likelihood_d
                if self.debug:
                    print("  Save d model")


        #shorten the copy lists (that was changed) to the length of the original ego_s
        if whole_lap:
            n=0
            for i in range(len(ego_s_sorted_copy)):
                if ego_s_sorted_copy[i-n] >= self.track_length or ego_s_sorted_copy[i-n] < 0:
                    ego_s_sorted_copy.pop(i-n)
                    if not whole_lap: #(CCMA)
                        d_pred_GP = np.delete(d_pred_GP, i-n)
                    vs_pred = np.delete(vs_pred, i-n) 
                    n+=1

        if whole_lap: #(CCMA)
        #if False:
            resampled_opponent_d = d_pred_CCMA
        else:
            resampled_opponent_d = d_pred_GP
        resampled_opponent_vs = vs_pred
        resampled_opponent_vd = np.interp(ego_s_sorted, opponent_s_sorted, opponent_vd_sorted)

        if arond_origin:
            ego_s=[ego_s_sorted[i]%self.track_length for i in range(len(ego_s_sorted_copy))]
        else:
            ego_s=ego_s_sorted_copy

        
        resampled_wpnts_xy = self.converter.get_cartesian(ego_s , resampled_opponent_d.tolist())

        
        # replace all the entries where there is a corresponding ego_s with the interpolated values
        i=0

        for i in range(len(oppwpnts_list)):
                for j in range(len(ego_s)):
                    if abs(ego_s[j]-oppwpnts_list[i].s_m) < 1e-8:
                        oppwpnts_list[i].x_m = resampled_wpnts_xy[0][j]
                        oppwpnts_list[i].y_m = resampled_wpnts_xy[1][j]
                        oppwpnts_list[i].d_m = resampled_opponent_d[j]
                        oppwpnts_list[i].proj_vs_mps = resampled_opponent_vs[j] 
                        oppwpnts_list[i].vd_mps = resampled_opponent_vd[j] 
                        if not whole_lap:
                            oppwpnts_list[i].d_var = sigma_d[j]
                        else:
                            oppwpnts_list[i].d_var = 0
                        oppwpnts_list[i].vs_var = sigma_vs[j]
        opp_traj_marker_array = self._visualize_opponent_wpnts(oppwpnts_list=oppwpnts_list)

        # if self.debug and self.lap_count >= 1.1:
        #     self.save_monitor_data(
        #         oppwpnts_list, 
        #         [],    
        #         [],   
        #         [], 
        #         [],
        #         [],
        #         [],
        #         self.lap_count,
        #         "CCMA",
        #         [], 
        #         []
        #     )

        return oppwpnts_list , opp_traj_marker_array

    def get_opponnent_wpnts_Kmeans(self,whole_lap: bool, ego_s_sorted: list, opponent_s_sorted: list, opponent_s_sorted_old: list, opponent_d_sorted: list, opponent_d_sorted_old: list, opponent_vs_sorted: list, opponent_vs_sorted_old: list, opponent_vd_sorted: list, opponent_vd_sorted_old: list, arond_origin: bool, oppwpnts_list: list):

        """Resample the opponent trajectory based on the ego vehicle's s position and return the resampled opponent trajectory (aso return the resampled opponent trajectory as a marker array)"""
        ego_s_sorted_copy = ego_s_sorted.copy()

        #testing with Rosbag
        #split self.opp_positions_in_map_frenet in a s/s/vs/vd list 
        # opp_s = [position[0] for position in self.opp_positions_in_map_frenet]
        # opp_d = [position[1] for position in self.opp_positions_in_map_frenet]
        # opp_vs = [position[2] for position in self.opp_positions_in_map_frenet]
        # opp_vd = [position[3] for position in self.opp_positions_in_map_frenet]
        if whole_lap:

            #Preparing the data for the Gaussian Process
            #prepend the last points of the opponent trajectory to the beginning of the list
            n=-1
            nr_of_points_pre = 0
            for i in range(len(opponent_s_sorted)):
                if abs(opponent_s_sorted[n]-self.track_length) < 3: #go 3 m in negative direction
                    opponent_s_sorted.insert(0,opponent_s_sorted[n]-self.track_length)
                    n=n-1
                    nr_of_points_pre = nr_of_points_pre+1
            n=-1
            for i in range(nr_of_points_pre):
                opponent_d_sorted.insert(0,opponent_d_sorted[n])
                n=n-1
            n=-1
            for i in range(nr_of_points_pre):
                opponent_vs_sorted.insert(0,opponent_vs_sorted[n])
                n=n-1
            n=-1
            for i in range(nr_of_points_pre):
                opponent_vd_sorted.insert(0,opponent_vd_sorted[n])
                n=n-1
            n=-1
            #prepend last points of the ego_s_sorted to the beginning of the list as a negative value
            nr_of_points_ego_s = 0
            for i in range(len(ego_s_sorted_copy)):
                if abs(ego_s_sorted_copy[n]-self.track_length) < 3:
                    ego_s_sorted_copy.insert(0,ego_s_sorted_copy[n]-self.track_length)
                    n=n-1
                    nr_of_points_ego_s = nr_of_points_ego_s+1
            #append the first points of the opponent trajectory to the end of the list
            n=0
            nr_of_points_app = 0
            for i in range(len(opponent_s_sorted)):
                if abs(opponent_s_sorted[nr_of_points_pre+n]) < 3:
                    opponent_s_sorted.append(opponent_s_sorted[nr_of_points_pre+n]+self.track_length)
                    n=n+1
                    nr_of_points_app = nr_of_points_app+1
            n=0
            for i in range(nr_of_points_app):
                opponent_d_sorted.append(opponent_d_sorted[nr_of_points_pre+n])
                n=n+1
            n=0
            for i in range(nr_of_points_app):
                opponent_vs_sorted.append(opponent_vs_sorted[nr_of_points_pre+n])
                n=n+1
            n=0
            for i in range(nr_of_points_app):
                opponent_vd_sorted.append(opponent_vd_sorted[nr_of_points_pre+n])
                n=n+1
            n=0

            #append first 3m of the ego_s_sorted to the end of the list as a value bigger than the track_length
            for i in range(len(ego_s_sorted_copy)):
                if abs(ego_s_sorted_copy[nr_of_points_ego_s+n]) < 3:
                    ego_s_sorted_copy.append(ego_s_sorted_copy[nr_of_points_ego_s+n]+self.track_length)
                    n=n+1
            train_s = np.array([opponent_s_sorted[i] for i in range(len(opponent_s_sorted))])
            train_d = np.array([opponent_d_sorted[i] for i in range(len(opponent_d_sorted))])
            train_vs = np.array([opponent_vs_sorted[i] for i in range(len(opponent_vs_sorted))])
            train_vd = np.array([opponent_vd_sorted[i] for i in range(len(opponent_vd_sorted))])
        else:
            train_s = np.array([opponent_s_sorted[i] for i in range(len(opponent_s_sorted))])
            train_d = np.array([opponent_d_sorted[i] for i in range(len(opponent_d_sorted))])
            train_vs = np.array([opponent_vs_sorted[i] for i in range(len(opponent_vs_sorted))])
            train_vd = np.array([opponent_vd_sorted[i] for i in range(len(opponent_vd_sorted))])

        opponent_s_sorted_reshape = train_s.reshape(-1, 1)
        opponent_d_sorted_reshape = train_d.reshape(-1, 1)
        opponent_vs_sorted_reshape = train_vs.reshape(-1, 1)
        opponent_vd_sorted_reshape = train_vd.reshape(-1, 1)

        #Define a range of s values for prediction
        ego_s_sorted_nparray = np.array(ego_s_sorted_copy)
        s_pred = ego_s_sorted_nparray.reshape(-1, 1)

        #deal old data gppre
        if whole_lap:
            
            # Preparing the data for the Gaussian Process
            # prepend the last points of the opponent trajectory to the beginning of the list
            n = -1
            nr_of_points_pre = 0
            for i in range(len(opponent_s_sorted_old)):
                if abs(opponent_s_sorted_old[n] - self.track_length) < 3:  # go 3 m in negative direction
                    opponent_s_sorted_old.insert(0, opponent_s_sorted_old[n] - self.track_length)
                    n = n - 1
                    nr_of_points_pre = nr_of_points_pre + 1
            n = -1
            for i in range(nr_of_points_pre):
                opponent_d_sorted_old.insert(0, opponent_d_sorted_old[n])
                n = n - 1
            n = -1
            for i in range(nr_of_points_pre):
                opponent_vs_sorted_old.insert(0, opponent_vs_sorted_old[n])
                n = n - 1
            n = -1
            for i in range(nr_of_points_pre):
                opponent_vd_sorted_old.insert(0, opponent_vd_sorted_old[n])
                n = n - 1
            n = -1
            # prepend last points of the ego_s_sorted to the beginning of the list as a negative value
            nr_of_points_ego_s = 0
            # for i in range(len(ego_s_sorted_copy)):
            #     if abs(ego_s_sorted_copy[n] - self.track_length) < 3:
            #         ego_s_sorted_copy.insert(0, ego_s_sorted_copy[n] - self.track_length)
            #         n = n - 1
            #         nr_of_points_ego_s = nr_of_points_ego_s + 1
            # append the first points of the opponent trajectory to the end of the list
            n = 0
            nr_of_points_app = 0
            for i in range(len(opponent_s_sorted_old)):
                if abs(opponent_s_sorted_old[nr_of_points_pre + n]) < 3:
                    opponent_s_sorted_old.append(opponent_s_sorted_old[nr_of_points_pre + n] + self.track_length)
                    n = n + 1
                    nr_of_points_app = nr_of_points_app + 1
            n = 0
            for i in range(nr_of_points_app):
                opponent_d_sorted_old.append(opponent_d_sorted_old[nr_of_points_pre + n])
                n = n + 1
            n = 0
            for i in range(nr_of_points_app):
                opponent_vs_sorted_old.append(opponent_vs_sorted_old[nr_of_points_pre + n])
                n = n + 1
            n = 0
            for i in range(nr_of_points_app):
                opponent_vd_sorted_old.append(opponent_vd_sorted_old[nr_of_points_pre + n])
                n = n + 1
            n = 0

            # append first 3m of the ego_s_sorted to the end of the list as a value bigger than the track_length
            # for i in range(len(ego_s_sorted_copy)):
            #     if abs(ego_s_sorted_copy[nr_of_points_ego_s + n]) < 3:
            #         ego_s_sorted_copy.append(ego_s_sorted_copy[nr_of_points_ego_s + n] + self.track_length)
            #         n = n + 1
            train_s = np.array([opponent_s_sorted_old[i] for i in range(len(opponent_s_sorted_old))])
            train_d = np.array([opponent_d_sorted_old[i] for i in range(len(opponent_d_sorted_old))])
            train_vs = np.array([opponent_vs_sorted_old[i] for i in range(len(opponent_vs_sorted_old))])
            train_vd = np.array([opponent_vd_sorted_old[i] for i in range(len(opponent_vd_sorted_old))])
        else:
            train_s = np.array([opponent_s_sorted_old[i] for i in range(len(opponent_s_sorted_old))])
            train_d = np.array([opponent_d_sorted_old[i] for i in range(len(opponent_d_sorted_old))])
            train_vs = np.array([opponent_vs_sorted_old[i] for i in range(len(opponent_vs_sorted_old))])
            train_vd = np.array([opponent_vd_sorted_old[i] for i in range(len(opponent_vd_sorted_old))])

        opponent_s_sorted_reshape_old = train_s.reshape(-1, 1)
        opponent_d_sorted_reshape_old = train_d.reshape(-1, 1)
        opponent_vs_sorted_reshape_old = train_vs.reshape(-1, 1)
        opponent_vd_sorted_reshape_old = train_vd.reshape(-1, 1)


        #Fit the Gaussian Process Regressor to the data
        def optimizer_wrapper(obj_func, initial_theta, bounds):
            solution, function_value, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
            return solution, function_value
        




        # Method 2: Using SparseGPRegression (GPyTorch)
        


        opponent_s_reduced, opponent_vs_reduced = data_selection(self.model_vs_old, self.likelihood_vs_old, opponent_s_sorted_reshape_old, opponent_vs_sorted_reshape_old, opponent_s_sorted_reshape, opponent_vs_sorted_reshape, self.Kmeans_target_size, self.dataset_vsmin, self.dataset_vsmax)
        

        #Fit Vs
        # Method 1: Using GaussianProcessRegressor
        start_time_gpmethod_train_vs = time.time()
        gpr_vs = GaussianProcessRegressor(kernel=self.kernel_vs, optimizer=optimizer_wrapper)
        gpr_vs.fit(opponent_s_reduced, opponent_vs_reduced)
        end_time_gpmethod_train_vs = time.time()

        start_time_gpmethod_predict_vs = time.time()
        vs_pred0, sigma_vs0 = gpr_vs.predict(s_pred, return_std=True)
        end_time_gpmethod_predict_vs = time.time()

        start_time_gpytorch_train_vs = time.time()
        model_vs, likelihood_vs = sparse_gp_model_train(opponent_s_reduced, opponent_vs_reduced, self.inducing_points_number, self.sgp_training_iterations, train_viz_disable=True)
        end_time_gpytorch_train_vs = time.time()
        


        start_time_gpytorch_predict_vs = time.time()
        vs_pred, sigma_vs, inducing_points_value_vs = sparse_gp_model_predict(model_vs, likelihood_vs, s_pred)
        end_time_gpytorch_predict_vs = time.time()

        if self.debug:
            print("\nKmeansGP Vs Data Size:")
            # print("After clustering, first 5 s values:", opponent_s_reduced[:5])
            # print("After clustering, last 5 s values:", opponent_s_reduced[-5:])
            # print("After clustering, maximum s value:", max(opponent_s_reduced))
            # print("After clustering, minimum s value:", min(opponent_s_reduced))
            # print("Raw data, first 5 s values:", opponent_s_sorted_reshape[:5])
            # print("Raw data, last 5 s values:", opponent_s_sorted_reshape[-5:])
            # print("Raw data, maximum s value:", max(opponent_s_sorted_reshape))
            # print("Raw data, minimum s value:", min(opponent_s_sorted_reshape))
            print(f"  Training data size : {opponent_s_reduced.shape}")
            print(f"  New data size : {opponent_s_sorted_reshape.shape}")
            print(f"  Old data size : {opponent_s_sorted_reshape_old.shape}")
            
            # print("\nVS Calculation Times:")
            print(f"Method 1 (GaussianProcessRegressor):")
            print(f"  Training time: {end_time_gpmethod_train_vs - start_time_gpmethod_train_vs:.4f} s")
            print(f"  Prediction time: {end_time_gpmethod_predict_vs - start_time_gpmethod_predict_vs:.4f} s")

            print(f"Method 2 (SparseGPRegression):")
            vs_training_time = end_time_gpytorch_train_vs - start_time_gpytorch_train_vs
            print(f"  Training time: {vs_training_time:.4f} s")
            vs_predict_time = end_time_gpytorch_predict_vs - start_time_gpytorch_predict_vs
            print(f"  Prediction time: {vs_predict_time:.4f} s")

        self.model_vs_old = model_vs
        self.likelihood_vs_old = likelihood_vs
        if self.debug:
            print("Save vs model")


        
        # D GP
        # if not whole_lap:
        # Method 1: Using GaussianProcessRegressor


        # Method 2: Using SparseGPRegression (GPyTorch)

        opponent_s_reduced, opponent_d_reduced = data_selection(self.model_d_old, self.likelihood_d_old, opponent_s_sorted_reshape_old, opponent_d_sorted_reshape_old, opponent_s_sorted_reshape, opponent_d_sorted_reshape, self.Kmeans_target_size, self.dataset_dmin, self.dataset_dmax)
        
        start_time_gpmethod_train_d = time.time()
        gpr_d = GaussianProcessRegressor(kernel=self.kernel_d, optimizer=optimizer_wrapper)
        gpr_d.fit(opponent_s_reduced, opponent_d_reduced)
        end_time_gpmethod_train_d = time.time()

        start_time_gpmethod_predict_d = time.time()
        d_pred_GP0, sigma_d0 = gpr_d.predict(s_pred, return_std=True)
        end_time_gpmethod_predict_d = time.time()

        start_time_gpytorch_train_d = time.time()
        model_d, likelihood_d = sparse_gp_model_train(opponent_s_reduced, opponent_d_reduced, self.inducing_points_number, self.sgp_training_iterations, train_viz_disable=True)
        end_time_gpytorch_train_d = time.time()

        start_time_gpytorch_predict_d = time.time()
        d_pred_GP, sigma_d, inducing_points_value_d = sparse_gp_model_predict(model_d, likelihood_d, s_pred)
        end_time_gpytorch_predict_d = time.time()


        # # Print data size
        if self.debug:
            print("\nKmeansGP D Data Size:")
            print(f"  Training data size : {opponent_d_reduced.shape}")
            print(f"  New data size : {opponent_d_sorted_reshape.shape}")
            print(f"  Old data size : {opponent_d_sorted_reshape_old.shape}")

            # # Print calculation times
            # print("\nD Calculation Times:")
            print(f"Method 1 (GaussianProcessRegressor):")
            print(f"  Training time: {end_time_gpmethod_train_d - start_time_gpmethod_train_d:.4f} s")
            print(f"  Prediction time: {end_time_gpmethod_predict_d - start_time_gpmethod_predict_d:.4f} s")

            print(f"Method 2 (SparseGPRegression):")
            d_training_time = end_time_gpytorch_train_d - start_time_gpytorch_train_d
            print(f"  Training time: {d_training_time:.4f} s")
            d_predict_time = end_time_gpytorch_predict_d - start_time_gpytorch_predict_d
            print(f"  Prediction time: {d_predict_time:.4f} s")

        self.model_d_old = model_d
        self.likelihood_d_old = likelihood_d
        if self.debug:
            print("  Save d model")


        #shorten the copy lists (that was changed) to the length of the original ego_s
        if whole_lap:
            n=0
            for i in range(len(ego_s_sorted_copy)):
                if ego_s_sorted_copy[i-n] >= self.track_length or ego_s_sorted_copy[i-n] < 0:
                    ego_s_sorted_copy.pop(i-n)
                    # if not whole_lap: #(CCMA)
                    #     d_pred_GP = np.delete(d_pred_GP, i-n)
                    d_pred_GP = np.delete(d_pred_GP, i-n)
                    vs_pred = np.delete(vs_pred, i-n) 
                    n+=1
                    if self.debug:
                        sigma_d = np.delete(sigma_d, i-n) 

        
        # if whole_lap: #(CCMA)
        # #if False:
        #     resampled_opponent_d = d_pred_CCMA
        # else:
        #     resampled_opponent_d = d_pred_GP
        resampled_opponent_d = d_pred_GP
        resampled_opponent_vs = vs_pred
        resampled_opponent_vd = np.interp(ego_s_sorted, opponent_s_sorted, opponent_vd_sorted)


            # for i in range(len(oppwpnts_list)):
            #     for j in range(len(ego_s)):
            #         if abs(ego_s[j]-oppwpnts_list[i].s_m) < 1e-8:
                        # oppwpnts_list_up[i].x_m = resampled_wpnts_xy_up[0][j]
                        # oppwpnts_list_up[i].y_m = resampled_wpnts_xy_up[1][j]
                        # oppwpnts_list_down[i].x_m = resampled_wpnts_xy_down[0][j]
                        # oppwpnts_list_down[i].y_m = resampled_wpnts_xy_down[1][j]

        if arond_origin:
            ego_s=[ego_s_sorted[i]%self.track_length for i in range(len(ego_s_sorted_copy))]
        else:
            ego_s=ego_s_sorted_copy

        resampled_wpnts_xy = self.converter.get_cartesian(ego_s , resampled_opponent_d.tolist())

        if self.debug:
            resampled_opponent_d_up = resampled_opponent_d + sigma_d
            resampled_wpnts_xy_up = self.converter.get_cartesian(ego_s , resampled_opponent_d_up.tolist()) 
            resampled_opponent_d_down = resampled_opponent_d - sigma_d
            resampled_wpnts_xy_down = self.converter.get_cartesian(ego_s , resampled_opponent_d_down.tolist())
        
        # replace all the entries where there is a corresponding ego_s with the interpolated values
        i=0

        for i in range(len(oppwpnts_list)):
                for j in range(len(ego_s)):
                    if abs(ego_s[j]-oppwpnts_list[i].s_m) < 1e-8:
                        oppwpnts_list[i].x_m = resampled_wpnts_xy[0][j]
                        oppwpnts_list[i].y_m = resampled_wpnts_xy[1][j]
                        oppwpnts_list[i].d_m = resampled_opponent_d[j]
                        oppwpnts_list[i].proj_vs_mps = resampled_opponent_vs[j] 
                        oppwpnts_list[i].vd_mps = resampled_opponent_vd[j] 
                        # if not whole_lap:
                        #     oppwpnts_list[i].d_var = sigma_d[j]
                        # else:
                        #     oppwpnts_list[i].d_var = 0
                        if self.showsigma == True:
                            oppwpnts_list[i].d_var = sigma_d[j]
                        else:
                            oppwpnts_list[i].d_var = 0
                        oppwpnts_list[i].vs_var = sigma_vs[j]
        opp_traj_marker_array = self._visualize_opponent_wpnts(oppwpnts_list=oppwpnts_list)

        if self.debug and self.lap_count >= 1.0:
            self.save_monitor_data(
                oppwpnts_list, 
                inducing_points_value_vs,    
                inducing_points_value_d,   
                vs_training_time, 
                d_training_time,
                vs_predict_time,
                d_predict_time,
                self.lap_count,
                "OURS",
                resampled_wpnts_xy_up,
                resampled_wpnts_xy_down,
            )

        return oppwpnts_list , opp_traj_marker_array

    def make_initial_opponent_trajectory_msg(self, ego_s_original:list):

         #make trajectory with velocity 100 for the first half lap
        resampled_wpnts_xy_original = self.converter.get_cartesian(ego_s_original , np.zeros(len(ego_s_original)).tolist())
        oppwpnts_list = []
        i=0
        
        for i in range(len(ego_s_original)):
            oppwpnts = OppWpnt()
            oppwpnts.x_m = resampled_wpnts_xy_original[0][i]
            oppwpnts.y_m = resampled_wpnts_xy_original[1][i]
            oppwpnts.s_m = ego_s_original[i]
            oppwpnts.d_m = 0
            oppwpnts.proj_vs_mps = 100
            oppwpnts.vd_mps = 0
            oppwpnts_list.append(oppwpnts)
        return oppwpnts_list



    def make_opponent_trajectory_msg(self, oppwpnts_list: list, lap_count: int, raw_oppenent_traj_msg: ObstacleArray):
         
        """Make the opponent trajectory message and return it"""

        opponent_trajectory_msg = OpponentTrajectory()
        opponent_trajectory_msg.header.seq = lap_count
        opponent_trajectory_msg.header.stamp = rospy.Time.now()
        opponent_trajectory_msg.header.frame_id = "opponent_trajectory"
        opponent_trajectory_msg.lap_count = lap_count
        opponent_trajectory_msg.oppwpnts = oppwpnts_list

        return opponent_trajectory_msg

    
    
    def _visualize_opponent_wpnts(self, oppwpnts_list: list):

        """Visualize the resampled opponent trajectory as a marker array"""
        opp_traj_marker_array = MarkerArray()
    
        i=0
        for i in range(len(oppwpnts_list)):
            marker_height = oppwpnts_list[i].proj_vs_mps/self.max_velocity

            marker = Marker(header=rospy.Header(frame_id="map"), id = i, type = Marker.CYLINDER)
            marker.pose.position.x = oppwpnts_list[i].x_m
            marker.pose.position.y = oppwpnts_list[i].y_m
            marker.pose.position.z = marker_height/2
            marker.pose.orientation.w = 1.0
            marker.scale.x = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.y = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.z = marker_height
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            opp_traj_marker_array.markers.append(marker)#frenpy 
    

        return opp_traj_marker_array
    
    def save_monitor_data(
        self,
        oppwpnts_list, 
        inducing_points_value_vs,    
        inducing_points_value_d,   
        vs_training_time,
        d_training_time,
        vs_predict_time,
        d_predict_time,
        lap_count,
        method,
        resampled_wpnts_xy_up,
        resampled_wpnts_xy_down,              
    ):
        """
        Saves monitoring data to a specified Excel file, organized by lap count.
        The data includes opponent coordinates, resampled coordinates for VS and D models,
        and both training and prediction times.

        Parameters:
            oppwpnts_list (list): List of opponent waypoint objects.
            inducing_points_value_vs (list): Inducing points values for the VS model.
            inducing_points_value_d (list): Inducing points values for the D model.
            vs_training_time (float): Training time for the VS model (in seconds).
            d_training_time (float): Training time for the D model (in seconds).
            vs_predict_time (float): Prediction time for the VS model (in seconds).
            d_predict_time (float): Prediction time for the D model (in seconds).
            lap_count (int): Current lap count.
        """

        # Process opponent coordinates data
        opp_data = []
        for point in oppwpnts_list:
            x = getattr(point, 'x_m', None)
            y = getattr(point, 'y_m', None)
            s = getattr(point, 's_m', None)
            if x is not None and y is not None:
                opp_data.append([float(x), float(y), float(s)])

        # Prepare timing data only if all relevant lists are non-empty
        timing_data = {}
        if vs_training_time or d_training_time or vs_predict_time or d_predict_time:
            timing_data['Model'] = []
            timing_data['Training Time (s)'] = []
            timing_data['Prediction Time (s)'] = []

            if vs_training_time:
                timing_data['Model'].append('VS Model')
                timing_data['Training Time (s)'].append(vs_training_time)
                timing_data['Prediction Time (s)'].append(vs_predict_time)
            if d_training_time:
                timing_data['Model'].append('D Model')
                timing_data['Training Time (s)'].append(d_training_time)
                timing_data['Prediction Time (s)'].append(d_predict_time)

        if timing_data:
            timing_df = pd.DataFrame(timing_data)
        else:
            timing_df = pd.DataFrame(columns=['Model', 'Training Time (s)', 'Prediction Time (s)'])

        # Define the base output directory using ROS package path
        try:
            sgp_monitor_path = rospkg.RosPack().get_path('sgp_monitor')
            file_dir = os.path.join(sgp_monitor_path, 'data')
        except rospkg.ResourceNotFound:
            # Fallback to hardcoded path if package not found (for development)
            rospy.logwarn("sgp_monitor package not found, using fallback path")
            file_dir = os.path.abspath('/f110_utils/nodes/sgp_monitor/data')
        
        # Create a subdirectory for the current lap count
        lap_dir = os.path.join(file_dir, f'lap_{lap_count}')
        os.makedirs(lap_dir, exist_ok=True)  # Ensure the directory exists
        
        # Define the full path for the Excel file
        # Define the full path for the Excel file with method in filename
        excel_filename = f'monitor_data_{method}.xlsx'
        excel_path = os.path.join(lap_dir, excel_filename)
        
        # Write data to the Excel file
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write opponent coordinates to 'Opponent Coordinates' sheet if data exists
            if opp_data:
                opp_df = pd.DataFrame(opp_data, columns=['x(m)', 'y(m)', 's(m)'])
                opp_df.to_excel(writer, sheet_name='Opponent Coordinates', index=False)
            
            # Write VS model resampled coordinates to 'VS Resampled Coordinates' sheet if data exists
            if len(inducing_points_value_vs)>0:
                flattened_list_vs = [item[0] % self.track_length for item in inducing_points_value_vs]
                zeros_list = [0] * len(flattened_list_vs)
                resampled_wpnts_xy_vs = self.converter.get_cartesian(flattened_list_vs, zeros_list).transpose()
                vs_df = pd.DataFrame(resampled_wpnts_xy_vs, columns=['x(m)', 'y(m)'])
                vs_df.to_excel(writer, sheet_name='VS Resampled Coordinates', index=False)
            
            # Write D model resampled coordinates to 'D Resampled Coordinates' sheet if data exists
            if len(inducing_points_value_d)>0:
                flattened_list_d = [item[0] % self.track_length for item in inducing_points_value_d]
                zeros_list = [0] * len(flattened_list_d)
                resampled_wpnts_xy_d = self.converter.get_cartesian(flattened_list_d, zeros_list).transpose()
                d_df = pd.DataFrame(resampled_wpnts_xy_d, columns=['x(m)', 'y(m)'])
                d_df.to_excel(writer, sheet_name='D Resampled Coordinates', index=False)

            # Handle UP and DOWN resampled coordinates.
            if resampled_wpnts_xy_up.size > 0:
                x_values = resampled_wpnts_xy_up[0]
                y_values = resampled_wpnts_xy_up[1]
                up_points = list(zip(x_values, y_values))
                up_df = pd.DataFrame(up_points, columns=['x(m)', 'y(m)'])
                up_df.to_excel(writer, sheet_name='UP Resampled Coordinates', index=False)
            
            if resampled_wpnts_xy_down.size > 0:
                x_values = resampled_wpnts_xy_down[0]
                y_values = resampled_wpnts_xy_down[1]
                down_points = list(zip(x_values, y_values))
                down_df = pd.DataFrame(down_points, columns=['x(m)', 'y(m)'])
                down_df.to_excel(writer, sheet_name='DOWN Resampled Coordinates', index=False)
            
            # Write training and prediction times to 'Training and Prediction Summary' sheet if data exists
            if not timing_df.empty:
                timing_df.to_excel(writer, sheet_name='Training and Prediction Summary', index=False)
        
        # Print confirmation message
        print(f"Lap Count: {lap_count}")
        print(f"Monitoring data saved successfully to: {excel_path}")


if __name__ == '__main__':


    node = GaussianProcessOppTraj()
    node.get_gp_opp_traj()
    rospy.spin()
