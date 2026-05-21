import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from frenet_converter.frenet_converter import FrenetConverter
import copy
import time
class PolynomialPath:
    def __init__(self, L, dt, evasion_x, evasion_y, evasion_v, max_s_updated, converter, initial_ref_spline):
        """
        Initialize polynomial path
        :param L: Wheel base
        """
        self.L = L
        self.converter = converter
        self.P = None
        self.evasion_x = np.array(evasion_x)
        self.evasion_y = np.array(evasion_y) 
        self.evasion_v = np.array(evasion_v)
        self.max_s = max_s_updated
        self.x_coeff = None
        self.y_coeff = None
        self.start_vx = None
        self.start_vy = None
        self.end_vx = None
        self.end_vy = None
        self.initial_ref_spline = initial_ref_spline
        self.dt = dt
        self.construct_P()

    def calc_first_derivative(self, a, t):
        return 5 * a[0] * t**4 + 4 * a[1] * t**3 + 3 * a[2] * t**2 + 2 * a[3] * t + a[4]

    def calc_second_derivative(self, a, t):
        return 20 * a[0] * t**3 + 12 * a[1] * t**2 + 6 * a[2] * t + 2 * a[3]

    def calc_third_derivative(self, a, t):
        return 60 * a[0] * t**2 + 24 * a[1] * t + 6 * a[2]

    def calculate_position_batch(self, t_batch, a):
        """
        Calculate positions for multiple time points t_batch at once using vectorized operation.
        :param t_batch: Array of time values [t1, t2, ..., tn]
        :param a: Polynomial coefficients [a0, a1, a2, a3, a4, a5]
        :return: Array of positions [x1, x2, ..., xn]
        """
        t_batch = np.array(t_batch)
        return a[0]*t_batch**5 + a[1]*t_batch**4 + a[2]*t_batch**3 + a[3]*t_batch**2 + a[4]*t_batch + a[5]

    def calculate_orientation_batch(self, t_batch, x_coeffs=None, y_coeffs=None):
        """
        Calculate orientations for multiple time points t_batch using vectorized operation.
        :param t_batch: Array of time values [t1, t2, ..., tn]
        :param x_coeffs: Polynomial coefficients for x(t)
        :param y_coeffs: Polynomial coefficients for y(t)
        :return: Array of orientations [phi1, phi2, ..., phin]
        """
        x_coeffs = self.x_coeff if x_coeffs is None else x_coeffs
        y_coeffs = self.y_coeff if y_coeffs is None else y_coeffs
        
        dx_batch = self.calc_first_derivative(x_coeffs, t_batch)
        dy_batch = self.calc_first_derivative(y_coeffs, t_batch)
        return np.arctan2(dy_batch, dx_batch)

    def calculate_speed_batch(self, t_batch, x_coeffs=None, y_coeffs=None):
        """
        Calculate speeds for multiple time points t_batch using vectorized operation.
        :param t_batch: Array of time values [t1, t2, ..., tn]
        :param x_coeffs: Polynomial coefficients for x(t)
        :param y_coeffs: Polynomial coefficients for y(t)
        :return: Array of speeds [v1, v2, ..., vn]
        """
        x_coeffs = self.x_coeff if x_coeffs is None else x_coeffs
        y_coeffs = self.y_coeff if y_coeffs is None else y_coeffs
        
        dx_batch = self.calc_first_derivative(x_coeffs, t_batch)
        dy_batch = self.calc_first_derivative(y_coeffs, t_batch)
        return np.sqrt(dx_batch**2 + dy_batch**2)

    def calculate_acceleration_batch(self, t_batch, x_coeffs=None, y_coeffs=None):
        """
        Calculate accelerations for multiple time points t_batch using vectorized operation.
        :param t_batch: Array of time values [t1, t2, ..., tn]
        :param x_coeffs: Polynomial coefficients for x(t)
        :param y_coeffs: Polynomial coefficients for y(t)
        :return: Array of accelerations [a1, a2, ..., an]
        """
        x_coeffs = self.x_coeff if x_coeffs is None else x_coeffs
        y_coeffs = self.y_coeff if y_coeffs is None else y_coeffs
        
        dx_batch = self.calc_first_derivative(x_coeffs, t_batch)
        dy_batch = self.calc_first_derivative(y_coeffs, t_batch)
        ddx_batch = self.calc_second_derivative(x_coeffs, t_batch)
        ddy_batch = self.calc_second_derivative(y_coeffs, t_batch)

        speed_batch = self.calculate_speed_batch(t_batch, x_coeffs, y_coeffs)
        speed_squared_batch = speed_batch**2

        return (dx_batch * ddx_batch + dy_batch * ddy_batch) / np.sqrt(dx_batch**2 + dy_batch**2)

    def calculate_wheel_angle_batch(self, t_batch, x_coeffs=None, y_coeffs=None):
        """
        Calculate wheel angles for multiple time points t_batch using vectorized operation.
        :param t_batch: Array of time values [t1, t2, ..., tn]
        :param x_coeffs: Polynomial coefficients for x(t)
        :param y_coeffs: Polynomial coefficients for y(t)
        :return: Array of wheel angles [theta1, theta2, ..., thetan]
        """
        x_coeffs = self.x_coeff if x_coeffs is None else x_coeffs
        y_coeffs = self.y_coeff if y_coeffs is None else y_coeffs
        
        dx_batch = self.calc_first_derivative(x_coeffs, t_batch)
        dy_batch = self.calc_first_derivative(y_coeffs, t_batch)
        ddx_batch = self.calc_second_derivative(x_coeffs, t_batch)
        ddy_batch = self.calc_second_derivative(y_coeffs, t_batch)

        speed_batch = self.calculate_speed_batch(t_batch, x_coeffs, y_coeffs)
        speed_squared_batch = speed_batch**2

        num = (dx_batch * ddy_batch - dy_batch * ddx_batch) * self.L
        den = np.sqrt(speed_squared_batch**3)
        
        # Avoid division by zero
        wheel_angles = np.arctan(num / den)
        wheel_angles[speed_squared_batch == 0] = 0
        return wheel_angles

    def calculate_at_t_batch(self, t_batch, x_coeffs=None, y_coeffs=None):
        """
        Calculate all states and inputs for a batch of time points t_batch.
        :param t_batch: Array of time values [t1, t2, ..., tn]
        :param x_coeffs: Polynomial coefficients for x(t)
        :param y_coeffs: Polynomial coefficients for y(t)
        :return: Tuple of (states, inputs) where states and inputs are 2D arrays.
        """
        x_coeffs = self.x_coeff if x_coeffs is None else x_coeffs
        y_coeffs = self.y_coeff if y_coeffs is None else y_coeffs

        x_batch = self.calculate_position_batch(t_batch, x_coeffs)
        y_batch = self.calculate_position_batch(t_batch, y_coeffs)
        orientation_batch = self.calculate_orientation_batch(t_batch, x_coeffs, y_coeffs)
        speed_batch = self.calculate_speed_batch(t_batch, x_coeffs, y_coeffs)
        acceleration_batch = self.calculate_acceleration_batch(t_batch, x_coeffs, y_coeffs)
        wheel_angle_batch = self.calculate_wheel_angle_batch(t_batch, x_coeffs, y_coeffs)

        # Stack the results into 2D arrays (each row is a time step, each column a state/input)
        states_batch = np.column_stack((x_batch, y_batch, speed_batch, orientation_batch))
        inputs_batch = np.column_stack((acceleration_batch, wheel_angle_batch))

        return states_batch, inputs_batch

    def construct_P(self):
        # Validate input dimensions
        if len(self.evasion_x) == 0 or len(self.evasion_y) == 0 or len(self.evasion_v) == 0:
            raise ValueError("Input arrays evasion_x, evasion_y, and evasion_v must not be empty.")
        
        if len(self.evasion_x) != len(self.evasion_y) or len(self.evasion_x) != len(self.evasion_v):
            raise ValueError("Input arrays must have the same length.")

        # Initialize time array
        t = np.zeros_like(self.evasion_x, dtype=float)

        # Compute time for each point
        for i in range(1, len(self.evasion_x)):
            distance = np.sqrt((self.evasion_x[i] - self.evasion_x[i-1])**2 + (self.evasion_y[i] - self.evasion_y[i-1])**2)
            if self.evasion_v[i-1] <= 1e-6:  # Use a small threshold instead of checking for exact zero
                raise ValueError(f"Zero or near-zero velocity detected at index {i-1}. Cannot compute time.")
            t[i] = t[i-1] + distance / np.fabs(self.evasion_v[i-1])  # Incremental time calculation

        # Construct self.P
        self.P = np.column_stack((self.evasion_x, self.evasion_y, t))

    # def fit_traj(self):
    #     '''
    #     Fitting curves using optimization
    #     '''
    #     x = self.P[:, 0]
    #     y = self.P[:, 1]
    #     t = self.P[:, 2]

    #     # Generate Vandermonde matrix for polynomial fitting
    #     X = np.vander(t, 6)

    #     # Quadratic term for the objective function (only fitting to P)
    #     H1 = np.dot(X.T, X)  # Quadratic term for fitting

    #     # Ensure symmetry of H1
    #     H1 = (H1 + H1.T) / 2

    #     # Constraint matrices
    #     A_eq_x = np.vstack([
    #         X[0],  # Start position constraint
    #         X[-1],  # End position constraint
    #         np.array([5 * t[0]**4, 4 * t[0]**3, 3 * t[0]**2, 2 * t[0], 1, 0]),  # Start velocity constraint
    #         np.array([5 * t[-1]**4, 4 * t[-1]**3, 3 * t[-1]**2, 2 * t[-1], 1, 0])  # End velocity constraint
    #     ])
    #     A_eq_y = A_eq_x.copy()
    #     b_eq_x = np.array([x[0], x[-1], self.start_vx, self.end_vx])
    #     b_eq_y = np.array([y[0], y[-1], self.start_vy, self.end_vy])

    #     # Convert to cvxopt matrix format
    #     A_eq_x = cvxopt.matrix(A_eq_x, tc='d')
    #     A_eq_y = cvxopt.matrix(A_eq_y, tc='d')
    #     b_eq_x = cvxopt.matrix(b_eq_x, (b_eq_x.size, 1), tc='d')
    #     b_eq_y = cvxopt.matrix(b_eq_y, (b_eq_y.size, 1), tc='d')

    #     # Objective function (fitting to P)
    #     H_x = self.a1 * H1  # Only fitting term remains
    #     H_y = self.a1 * H1

    #     # Linear term
    #     f_x1 = -self.a1 * np.dot(X.T, x)  # Linear term for x
    #     f_y1 = -self.a1 * np.dot(X.T, y)
    #     f_x = cvxopt.matrix(f_x1, (f_x1.size, 1), tc='d')
    #     f_y = cvxopt.matrix(f_y1, (f_y1.size, 1), tc='d')

    #     # Solve the quadratic programming problem using cvxopt
    #     solution_x = cvxopt.solvers.qp(cvxopt.matrix(H_x, tc='d'), f_x, A=A_eq_x, b=b_eq_x)
    #     self.coeff_x = np.array(solution_x['x']).flatten()

    #     solution_y = cvxopt.solvers.qp(cvxopt.matrix(H_y, tc='d'), f_y, A=A_eq_y, b=b_eq_y)
    #     self.coeff_y = np.array(solution_y['x']).flatten()

    #     self.t_s = t.min()
    #     self.t_e = t.max()
    
    def fit_quintic_polynomial(self):
        """
        Fit quintic polynomials for given coordinates and corresponding times.
        """
        # Ensure inputs are numpy arrays
        
        x = self.P[:, 0]
        y = self.P[:, 1]
        t = self.P[:, 2]
        
        
        # Fit a quintic polynomial for x(t)
        self.x_coeff = np.polyfit(t, x, 5)  # 5th-degree polynomial
        
        # Fit a quintic polynomial for y(t)
        self.y_coeff = np.polyfit(t, y, 5)  # 5th-degree polynomial

    def set_coeffs(self, poly_x, poly_y):
        self.x_coeff = poly_x[:]
        self.y_coeff = poly_y[:]

    def handle_circular_track(self, raw_evasion_s):
        evasion_s = copy.deepcopy(raw_evasion_s)
        for i in range(len(evasion_s) - 1):
            if evasion_s[i] > evasion_s[i + 1]:
                evasion_s[i + 1:] = [x + self.max_s for x in evasion_s[i + 1:]]
                return evasion_s
        return evasion_s
    
    def get_flatness_params(self, evasion_s, interp_nums):
        """
        Vectorized version of get_flatness_params: Computes states and inputs for multiple sampled times.

        :param evasion_s: Longitudinal distances (self.P[:, 0] corresponds to the points in evasion_s)
        :param interp_nums: Number of evenly spaced points to sample
        :return: states_list (Nx4), inputs_list (Nx2)
        """
        # Ensure evasion_s and self.P[:, 2] have the same length
        if len(evasion_s) != len(self.P[:, 2]):
            raise ValueError("evasion_s and time must have the same length.")
        
        # Extract the times corresponding to evasion_s
        t = self.P[:, 2]

        # # Handle circular track issue
        # handled_evasion_s = self.handle_circular_track(evasion_s)

        # # Sampled s values (evenly spaced)
        # sampled_s = np.linspace(handled_evasion_s[0], handled_evasion_s[-1], interp_nums)

        # # Interpolate times based on evasion_s and t
        # sampled_t = np.interp(sampled_s, handled_evasion_s, t)
        sampled_t = np.linspace(t[0], t[-1], interp_nums)

        # **Batch calculation using the optimized function**
        states_list, inputs_list = self.calculate_at_t_batch(sampled_t)

        return states_list, inputs_list

    def converter_flatness_tofrenet(self, states_list, inputs_list):
        # Use approximate s from path distance to speed up get_frenet
        x_arr = states_list[:, 0]
        y_arr = states_list[:, 1]
        
        # Compute approximate s from cumulative distance (faster than get_approx_s)
        dx = np.diff(x_arr, prepend=x_arr[0])
        dy = np.diff(y_arr, prepend=y_arr[0])
        ds = np.sqrt(dx**2 + dy**2)
        approx_s = np.cumsum(ds)
        # Offset by first point's approximate s
        first_s = self.converter.get_approx_s(x_arr[:1], y_arr[:1])[0]
        approx_s = (approx_s + first_s - approx_s[0]) % self.converter.raceline_length
        
        resp = self.converter.get_frenet(x_arr, y_arr, approx_s)
        s, d = resp[0, :].flatten(), resp[1, :].flatten()
        
        # Vectorized delta_phi computation
        delta_phi = self.initial_ref_spline.get_delta_phi_batch(s, states_list[:, 3])

        # Frenet states: [s, d, v(Cartesian), delta_phi], inputs: [acc(Cartesian), wheel_angle]
        frenet_states_list = np.column_stack((s, d, states_list[:, 2], delta_phi))
        
        return frenet_states_list, inputs_list
        
    def get_df_frenet(self, evasion_s, evasion_d, poly_x, poly_y):
        start_time_fit = time.time()
        self.set_coeffs(poly_x, poly_y)
        end_time_fit = time.time()
        fit_time = end_time_fit - start_time_fit

        # interp_nums = len(evasion_s)
        # interp_nums = int((self.P[:, 2][-1] - self.P[:,2][0]) / self.dt) + 1
        interp_nums = 40
        print(f'interp_nums:{interp_nums}')
        start_cest_flatness = time.time()
        states_list, inputs_list = self.get_flatness_params(evasion_s, interp_nums)
        end_cest_flatness = time.time()
        cest_df_time = end_cest_flatness -start_cest_flatness
        start_frenet_flatness = time.time()
        frenet_states_list, frenet_inputs_list = self.converter_flatness_tofrenet(states_list, inputs_list)
        # print(f'states_list:{states_list}, inputs_list:{inputs_list}')
        end_frenet_flatness = time.time()
        frenet_df_time = end_frenet_flatness - start_frenet_flatness

        total_time = fit_time + cest_df_time + frenet_df_time

        # consume time
        print(f"fit_time: {fit_time*1000:.3f}ms")
        print(f"cest_flatness_time: {cest_df_time*1000:.3f}ms")
        print(f"frenet_flatness_time: {frenet_df_time*1000:.3f}ms")
        print(f"total_time: {total_time*1000:.3f}ms")
        # # for debug visualization
        # self.visualize_trajectories(states_list)
        # self.vis_sd(evasion_s, evasion_d, frenet_states_list[:,0], frenet_states_list[:,1])
        return frenet_states_list, frenet_inputs_list 

    def visualize_trajectories(self, state_list):
        # Extract x and y from state_list
        traj2_x = state_list[:,0]
        traj2_y = state_list[:,1]

        save_dir = '~/eth_race/fig'
        save_dir = os.path.expanduser(save_dir)
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(save_dir, f"trajectory_{timestamp}.png")

        # Initialize the plot
        plt.figure()
        plt.title("Trajectories Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        # Plot traj1 (evasion_x, evasion_y)
        plt.plot(self.evasion_x, self.evasion_y, label="Trajectory 1 (evasion)", color="blue", linestyle="--")

        # Plot traj2 (state_list x and y)
        plt.plot(traj2_x, traj2_y, label="Trajectory 2 (state_list)", color="red", linestyle="-")

        # Add legend
        plt.legend()

        # Save the plot
        plt.savefig(file_path)
        plt.close()  # Close the figure to free memory

    def vis_sd(self, traj1_s, traj1_d, traj2_s, traj2_d, save_dir='~/eth_race/fig'):
        save_dir = os.path.expanduser(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(8, 6))
        
        plt.plot(traj1_s, traj1_d, label='evasion_sd', color='b')
        plt.plot(traj2_s, traj2_d, label='state_list_sd', color='r')

        plt.xlabel('s (Longitudinal coordinate)')
        plt.ylabel('d (Lateral coordinate)')
        
        plt.title('sd Comparison')
        
        plt.legend()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(save_dir, f"sd_comparison_{timestamp}.png")
        plt.savefig(save_path)

        plt.close()
        

        

        


    

