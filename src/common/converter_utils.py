from typing import List, Union, Tuple
import numpy as np
from numpy import arctan2
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import rospy
from frenet_converter.frenet_converter import FrenetConverter
from f110_msgs.msg import WpntArray

class InitialRefSpline():
    """
    A class that defines a initial spline, with functions that help.
    """
    def __init__(self) -> None:
        """
        Initialise Reference Spline object.

        Input:  conf_file   : String containing the path to the param_config.yaml file
        """
        # Init the solver
        self.initialize_spline()

    def initialize_spline(self) -> None:
        """Initialises the Reference Spline. Global waypoints are stored in a SplineTrack. All necessary parameters are stored. """

        rospy.loginfo(f"[Reference Spline] Waiting for global waypoints")
        raceline = rospy.wait_for_message("/global_waypoints", WpntArray)
        rospy.loginfo(f"[Reference Spline] Global waypoints obtained")

        x, y = self._transform_waypoints_to_cartesian(raceline.wpnts)
        self.fren_conv = FrenetConverter(x, y)

        # on f track trajectory is 81.803 m long.
        d_left, coords_path, d_right = self._transform_waypoints_to_coords(raceline.wpnts)

        self.spline = SplineTrack(coords_direct=coords_path)

        self.nr_laps = 0

        kapparef = [x.kappa_radpm for x in raceline.wpnts]
        vx_ref = [x.vx_mps for x in raceline.wpnts]
        self.s_ref = np.array([x.s_m for x in raceline.wpnts])

        self.kappa = kapparef

    def _transform_waypoints_to_coords(self, data: WpntArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper function to store the received waypoints into the right format such that they can be used for initialisation of SplineTrack.

        Input: data: WpntArray: Holds the data received from the globalwaypoints.

        Returns: Tuple: Stores the waypoints as follows: Left boundaries, reference path, right boundaries
        """
        waypoints = np.zeros((len(data), 2))
        d_left = np.zeros(len(data))
        d_right = np.zeros(len(data))
        boundaries = np.zeros((len(data), 2))
        for idx, wpnt in enumerate(data):
            waypoints[idx] = [wpnt.x_m, wpnt.y_m]
            d_left[idx] = wpnt.d_left              # Fix for boundaries due to frenet cartesian conversion
            d_right[idx] = wpnt.d_right              # Fix for boundaries due to frenet cartesian conversion
        res_coords = np.array([boundaries[:-1], waypoints[:-1], boundaries[:-1]])
        return d_left, res_coords, d_right

    def _transform_waypoints_to_cartesian(self, data: WpntArray) -> Tuple[np.ndarray, np.ndarray]:
        """Helper function to store the received waypoints into the right format such that they can be used for initialisation of SplineTrack.

        Input: data: WpntArray: Holds the data received from the globalwaypoints.

        Returns: Tuple: Stores the waypoints as follows: Left boundaries, reference path, right boundaries
        """
        x = np.zeros(len(data))
        y = np.zeros(len(data))
        for idx, wpnt in enumerate(data):
            x[idx] = wpnt.x_m
            y[idx] = wpnt.y_m
        return np.array(x), np.array(y)
    
    def get_delta_phi(self, center_car_s, cartesian_phi):
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = cartesian_phi - np.arctan2(deriv_center[1], deriv_center[0])
        # make aplha_center between -pi and pi
        alpha_center = alpha_center % (2 * np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2 * np.pi
        return alpha_center

    def get_delta_phi_batch(self, center_car_s, cartesian_phi):
        """
        Vectorized version of get_delta_phi for arrays. Same result as calling
        get_delta_phi(si, phi) for each (si, phi) in zip(center_car_s, cartesian_phi).
        """
        center_car_s = np.asarray(center_car_s, dtype=float)
        cartesian_phi = np.asarray(cartesian_phi, dtype=float)
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = cartesian_phi - np.arctan2(deriv_center[1], deriv_center[0])
        alpha_center = alpha_center % (2 * np.pi)
        alpha_center = np.where(alpha_center > np.pi, alpha_center - 2 * np.pi, alpha_center)
        return alpha_center

    def get_phi(self, center_car_s):
        deriv_center = self.spline.get_derivative(center_car_s)
        phi = np.arctan2(deriv_center[1], deriv_center[0])
        return phi

class EnhancedSpline():
    """
    A class that defines a spline, with functions that help.
    """

    def __init__(self, coords, params, track_length, lame = True) -> None:
        """
        Args:
            coords: coordinate that define the spline
            params: parameters that correspon to the coordinates
            track_length: total length
            safety_margin: the track will be shrinked by this amount on both
            sides so that the car will be at a more safe distance from the
            borders

        """
        self.track_length = track_length

        k = 2
        if lame:
            self.line_x = Spline(
                params,
                np.concatenate((coords[:, 0], np.array([coords[0, 0]]))),
                k = k,
                )
            self.line_y = Spline(params,
                np.concatenate((coords[:, 1], np.array([coords[0, 1]]))),
                k = k,
            )
        else:
            self.line_x = Spline(
                params,
                coords[:,0],
                k = k,
                )
            self.line_y = Spline(
                params,
                coords[:,1],
                k = k,
            )

    def get_coordinate(self, theta) -> np.array:
        """
        Returns the coordinate of the point corresponding to theta on the
        chosen line.

        Args:
            theta: parameter which is used to evaluate the spline
        Returns:
            coord: 2-d coordinate
        """
        theta = theta%self.track_length

        coord = np.array([self.line_x(theta), self.line_y(theta)])

        return coord

    def find_theta(self, coord: np.ndarray, theta_est, eps: float = 0.01):
        """
        Find the parameter of the track nearest to the point given an
        approximation of it.

        Args:
            coord: coordinate of point projected around, as np.array
            track: splinified track
            theta_est: best guess at theta
            eps: precision at which the algorithm looks around for theta
        """

        found = False
        max_dist = self.track_length/20

        dist = np.linalg.norm(
            coord - np.array(self.get_coordinate(theta_est)).reshape(2)
            )
        min_dist = dist
        min_theta = theta_est
        theta_i = theta_est
        while not found:
            theta_i += eps
            dist = np.linalg.norm(
                coord - np.array(self.get_coordinate(theta_i)).reshape(2)
                )

            if dist <= min_dist:
                min_dist = dist
                min_theta = theta_i
            else:
                found = True

            if theta_i - theta_est >= max_dist:
                found = True

        # now try to look backwards
        found = False
        theta_i = theta_est
        while not found:
            theta_i -= eps

            dist = np.linalg.norm(
                coord - np.array(self.get_coordinate(theta_i)).reshape(2)
                )

            if dist <= min_dist:
                min_dist = dist
                min_theta = theta_i
            else:
                found = True

            if theta_est - theta_i >= max_dist:
                found = True

        return min_theta

    def get_angle(self, theta) -> float:
        """
        Returns the angle wrt x axis tangent to the line given the
        parameter theta
        """
        theta = theta%self.track_length
        delt_y = self.line_y(theta, 1)
        delt_x = self.line_x(theta, 1)

        angle = arctan2(delt_y, delt_x)
        if angle<0:
            angle += 2*np.pi

        return angle

    def get_derivative(self, theta):
        """
        Returns the derivative of the point corresponding to theta.
        Supports both scalar and array inputs.

        Args:
            theta: parameter which is used to evaluate the spline (scalar or array)

        Returns:
            der: [dx/dtheta, dy/dtheta] as list or [2, N] array for batch input
        """
        theta = np.asarray(theta) % self.track_length
        
        der = [self.line_x(theta, 1), self.line_y(theta, 1)]

        return der

    def get_curvature(self, theta):
        """
        Returns the derivative of the point corresponding to theta.

        Args:
            theta: parameter which is used to evaluate the spline

        Returns:
            der: dx/dtheta, dy/dtheta
        """
        theta = theta%self.track_length

        x_d = self.line_x(theta, 1)
        x_dd = self.line_x(theta, 2)
        y_d = self.line_y(theta, 1)
        y_dd = self.line_y(theta, 2)

        curvature = (
            np.abs(x_d * y_dd - y_d * x_dd)/pow((x_d**2 + y_d **2), 1.5)
        )

        return curvature

    def get_dphi_dtheta(self, theta, line='mid'):
        """
        Returns the derivative of the angle corresponding to theta on the
        chosen line.

        Args:
            theta: parameter which is used to evaluate the spline
            line: argument used to choose the line. Can be 'int', 'mid',
            'out'. Default is 'mid'.

        Returns:
            der: dx/dtheta, dy/dtheta
        """
        theta = theta%self.track_length


        x_d = self.line_x(theta, 1)
        x_dd = self.line_x(theta, 2)
        y_d = self.line_y(theta, 1)
        y_dd = self.line_y(theta, 2)

        derivative = (x_d * y_dd - y_d * x_dd)/(x_d**2 + y_d **2)

        return derivative

    def find_theta_slow(self, coord: np.ndarray, eps: float = 0.1):
        """
        Find the parameter of the line nearest to the point without an
        approximation of it.
        Could be slower than find_theta as it looks along the whole line.

        Args:
            coord: coordinate of point projected around, as np.array
            track: splinified track
            eps: precision at which the algorithm looks around for theta
        """

        found = False
        dist = np.linalg.norm(coord - np.array(self.get_coordinate(0)))

        min_dist = dist
        min_theta = 0
        theta_i = 0

        while theta_i <= self.track_length:
            theta_i += eps
            dist = np.linalg.norm(
                coord - np.array(self.get_coordinate(theta_i))
                )

            if dist <= min_dist:
                min_dist = dist
                min_theta = theta_i


        return min_theta

    def is_coord_behind(self, coord: np.ndarray, theta: float):
        """
        Checks if the given coordinate is behind the given theta.
        Forward is defined as the track direction and a line is
        centered in the point corresponding to theta to divide
        "in front of" and "behind".
        The line is perpendicular to the direction of the line.

        Args:
            coord: coordinate to be checked
            theta: parameter of the spline for reference

        Returns:
            <bool>: true if the cordinate is behind, false if the
                coordinate is in front of the line
        """

        angle = self.get_angle(theta)
        norm_vec = np.array([np.cos(angle), np.sin(angle)])
        coord_on_boundary = self.get_coordinate(theta)
        coord_forward = self.get_coordinate(theta + 0.01)
        q_forward = np.dot(coord_forward, norm_vec)
        q_bound = np.dot(coord_on_boundary, norm_vec)
        q_coord = np.dot(coord, norm_vec)

        if (q_coord-q_bound)*(q_forward-q_bound) < 0:
            return True
        else:
            return False

class SplineTrack():
    """
    a class to represent the splinification of the track, with the
    center line and the two sides of the track
    """

    def __init__(
        self,
        path_to_file = None,
        coords_direct = None,
        path_to_traj = None,
        safety_margin: float= 0.0,
        ) -> None:
        """
        Args:
            path_to_file: path to the file where the coordinates of the
                track are saved
            coords_direct: array of shape ()
            path_to_traj: path to the file where the coordinates of the
                trajectory are saved
            safety_margin: the track will be shrinked by this amount on
                both sides so that the car will be at a more safe distance
                from the borders

        """

        # 1 Obtain points on the track #
        #################################
        if path_to_file is not None:
            coords = read_coords_from_file(path_to_file)
            coords = np.array(coords)
            params, self.track_length = find_params(coords[1])
        elif coords_direct is not None:
            coords = coords_direct
            params, self.track_length = find_params(coords[1])
        else:
            raise NotImplementedError("neither coordinates or the path to the coordinates was given")

        # 2 shrink the track for robustness #
        #####################################
        for i in range(len(params)-1):
            # shrink int line
            direction = coords[1,i,:] - coords[0,i,:]
            coords[0,i,:] = coords[0,i,:] + \
                safety_margin*direction/np.linalg.norm(direction+0.001)

            # shrink out line
            direction = coords[1,i,:] - coords[2,i,:]
            coords[2,i,:] = coords[2,i,:] + \
                safety_margin*direction/np.linalg.norm(direction+0.001)

        # 3 set track #
        ###############
        self.refline = EnhancedSpline(
            coords[1,:,:], params, self.track_length
            )
        self.intline = EnhancedSpline(
            coords[0,:,:], params, self.track_length
            )
        self.outline = EnhancedSpline(
            coords[2,:,:], params, self.track_length
            )

        # 4 set trajectory #
        ####################
        if path_to_traj is not None:
            traj_coords = read_sing_coords_from_file(path_to_traj)
            traj_coords = np.array(traj_coords)
            params, self.traj_len = find_params(traj_coords)
            self.trajectory = EnhancedSpline(
                traj_coords,
                params,
                self.track_length,
                )
            self.delta_traj = self.trajectory.find_theta_slow(
                self.refline.get_coordinate(0)
                )
        else:
            self.trajectory = self.refline
            self.traj_len = self.track_length
            self.delta_traj = 0


    def get_angle(self, theta, line = 'mid') -> float:
        """
        Returns the angle wrt x axis tangent to the midline interpolation
        given the parameter theta
        """

        if line == 'mid':
            return self.refline.get_angle(theta)
        elif line == 'int':
            return self.intline.get_angle(theta)
        elif line == 'out':
            return self.outline.get_angle(theta)
        else:
            raise ValueError("line can only be 'int', 'mid', or 'out'.")

    def get_coordinate(self, theta, line = 'mid') -> np.array:
        """
        Returns the coordinate of the point corresponding to theta on the
        chosen line.

        Args:
            theta: parameter which is used to evaluate the spline
            line: argument used to choose the line. Can be 'int', 'mid',
            'out'. Default is 'mid'.

        Returns:
            coord: 2-d coordinate
        """

        if line == 'mid':
            coord = self.refline.get_coordinate(theta)
        elif line == 'int':
            coord = self.intline.get_coordinate(theta)
        elif line == 'out':
            coord = self.outline.get_coordinate(theta)
        else:
            raise ValueError("line can only be 'int', 'mid', or 'out'.")

        return coord

    def get_derivative(self, theta, line = 'mid'):
        """
        Returns the derivative of the point corresponding to theta on the
        chosen line. Supports both scalar and array inputs.

        Args:
            theta: parameter which is used to evaluate the spline (scalar or array)
            line: argument used to choose the line. Can be 'int', 'mid',
            'out'. Default is 'mid'.

        Returns:
            der: dx/dtheta, dy/dtheta
        """
        theta = np.asarray(theta) % self.track_length

        if line == 'mid':
            der = self.refline.get_derivative(theta)
        elif line == 'int':
            der = self.intline.get_derivative(theta)
        elif line == 'out':
            der = self.outline.get_derivative(theta)
        else:
            raise ValueError("line can only be 'int', 'mid', or 'out'.")

        return der

    def get_curvature(self, theta, line='mid'):
        """
        Returns the derivative of the point corresponding to theta on the
        chosen line.

        Args:
            theta: parameter which is used to evaluate the spline
            line: argument used to choose the line. Can be 'int', 'mid',
            'out'. Default is 'mid'.

        Returns:
            der: dx/dtheta, dy/dtheta
        """
        theta = theta%self.track_length

        if line == 'mid':
            curvature = self.refline.get_curvature(theta)
        elif line == 'int':
            curvature = self.intline.get_curvature(theta)
        elif line == 'out':
            curvature = self.outline.get_curvature(theta)
        else:
            raise ValueError("line can only be 'int', 'mid', or 'out'.")

        return curvature

    def get_dphi_dtheta(self, theta, line='mid'):
        """
        Returns the derivative of the angle corresponding to theta on the
        chosen line.

        Args:
            theta: parameter which is used to evaluate the spline
            line: argument used to choose the line. Can be 'int', 'mid',
            'out'. Default is 'mid'.

        Returns:
            der: dx/dtheta, dy/dtheta
        """
        theta = theta%self.track_length

        if line == 'mid':
            derivative = self.refline.get_dphi_dtheta(theta)
        elif line == 'int':
            derivative = self.intline.get_dphi_dtheta(theta)
        elif line == 'out':
            derivative = self.outline.get_dphi_dtheta(theta)
        else:
            raise ValueError("line can only be 'int', 'mid', or 'out'.")

        return derivative

    def find_theta(self, coord: np.ndarray, theta_est, eps: float = 0.01):
        """
        Find the parameter of the track nearest to the point given an
        approximation of it.

        Args:
            coord: coordinate of point projected around, as np.array
            track: splinified track
            theta_est: best guess at theta
            eps: precision at which the algorithm looks around for theta
        """

        min_theta = self.refline.find_theta(coord, theta_est, eps)

        return min_theta

    def find_theta_slow(self, coord: np.ndarray, eps: float = 0.1):
        """
        Find the parameter of the track nearest to the point without an
        approximation of it.
        Could be slower

        Args:
            coord: coordinate of point projected around, as np.array
            track: splinified track
            eps: precision at which the algorithm looks around for theta
        """

        min_theta = self.refline.find_theta_slow(coord, eps)

        return min_theta

    def is_coord_behind(self, coord: np.ndarray, theta: float):
        return self.refline.is_coord_behind(coord, theta)


def read_coords_from_file(path_to_file: str):
    out_arr = []
    mid_arr = []
    int_arr = []

    with open(path_to_file, 'r') as file:
        coords = file.readlines()

    coords = coords[1:] # discard 1st line

    for el in coords:
        out, mid, inn = el.strip().split('], [')
        out_arr.append([float(el) for el in out.strip('[],').split(',')])
        mid_arr.append([float(el) for el in mid.strip('[],').split(',')])
        int_arr.append([float(el) for el in inn.strip('[],').split(',')])

    return out_arr, mid_arr, int_arr

def read_coords_from_file_new(path_to_file: str):
    """
    Reads coordinates from a file that save a track this way:

        first line:
            length: <number of coordinates>

        second line:
            center_line_x, centerline_y, left_width, right_width

        other lines:
            [numbers int the order specified above]

    Args:
        path_to_file: path to the coordinates' file

    Returns:
        center_line: np.array of shape (<length>, 2)
        left_widths: np.array of shape (<length>,)
        right_widths: np.array of shape (<length>,)
    """

    with open(path_to_file, 'r') as file:
        coords = file.readlines()

    # read file length
    length = int(coords[0].strip(', \n').split(':')[1])
    center_line = np.empty((length, 2))
    left_widths = np.empty((length))
    right_widths = np.empty((length))

    coords = coords[2:] # discard first two lines lines

    for i, el in enumerate(coords):
        center_line_x, center_line_y, left_w, right_w, _ = el.strip().split(',')
        center_line[i, :] = float(center_line_x), float(center_line_y)
        left_widths[i] = float(left_w)
        right_widths[i] = float(right_w)

    return center_line, left_widths, right_widths

def read_sing_coords_from_file(path_to_file: str):

    arr = []

    with open(path_to_file, 'r') as file:
        coords = file.readlines()

    coords = coords[1:] # discard 1st line

    for el in coords:
        out = el.strip().split(',')
        arr.append([float(el) for el in out])

    return arr

def find_params(coords: np.ndarray):
    """
    Finds the parameters to fit the spline(basically x axis for
    interpolation). Should be only used with central line.

    Returns:
        cum_param: an array of the parameters of shape len(coords) + 1
        length: length of the track
    """

    cum_param = [0]

    prev_el = coords[0]
    length = 0

    for el in coords[1:]:
        length += np.linalg.norm((el-prev_el))
        prev_el = el
        cum_param.append(length)

    length += np.linalg.norm(el - coords[0])
    cum_param.append(length)

    return cum_param, length
