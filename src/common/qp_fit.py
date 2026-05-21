import numpy as np
import osqp
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

class QPFit:
    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.t_data = None
        self.v_start_x = None
        self.v_start_y = None
        self.v_end_x = None
        self.v_end_y = None
        self.coeffs_opt = None

    # Method to set external data
    def set_data(self, x_data, y_data, t_data, v_start_x, v_start_y, v_end_x, v_end_y):
        self.x_data = x_data
        self.y_data = y_data
        self.t_data = t_data
        self.v_start_x = v_start_x
        self.v_start_y = v_start_y
        self.v_end_x = v_end_x
        self.v_end_y = v_end_y

    # Quintic polynomial fitting function
    def poly(self, t, coeffs):
        return coeffs[5] + coeffs[4]*t + coeffs[3]*t**2 + coeffs[2]*t**3 + coeffs[1]*t**4 + coeffs[0]*t**5

    # Derivative function (velocity)
    def poly_prime(self, t, coeffs):
        return coeffs[5] + 2*coeffs[4]*t + 3*coeffs[3]*t**2 + 4*coeffs[2]*t**3 + 5*coeffs[1]*t**4

    # Design matrix
    def design_matrix(self, t):
        return np.vstack([t**i for i in range(6)]).T

    # Construct the quadratic programming matrices
    def build_qp_matrices(self):
        # Start and end times
        t_start = self.t_data[0]
        t_end = self.t_data[-1]

        # Design matrices
        A_x = self.design_matrix(self.t_data)
        A_y = self.design_matrix(self.t_data)

        # Objective matrices Q (symmetric)
        Q_x = 2 * A_x.T @ A_x
        Q_y = 2 * A_y.T @ A_y
        Q = sp.bmat([[Q_x, None], [None, Q_y]])  # Combine two objective matrices

        # Linear terms c
        c_x = -2 * A_x.T @ self.x_data
        c_y = -2 * A_y.T @ self.y_data
        c = np.concatenate([c_x, c_y])  # Combine both linear terms

        # Equality constraint matrix A_eq (equality constraints)
        A_eq = np.zeros((8, 12))  # 8 constraints (start position, end position, start velocity, end velocity)

        # Start position
        A_eq[0, 0:6] = [1, 0, 0, 0, 0, 0]  # x(t_start) = x_data[0]
        A_eq[1, 6:12] = [1, 0, 0, 0, 0, 0]  # y(t_start) = y_data[0]

        # End position
        A_eq[2, 0:6] = [1, t_end, t_end**2, t_end**3, t_end**4, t_end**5]  # x(t_end) = x_data[-1]
        A_eq[3, 6:12] = [1, t_end, t_end**2, t_end**3, t_end**4, t_end**5]  # y(t_end) = y_data[-1]

        # Start velocity
        A_eq[4, 1] = 1
        A_eq[4, 2] = 2 * t_start
        A_eq[4, 3] = 3 * t_start**2
        A_eq[4, 4] = 4 * t_start**3
        A_eq[4, 5] = 5 * t_start**4  # x'(t_start) = v_start_x

        A_eq[5, 7] = 1
        A_eq[5, 8] = 2 * t_start
        A_eq[5, 9] = 3 * t_start**2
        A_eq[5, 10] = 4 * t_start**3
        A_eq[5, 11] = 5 * t_start**4  # y'(t_start) = v_start_y

        # End velocity
        A_eq[6, 1] = 1
        A_eq[6, 2] = 2 * t_end
        A_eq[6, 3] = 3 * t_end**2
        A_eq[6, 4] = 4 * t_end**3
        A_eq[6, 5] = 5 * t_end**4  # x'(t_end) = v_end_x

        A_eq[7, 7] = 1
        A_eq[7, 8] = 2 * t_end
        A_eq[7, 9] = 3 * t_end**2
        A_eq[7, 10] = 4 * t_end**3
        A_eq[7, 11] = 5 * t_end**4  # y'(t_end) = v_end_y

        # Equality constraint values
        b_eq = np.array([self.x_data[0], self.y_data[0], self.x_data[-1], self.y_data[-1],
                         self.v_start_x, self.v_start_y, self.v_end_x, self.v_end_y])

        return Q, c, A_eq, b_eq

    # Solve the quadratic programming problem using OSQP
    def solve_fit_qp(self):
        # Build the quadratic programming problem
        Q, c, A_eq, b_eq = self.build_qp_matrices()

        # Convert the data to sparse matrix format
        Q = sp.csc_matrix(Q)  # Convert Q to sparse matrix
        A_eq = sp.csc_matrix(A_eq)  # Convert A_eq to sparse matrix

        # Set up the OSQP solver
        qp = osqp.OSQP()
        qp.setup(P=Q, q=c, A=A_eq, l=b_eq, u=b_eq, verbose=False)

        # Solve the quadratic programming problem
        sol = qp.solve()

        # Extract the results
        self.coeffs_opt = sol.x

        coeff_x = self.coeffs_opt[:6]  # x(t)
        coeff_y = self.coeffs_opt[6:]  # y(t)
        coeff_x = coeff_x[::-1]
        coeff_y = coeff_y[::-1]

        return coeff_x, coeff_y

    # Get the coefficients of the fitted polynomial
    def get_coefficients(self):
        if self.coeffs_opt is None:
            raise ValueError("Please call solve_qp() first to solve the problem.")
        return self.coeffs_opt

    # Visualize the fitting results
    def visualize_results(self):
        coeff_x = self.coeffs_opt[:6]  # x(t)
        coeff_y = self.coeffs_opt[6:]  # y(t)
        coeff_x = coeff_x[::-1]
        coeff_y = coeff_y[::-1]
        # Calculate the fitting results
        x_fit = self.poly(self.t_data, coeff_x)  # x(t)
        y_fit = self.poly(self.t_data, coeff_y)  # y(t)

        # Plot the fitting results
        plt.plot(self.t_data, self.x_data, 'bo', label='x_data vs t')
        plt.plot(self.t_data, x_fit, 'r-', label='Fitted x(t)')
        plt.xlabel('Time (t)')
        plt.ylabel('x')
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(self.t_data, self.y_data, 'bo', label='y_data vs t')
        plt.plot(self.t_data, y_fit, 'r-', label='Fitted y(t)')
        plt.xlabel('Time (t)')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(self.x_data, self.y_data, 'bo', label='y_data vs x_data')
        plt.plot(self.t_data, y_fit, 'r-', label='Fitted y(t)')
        plt.xlabel('Time (t)')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        plt.show()

# Test case
if __name__ == "__main__":
    # Initialize QPFit class
    qp = QPFit()
    
    x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_data = np.array([0, 2, 5, 7, 10, 13, 15, 16, 17, 18, 18.5])
    t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    v_start_x = 1
    v_start_y = 1
    v_end_x = 1
    v_end_y = 1
    # Set the data
    qp.set_data(x_data, y_data, t_data, v_start_x, v_start_y, v_end_x, v_end_y)

    # Call the solve_qp method
    poly_x, poly_y = qp.solve_fit_qp()

    # Output the fitting coefficients
    print("coeff x: {poly_x}, coeff y:{poly_y}")

    # Visualize the results
    qp.visualize_results()
