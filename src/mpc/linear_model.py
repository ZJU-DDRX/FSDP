import numpy as np
from casadi import MX, Function, vertcat, jacobian, interpolant
import time

class Vehicle_LModel:
    def __init__(self, s0, kapparef, L):
        """Initialize the Vehicle Model with path data and vehicle length.
        
        Args:
            s0 (np.ndarray): Arc length data.
            kapparef (np.ndarray): Curvature data.
            L (float): Vehicle length.
        """
        self.s0 = s0
        self.kapparef = kapparef
        self.L_value = L

        # Extend data to create periodicity
        length = len(s0)
        s0_extended = np.append(s0, s0[-1] + s0[1:length])
        s0_shifted = np.append(s0[:length - 1] - s0[-1], s0_extended)

        kapparef_extended = np.append(kapparef, kapparef[1:length])
        kapparef_shifted = np.append(kapparef[:length - 1] - kapparef[-1], kapparef_extended)

        # Create spline interpolator for kappa_r(s)
        self.kapparef_s = interpolant("kapparef_s", "bspline", [s0_shifted], kapparef_shifted)

        # Define CasADi symbols for states, inputs, and parameters
        self.s = MX.sym('s')
        self.n = MX.sym('n')
        self.delta_phi = MX.sym('delta_phi')
        self.v = MX.sym('v')
        self.delta = MX.sym('delta')
        self.L = L

        # Define dynamics equations
        self.kappa_r = MX.sym('kappa_r')
        self.s_dot = self.v * MX.cos(self.delta_phi) / (1 - self.kappa_r * self.n)
        self.n_dot = self.v * MX.sin(self.delta_phi)
        self.delta_phi_dot = (self.v * MX.tan(self.delta) / self.L) - (self.kappa_r * self.v * MX.cos(self.delta_phi) / (1 - self.kappa_r * self.n))

        # Define state and input vectors
        self.x = vertcat(self.s, self.n, self.delta_phi)
        self.u = vertcat(self.v, self.delta)

        # Combine dynamics into a vector
        self.dynamics = vertcat(self.s_dot, self.n_dot, self.delta_phi_dot)

        # Compute Jacobians
        self.A = jacobian(self.dynamics, self.x)
        self.B = jacobian(self.dynamics, self.u)

        # Create CasADi functions for Jacobians
        self.A_func = Function('A', [self.x, self.u, self.kappa_r], [self.A])
        self.B_func = Function('B', [self.x, self.u, self.kappa_r], [self.B])

    def kappa_r_function(self, s_value):
        """Evaluate the road curvature at a specific s value."""
        return self.kapparef_s(s_value)

    def compute_jacobians(self, x_values, u_values, kappa_r_value):
        """Compute numerical values of the Jacobians A and B.
        
        Args:
            x_values (list): State values [s, n, delta_phi].
            u_values (list): Input values [v, delta].
            kappa_r_value (float): Curvature value at the current state.
        
        Returns:
            tuple: Numerical Jacobians A and B.
        """
        A_numeric = self.A_func(x_values, u_values, kappa_r_value)
        B_numeric = self.B_func(x_values, u_values, kappa_r_value)
        return A_numeric, B_numeric

# Example usage
if __name__ == "__main__":
    # Example s0 and kapparef data
    s0 = np.linspace(0, 10, 100)
    kapparef = np.sin(s0)


    L = 2.0

    # Initialize the model
    vehicle_model = Vehicle_LModel(s0, kapparef, L)

    # Example state and input values
    x_values = [0, 0, 0]  # [s, n, delta_phi]
    u_values = [1, 0]     # [v, delta]
    kappa_r_value = vehicle_model.kappa_r_function(0)

    # Measure computation time
    start_time = time.time()
    A_numeric, B_numeric = vehicle_model.compute_jacobians(x_values, u_values, kappa_r_value)
    end_time = time.time()

    print("A (Numeric):")
    print(A_numeric)
    print("\nB (Numeric):")
    print(B_numeric)
    print(f"\nComputation Time: {end_time - start_time:.6f} seconds")


