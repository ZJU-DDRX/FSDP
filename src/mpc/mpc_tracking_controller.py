import numpy as np
import osqp
import scipy.sparse as sparse

from mpc.linear_model import Vehicle_LModel

class MPC_Tracking_Controller:
    def __init__(self):
        self.N = 50
        self.DIM_X = 3
        self.DIM_U = 2
        self.L = 0.32

        self.u_min = np.array([0, -0.4])
        self.u_max = np.array([10, 0.4])
        self.du_min = np.array([-5, -5])
        self.du_max = np.array([5, 5])
        self.dt = 0.1  # time step
        self.vehicle_Lmodel = None

        self._A_u = None
        self._lb_u = None
        self._ub_u = None
        self._A_du = None
        self._lb_du = None
        self._ub_du = None
        self._Q1 = None
        self._Q2 = None
        self._R = None

        self.init_default_params()

    def update_vehicle_Lmodel(self, s_ref, kapparef):
        self.vehicle_Lmodel = Vehicle_LModel(s_ref, kapparef, self.L)

    def init_default_params(self):
        """ init default parameters """
        self._A_u, self._lb_u, self._ub_u = self.get_control_limits_cons()
        self._A_du, self._lb_du, self._ub_du = self.get_control_rate_limits_cons()
        self._Q1, self._Q2, self._R = self.get_cost_matrix()

    def get_control_limits_cons(self):
        """ u_min <= u <= u_max, lb <= Iu <= ub """
        lb = np.tile(self.u_min, (self.N, 1)).reshape(-1,)
        ub = np.tile(self.u_max, (self.N, 1)).reshape(-1,)
        A = np.eye(self.N * self.DIM_U)
        
        return A, lb, ub
    
    def get_control_rate_limits_cons(self):
        """ du_min <= u - u_prev <= du_max, lbA <= Au <= ubA """
        lbA = np.tile(self.du_min * self.dt, (self.N, 1)).reshape(-1,)
        ubA = np.tile(self.du_max * self.dt, (self.N, 1)).reshape(-1,)
        lbA[0:2] = self.u_min
        ubA[0:2] = self.u_max

        A = np.eye(self.N * self.DIM_U)
        for i in range(1, self.N):
            idx = i * self.DIM_U
            A[idx, idx - self.DIM_U] = -1
            A[idx + 1, idx + 1 - self.DIM_U] = -1

        return A, lbA, ubA
    
    def get_cur_cost_matrix(self):
        Q1_d = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0]])
        Q2_d = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]])
        R_d = np.array([[0.5, 0], [0, 0.0]])

        return Q1_d, Q2_d, R_d
    
    def get_cost_matrix(self):
        Q1 = np.zeros((self.N * self.DIM_X, self.N * self.DIM_X))
        Q2 = np.zeros((self.N * self.DIM_X, self.N * self.DIM_X))
        R = np.zeros((self.N * self.DIM_U, self.N * self.DIM_U))

        for i in range(self.N):
            idx_x = i * self.DIM_X
            idx_u = i * self.DIM_U

            Q1_d, Q2_d, R_d = self.get_cur_cost_matrix()
            Q1[idx_x:idx_x + self.DIM_X, idx_x:idx_x + self.DIM_X] = Q1_d
            Q2[idx_x:idx_x + self.DIM_X, idx_x:idx_x + self.DIM_X] = Q2_d
            R[idx_u:idx_u + self.DIM_U, idx_u:idx_u + self.DIM_U] = R_d

        return Q1, Q2, R
    
    def get_linear_matrix(self, x_ref, u_ref):
        """ get system linearization matrix """
        kappa = self.vehicle_Lmodel.kappa_r_function(x_ref[0])
        A_numeric, B_numeric = self.vehicle_Lmodel.compute_jacobians(x_ref, u_ref, kappa)
        Ad = np.array(A_numeric)
        Bd = np.array(B_numeric)

        # apply forward euler discretization
        Ad = np.eye(self.DIM_X) + Ad * self.dt
        Bd = Bd * self.dt

        return Ad, Bd
    
    def get_system_matrix(self, x_ref, u_ref):
        """ get system matrix X = A * x0 + B * U """
        A = np.zeros((self.N * self.DIM_X, self.DIM_X))
        B = np.zeros((self.N * self.DIM_X, self.N * self.DIM_U))

        for i in range(self.N):
            idx_x = i * self.DIM_X
            idx_u = i * self.DIM_U

            Ad, Bd = self.get_linear_matrix(x_ref[i, :], u_ref[i, :])
            if i == 0:
                A[idx_x:idx_x + self.DIM_X, :] = Ad
                B[idx_x:idx_x + self.DIM_X, idx_u:idx_u + self.DIM_U] = Bd
            else:
                idx_x_prev = (i - 1) * self.DIM_X
                A[idx_x:idx_x + self.DIM_X, :] = np.dot(Ad, A[idx_x_prev:idx_x, :])
                for j in range(i):
                    idx_u_j = j * self.DIM_U
                    B[idx_x:idx_x + self.DIM_X, idx_u_j:idx_u_j + self.DIM_U] = np.dot(
                        Ad, B[idx_x_prev:idx_x, idx_u_j:idx_u_j + self.DIM_U])
                B[idx_x:idx_x + self.DIM_X, idx_u:idx_u + self.DIM_U] = Bd
        
        return A, B
    
    def define_QP_matrices(self, x0, x_ref, u_ref):
        """ define QP matrices and return system matrices """
        As, Bs = self.get_system_matrix(x_ref, u_ref)

        P = Bs.T @ (self._Q1 + self._Q2) @ Bs + self._R
        q_transpose = (As @ x0 - x_ref.flatten()).reshape(1, -1) @ self._Q1 @ Bs + (As @ x0).reshape(1, -1) @ self._Q2 @ Bs
        q = q_transpose.reshape(-1,)

        A = np.vstack((self._A_u, self._A_du))
        l = np.hstack((self._lb_u, self._lb_du))
        u = np.hstack((self._ub_u, self._ub_du))
    
        return P, q, A, l, u, As, Bs
    
    def solve_qp(self, x0, x_ref, u_ref):
        """ 
        solve QP problem 
        Args:
            x0: current state, np.array([s, n, delta_phi])
            x_ref: reference states, np.array([[s, n, delta_phi]])
            u_ref: reference controls inputs, np.array([[v, delta]])
        Returns:
            optimal_control: optimal control inputs, control inputs for the next N steps (DIM_U * N, )
            x_next: optimal states (DIM_X * N, )
        """
        P, q, A, l, u, As, Bs = self.define_QP_matrices(x0, x_ref[:self.N, :], u_ref[:self.N, :])
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)

        m = osqp.OSQP()
        m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        res = m.solve()
        print(f"OSQP solver time: {res.info.solve_time * 1000} ms and the solver status: {res.info.status}")

        optimal_control = None
        x_next = None
        if res.info.status == 'solved':
            optimal_control = res.x
            x_next = As @ x0 + Bs @ optimal_control

        return optimal_control, x_next
    

if __name__ == "__main__":
    mpc_controller = MPC_Tracking_Controller()
    x_ref = np.random.rand(mpc_controller.N, 3)
    x_ref[:, 0] = np.sort(x_ref[:, 0])
    u_ref = np.random.rand(mpc_controller.N, 2)
    kappa_r = np.random.rand(mpc_controller.N)

    x0 = np.array([0, 0, 0])

    mpc_controller.update_vehicle_Lmodel(x_ref[:, 0], kappa_r)
    u, x = mpc_controller.solve_qp(x0, x_ref, u_ref)
    print(f'the shape of the optimal control is {u.shape} and the shape of the optimal states is {x.shape}')
