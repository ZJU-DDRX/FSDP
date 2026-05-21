import numpy as np
import osqp
import scipy.sparse as sparse
import time
from mpc.mpc_builder_loader import load_mpc_builder
from mpc.linear_model import Vehicle_LModel

mpc_builder = load_mpc_builder()
class CodeTimer:
    
    stats = {}

    def __init__(self, name=None, verbose=False):
        self.name = name
        self.verbose = verbose 
    def __enter__(self):
        self.tstart = time.perf_counter()

    def __exit__(self, type, value, traceback):
        
        self.elapsed = (time.perf_counter() - self.tstart) * 1000
        
        
        if self.name:
            if self.name not in CodeTimer.stats:
                CodeTimer.stats[self.name] = []
            CodeTimer.stats[self.name].append(self.elapsed)

        
        if self.verbose:
            print(f"--> [{self.name}]: {self.elapsed:.4f} ms")

    @staticmethod
    def print_summary(clear_history=True):
        """
        Print the performance summary report (Avg, Max, Min, Count).
        """
        if not CodeTimer.stats:
            return

        print("\n" + "="*80)
        print(f"{'PERFORMANCE REPORT':^80}")
        print("="*80)
        print(f"{'Metric Name':<35} | {'Avg (ms)':<10} | {'Max (ms)':<10} | {'Min (ms)':<10} | {'Count':<5}")
        print("-" * 80)

        
        for name, times in CodeTimer.stats.items():
            if not times: continue
            
            avg_t = np.mean(times)
            max_t = np.max(times)
            min_t = np.min(times)
            count = len(times)

            print(f"{name:<35} | {avg_t:<10.4f} | {max_t:<10.4f} | {min_t:<10.4f} | {count:<5}")

        print("="*80 + "\n")

        
        if clear_history:
            CodeTimer.stats = {}
class MPC_Tracking_Controller:
    def __init__(self):
        self.N = 40
        self.N_ca = 40
        self.DIM_X = 3
        self.DIM_U = 2
        self.L = 0.32

        self.u_min = np.array([0, -0.4])
        self.u_max = np.array([10, 0.4])
        self.du_min = np.array([-5, -0.5])
        self.du_max = np.array([5, 0.5])
        self.dt = 0.1  # time step
        self.vehicle_Lmodel = None

        self.half_width = 0.28 / 2.0
        self.margin = 0.05 #maximum value: 0.1

        self._A_u = None
        self._lb_u = None
        self._ub_u = None
        self._A_du = None
        self._lb_du = None
        self._ub_du = None
        self._Q1 = None
        self._Q2 = None
        self._Q3 = None
        self._R = None

        self.smooth_matrix = None
        self.init_default_params()

    def update_vehicle_Lmodel(self, s_ref, kapparef):
        self.vehicle_Lmodel = Vehicle_LModel(s_ref, kapparef, self.L)

    def init_default_params(self):
        """ init default parameters """
        self._A_u, self._lb_u, self._ub_u = self.get_control_limits_cons()
        self._A_du, self._lb_du, self._ub_du = self.get_control_rate_limits_cons()
        self._Q1, self._Q2, self._Q3, self._R = self.get_cost_matrix()
        self.get_smooth_matrix()

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
        Q1_d = np.array([[10, 0, 0], [0, 100, 0], [0, 0, 0]]) # for reference tracking
        Q2_d = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) # for race line
        Q3_d = np.array([[0, 0, 0], [0, 0.1, 0], [0, 0, 0]])  # for smoothness
        R_d = np.array([[0.2, 0], [0, 0]])

        return Q1_d, Q2_d, Q3_d, R_d
    
    def get_cost_matrix(self):
        Q1 = np.zeros((self.N * self.DIM_X, self.N * self.DIM_X))
        Q2 = np.zeros((self.N * self.DIM_X, self.N * self.DIM_X))
        Q3 = np.zeros((self.N * self.DIM_X, self.N * self.DIM_X))
        R = np.zeros((self.N * self.DIM_U, self.N * self.DIM_U))

        for i in range(self.N):
            idx_x = i * self.DIM_X
            idx_u = i * self.DIM_U

            Q1_d, Q2_d, Q3_d, R_d = self.get_cur_cost_matrix()
            Q1[idx_x:idx_x + self.DIM_X, idx_x:idx_x + self.DIM_X] = Q1_d
            Q2[idx_x:idx_x + self.DIM_X, idx_x:idx_x + self.DIM_X] = Q2_d
            Q3[idx_x:idx_x + self.DIM_X, idx_x:idx_x + self.DIM_X] = Q3_d
            R[idx_u:idx_u + self.DIM_U, idx_u:idx_u + self.DIM_U] = R_d

        return Q1, Q2, Q3, R
    
    def get_smooth_matrix(self):
        I = np.eye(self.N * self.DIM_X)
        tmp = np.zeros((self.N * self.DIM_X, self.N * self.DIM_X))
        for i in range(1, self.N):
            idx_x = i * self.DIM_X
            idx_x_prev = (i - 1) * self.DIM_X
            tmp[idx_x:idx_x + self.DIM_X, idx_x_prev:idx_x] = np.eye(self.DIM_X)

        self.smooth_matrix = I - tmp
    
    def derive_ca_constraints(self, x0, As, Bs, l_bound, r_bound):
        """ 
        derive collision avoidance constraints 
        Args:
            x0: current state, np.array([s, n, delta_phi])
            As: system matrix
            Bs: control matrix
            l_bound: left boundary, np.array
            r_bound: right boundary, np.array
        Returns:
            A_ca: collision avoidance constraints matrix
            lb_ca: lower bound
            ub_ca: upper bound
        """
        tmp = np.zeros((self.N_ca, self.N * self.DIM_X))
        for i in range(self.N_ca):
            idx_x = i * self.DIM_X
            tmp[i, idx_x:idx_x + self.DIM_X] = np.array([0, 1, 0])
        A_ca = np.dot(tmp, Bs)

        tmp_x0 = np.dot(tmp, As @ x0)
        lb_ca = r_bound - tmp_x0 + self.half_width + self.margin
        ub_ca = l_bound - tmp_x0 - self.half_width - self.margin

        return A_ca, lb_ca, ub_ca
    
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
        t0=time.perf_counter()
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

        print(f"  [DEBUG] System Matrix Loop Time: {(time.perf_counter()-t0)*1000:.4f} ms")
        
        return A, B
    
    def define_QP_matrices(self, x0, x_ref, u_ref, l_bound, r_bound):
        """ define QP matrices and return system matrices """
        As, Bs = self.get_system_matrix(x_ref, u_ref)
        x0_extend = np.zeros((self.N * self.DIM_X,))
        x0_extend[:self.DIM_X] = x0

        P = Bs.T @ (self._Q1 + self._Q2 + self.smooth_matrix.T @ self._Q3 @ self.smooth_matrix) @ Bs + self._R
        q1_transpose = (As @ x0 - x_ref.flatten()).reshape(1, -1) @ self._Q1 @ Bs
        q2_transpose = (As @ x0).reshape(1, -1) @ self._Q2 @ Bs
        q3_transpose = (self.smooth_matrix @ As @ x0 - x0_extend).reshape(1, -1) @ self._Q3 @ self.smooth_matrix @ Bs

        q_transpose = q1_transpose + q2_transpose + q3_transpose
        q = q_transpose.reshape(-1,)

        A_ca, lb_ca, ub_ca = self.derive_ca_constraints(x0, As, Bs, l_bound, r_bound)

        A = np.vstack((self._A_u, self._A_du, A_ca))
        l = np.hstack((self._lb_u, self._lb_du, lb_ca))
        u = np.hstack((self._ub_u, self._ub_du, ub_ca))
    
        return P, q, A, l, u, As, Bs
    # def get_linear_model_matrices(self, x_ref, u_ref):
    #     """
    #     Calculate all linearized Ad and Bd matrices for the prediction horizon in Python.
    #         This is O(N), not the bottleneck. The O(N^2) part is handled in C++.
    #     """
    #     Ad_list = []
    #     Bd_list = []
    #     for i in range(self.N):
    #         Ad, Bd = self.get_linear_matrix(x_ref[i], u_ref[i])
    #         Ad_list.append(Ad)
    #         Bd_list.append(Bd)
    #     return Ad_list, Bd_list
    
    def solve_qp(self, x0, x_ref, u_ref, l_bound, r_bound, solver_type="ptc"): # Solver backend selector
        """ 
        solve QP problem using Ultimate C++ Builder and selectable solver.
        """
        

        # [Phase 2] C++ Build
        with CodeTimer("1. Ultimate QP Build (C++)"):
            # [Phase 1] Prepare Data (Same as before)
            s_query = x_ref[:, 0] % self.vehicle_Lmodel.s0[-1]
            kappa_ref = np.interp(s_query, self.vehicle_Lmodel.s0, self.vehicle_Lmodel.kapparef)
            Q1_d, Q2_d, Q3_d, R_d = self.get_cur_cost_matrix()
            P_dense, q, A_dense, l, u, As, Bs = mpc_builder.build_qp_matrices(
                self.N, self.N_ca, self.DIM_X, self.DIM_U, self.dt, self.L,
                x0, x_ref[:self.N, :], u_ref[:self.N, :], kappa_ref[:self.N],
                Q1_d, Q2_d, Q3_d, R_d,
                self._A_u, self._lb_u, self._ub_u,
                self._A_du, self._lb_du, self._ub_du,
                l_bound[:self.N_ca], r_bound[:self.N_ca],
                self.half_width, self.margin
            )

        # [Phase 3] Solve
        optimal_control = None
        x_next = None

        if solver_type == "osqp":
            with CodeTimer("2. OSQP Solve"):
                P = sparse.csc_matrix(P_dense)
                A = sparse.csc_matrix(A_dense)
                m = osqp.OSQP()
                m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
                res = m.solve()
                if res.info.status == 'solved' or res.info.status == 'solved inaccurate':
                    optimal_control = res.x
        
        elif solver_type == "ptc":
            with CodeTimer("2. PTC Solve (C++)"):
                # Call the C++ PTC solver directly.
                # Inputs must be float64 numpy arrays for Eigen MatrixXd.
                optimal_control = mpc_builder.solve_qp_ptc_cpp(
                    P_dense, q, A_dense, l, u
                )

        # Recover next state (Same for both)
        if optimal_control is not None:
            x_next = As @ x0 + Bs @ optimal_control

        return optimal_control, x_next

if __name__ == "__main__":
    mpc_controller = MPC_Tracking_Controller()
    x_ref = np.random.rand(mpc_controller.N, 3)
    x_ref[:, 0] = np.sort(x_ref[:, 0])
    u_ref = np.random.rand(mpc_controller.N, 2)
    kappa_r = np.random.rand(mpc_controller.N)

    x0 = np.array([0, 0, 0])
    l_bound = np.ones((mpc_controller.N_ca,)) *  0.6
    r_bound = np.ones((mpc_controller.N_ca,)) * -0.6

    mpc_controller.update_vehicle_Lmodel(x_ref[:, 0], kappa_r)
    u, x = mpc_controller.solve_qp(x0, x_ref, u_ref, l_bound, r_bound)
    print(f'the shape of the optimal control is {u.shape} and the shape of the optimal states is {x.shape}')
