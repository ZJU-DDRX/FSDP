/**
 * @file mpc_builder.cpp
 * @brief Ultimate C++ Implementation with ROBUST PTC Solver.
 * Fixes: NaN issues caused by singular P matrix and large integration steps.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

namespace py = pybind11;
using namespace Eigen;

const double INF_VAL = 1e20; 

// --- Helper Functions (Unchanged) ---
MatrixXd build_block_diag(const MatrixXd& block, int N) {
    int rows = block.rows();
    int cols = block.cols();
    MatrixXd res = MatrixXd::Zero(rows * N, cols * N);
    for (int i = 0; i < N; ++i) res.block(i * rows, i * cols, rows, cols) = block;
    return res;
}

VectorXd linspace(double start, double end, int num) {
    VectorXd res(num);
    if (num == 0) return res;
    if (num == 1) { res(0) = start; return res; }
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) res(i) = start + i * step;
    return res;
}

// --- MPC Matrix Construction (Unchanged) ---
py::tuple build_qp_matrices(
    int N, int N_ca, int dim_x, int dim_u, double dt, double L, 
    const VectorXd& x0,
    const MatrixXd& x_ref, const MatrixXd& u_ref, const VectorXd& kappa_ref,    
    const MatrixXd& Q1_d, const MatrixXd& Q2_d, const MatrixXd& Q3_d, const MatrixXd& R_d,
    const MatrixXd& A_u, const VectorXd& lb_u, const VectorXd& ub_u,
    const MatrixXd& A_du, const VectorXd& lb_du, const VectorXd& ub_du,
    const VectorXd& l_bound, const VectorXd& r_bound,
    double half_width, double margin
) {
    MatrixXd Q1 = build_block_diag(Q1_d, N);
    MatrixXd Q2 = build_block_diag(Q2_d, N);
    MatrixXd Q3 = build_block_diag(Q3_d, N);
    MatrixXd R  = build_block_diag(R_d, N);

    MatrixXd smooth_matrix = MatrixXd::Identity(N * dim_x, N * dim_x);
    for (int i = 1; i < N; ++i) smooth_matrix.block(i * dim_x, (i - 1) * dim_x, dim_x, dim_x) = -MatrixXd::Identity(dim_x, dim_x);

    MatrixXd As = MatrixXd::Zero(N * dim_x, dim_x);
    MatrixXd Bs = MatrixXd::Zero(N * dim_x, N * dim_u);
    MatrixXd Ad = MatrixXd::Identity(dim_x, dim_x); 
    MatrixXd Bd = MatrixXd::Zero(dim_x, dim_u);

    for (int i = 0; i < N; ++i) {
        double n_k = x_ref(i, 1), mu_k = x_ref(i, 2), v_k = u_ref(i, 0), del_k = u_ref(i, 1), kappa = kappa_ref(i);
        double c_mu = std::cos(mu_k), s_mu = std::sin(mu_k), t_del = std::tan(del_k), c_del = std::cos(del_k);
        double denom = 1.0 - kappa * n_k;
        if (std::abs(denom) < 1e-6) denom = 1e-6;
        double denom_sq = denom * denom;

        double d_sdot_d_n = (v_k * c_mu * kappa) / denom_sq;
        double d_sdot_d_mu = -(v_k * s_mu) / denom;
        double d_ndot_d_mu = v_k * c_mu;
        double d_mudot_d_n = -(kappa * kappa * v_k * c_mu) / denom_sq; 
        double d_mudot_d_mu = (kappa * v_k * s_mu) / denom;            

        double d_sdot_d_v = c_mu / denom;
        double d_ndot_d_v = s_mu;
        double d_mudot_d_v = (t_del / L) - (kappa * c_mu / denom);
        double d_mudot_d_del = v_k / (L * c_del * c_del);

        Ad.setIdentity();
        Ad(0, 1) += d_sdot_d_n * dt; Ad(0, 2) += d_sdot_d_mu * dt;
        Ad(1, 2) += d_ndot_d_mu * dt;
        Ad(2, 1) += d_mudot_d_n * dt; Ad(2, 2) += d_mudot_d_mu * dt;

        Bd.setZero();
        Bd(0, 0) = d_sdot_d_v * dt;
        Bd(1, 0) = d_ndot_d_v * dt;
        Bd(2, 0) = d_mudot_d_v * dt; Bd(2, 1) = d_mudot_d_del * dt;

        if (i == 0) As.block(0, 0, dim_x, dim_x) = Ad;
        else As.block(i * dim_x, 0, dim_x, dim_x) = Ad * As.block((i - 1) * dim_x, 0, dim_x, dim_x);

        Bs.block(i * dim_x, i * dim_u, dim_x, dim_u) = Bd;
        for (int j = 0; j < i; ++j) Bs.block(i * dim_x, j * dim_u, dim_x, dim_u) = Ad * Bs.block((i - 1) * dim_x, j * dim_u, dim_x, dim_u);
    }

    MatrixXd Q_total = Q1 + Q2 + smooth_matrix.transpose() * Q3 * smooth_matrix;
    MatrixXd P = Bs.transpose() * Q_total * Bs + R;

    VectorXd x_ref_vec(N * dim_x);
    for(int i=0; i<N; ++i) x_ref_vec.segment(i*dim_x, dim_x) = x_ref.row(i);
    VectorXd x0_extend = VectorXd::Zero(N * dim_x); x0_extend.head(dim_x) = x0;
    VectorXd As_x0 = As * x0;
    
    VectorXd term1 = Q1 * (As_x0 - x_ref_vec);
    VectorXd term2 = Q2 * As_x0;
    VectorXd term3 = smooth_matrix.transpose() * Q3 * (smooth_matrix * As_x0 - x0_extend);
    VectorXd q = Bs.transpose() * (term1 + term2 + term3);

    MatrixXd tmp = MatrixXd::Zero(N_ca, N * dim_x);
    for(int i=0; i<N_ca; ++i) tmp(i, i * dim_x + 1) = 1.0; 
    MatrixXd A_ca = tmp * Bs;
    VectorXd tmp_x0_ca = tmp * As_x0;
    VectorXd lb_ca = r_bound - tmp_x0_ca + VectorXd::Constant(N_ca, half_width + margin);
    VectorXd ub_ca = l_bound - tmp_x0_ca - VectorXd::Constant(N_ca, half_width + margin);

    MatrixXd A_final(A_u.rows() + A_du.rows() + A_ca.rows(), P.cols());
    A_final << A_u, A_du, A_ca;
    VectorXd l_final(lb_u.size() + lb_du.size() + lb_ca.size());
    l_final << lb_u, lb_du, lb_ca;
    VectorXd u_final(ub_u.size() + ub_du.size() + ub_ca.size());
    u_final << ub_u, ub_du, ub_ca;

    return py::make_tuple(P, q, A_final, l_final, u_final, As, Bs);
}

// --- Spline Fitting (Unchanged) ---
VectorXd solve_single_dim_fit(const VectorXd& t_data, const VectorXd& x_data, double v_start, double v_end) {
    int N = t_data.size();
    int n_vars = 6; int n_cons = 4;
    MatrixXd A_data(N, n_vars);
    for (int i = 0; i < N; ++i) {
        double t = t_data(i), t2 = t*t;
        A_data(i, 0) = 1.0; A_data(i, 1) = t; A_data(i, 2) = t2;
        A_data(i, 3) = t2*t; A_data(i, 4) = t2*t2; A_data(i, 5) = t2*t2*t;
    }
    MatrixXd C_cons = MatrixXd::Zero(n_cons, n_vars);
    VectorXd b_cons = VectorXd::Zero(n_cons);
    double t0 = t_data(0), tT = t_data(N-1);
    C_cons(0,0)=1; C_cons(0,1)=t0; C_cons(0,2)=std::pow(t0,2); C_cons(0,3)=std::pow(t0,3); C_cons(0,4)=std::pow(t0,4); C_cons(0,5)=std::pow(t0,5); b_cons(0)=x_data(0);
    C_cons(1,1)=1; C_cons(1,2)=2*t0; C_cons(1,3)=3*std::pow(t0,2); C_cons(1,4)=4*std::pow(t0,3); C_cons(1,5)=5*std::pow(t0,4); b_cons(1)=v_start;
    C_cons(2,0)=1; C_cons(2,1)=tT; C_cons(2,2)=std::pow(tT,2); C_cons(2,3)=std::pow(tT,3); C_cons(2,4)=std::pow(tT,4); C_cons(2,5)=std::pow(tT,5); b_cons(2)=x_data(N-1);
    C_cons(3,1)=1; C_cons(3,2)=2*tT; C_cons(3,3)=3*std::pow(tT,2); C_cons(3,4)=4*std::pow(tT,3); C_cons(3,5)=5*std::pow(tT,4); b_cons(3)=v_end;
    
    MatrixXd LHS = MatrixXd::Zero(n_vars + n_cons, n_vars + n_cons);
    VectorXd RHS = VectorXd::Zero(n_vars + n_cons);
    LHS.topLeftCorner(n_vars, n_vars) = 2.0 * A_data.transpose() * A_data;
    LHS.topRightCorner(n_vars, n_cons) = C_cons.transpose();
    LHS.bottomLeftCorner(n_cons, n_vars) = C_cons;
    RHS.head(n_vars) = 2.0 * A_data.transpose() * x_data;
    RHS.tail(n_cons) = b_cons;
    return LHS.colPivHouseholderQr().solve(RHS).head(n_vars).reverse();
}

py::tuple solve_quintic_spline(const VectorXd& t_data, const VectorXd& x_data, const VectorXd& y_data, double vx0, double vy0, double vxf, double vyf) {
    return py::make_tuple(solve_single_dim_fit(t_data, x_data, vx0, vxf), solve_single_dim_fit(t_data, y_data, vy0, vyf));
}

// --- Raw Traj Gen (Unchanged) ---
py::tuple gen_raw_traj_cpp(const VectorXd& s, double current_d, const VectorXd& d_left, const VectorXd& d_right, const std::vector<int>& obs_idx, const VectorXd& obs_center, const VectorXd& obs_min_dist, const std::string& side) {
    int N = s.size(); VectorXd d = VectorXd::Zero(N); bool valid = true;
    if (obs_idx.empty()) return py::make_tuple(d, false);
    int idx_start_es = std::max(obs_idx.front() - 2, 0);
    int idx_mid_es = obs_idx[obs_idx.size() / 2];
    int idx_end_es = std::min(obs_idx.back() + 6, N - 1);
    int idx_mid_obs = obs_idx.size() / 2;
    int idx_end_obs = obs_idx.size() - 1;
    double safe_dist = 0.1, lon_dist = 1.5;

    if ((current_d < 0 && side == "left") || (current_d > 0 && side == "right")) {
        if (std::abs(s(idx_start_es) - s(0)) < lon_dist) return py::make_tuple(d, false);
    }

    double start_d, mid_d, end_d;
    if (side == "left") {
        mid_d = std::max(std::min(d_left(idx_mid_es) - safe_dist, obs_min_dist(idx_mid_obs) + safe_dist), obs_min_dist(idx_mid_obs));
        start_d = std::max(std::min(d_left(idx_start_es) - safe_dist, obs_center(0) + safe_dist), obs_min_dist(0));
        end_d = std::max(std::min(d_left(idx_end_es) - safe_dist, obs_center(idx_end_obs) + safe_dist), obs_min_dist(idx_end_obs));
        if (idx_start_es == 0) start_d = current_d;
        else if (idx_start_es < 4) start_d = current_d + (double(idx_start_es)/idx_mid_es)*(mid_d - current_d);
        else start_d = std::min(start_d, mid_d);
        end_d = std::min(end_d, mid_d);
    } else {
        mid_d = -std::max(std::min(d_right(idx_mid_es) - safe_dist, obs_min_dist(idx_mid_obs) + safe_dist), obs_min_dist(idx_mid_obs));
        start_d = -std::max(std::min(d_right(idx_start_es) - safe_dist, obs_center(0) + safe_dist), obs_min_dist(0));
        end_d = -std::max(std::min(d_right(idx_end_es) - safe_dist, obs_center(idx_end_obs) + safe_dist), obs_min_dist(idx_end_obs));
        if (idx_start_es == 0) start_d = current_d;
        else if (idx_start_es < 4) start_d = current_d + (double(idx_start_es)/idx_mid_es)*(mid_d - current_d);
        else start_d = std::max(start_d, mid_d);
        end_d = std::max(end_d, mid_d);
    }

    if (idx_start_es == 0) d(0) = current_d;
    else d.segment(0, idx_start_es) = linspace(current_d, start_d, idx_start_es);
    if (idx_mid_es > idx_start_es) d.segment(idx_start_es, idx_mid_es - idx_start_es) = linspace(start_d, mid_d, idx_mid_es - idx_start_es);
    if (idx_end_es > idx_mid_es) d.segment(idx_mid_es, idx_end_es - idx_mid_es) = linspace(mid_d, end_d, idx_end_es - idx_mid_es);
    if (N > idx_end_es) d.segment(idx_end_es, N - idx_end_es) = linspace(end_d, 0.0, N - idx_end_es);

    for (int i = 0; i < N; ++i) {
        double bound = (side == "left") ? d_left(i) : d_right(i);
        if (std::abs(d(i)) > (std::abs(bound) - safe_dist)) { valid = false; break; }
    }
    return py::make_tuple(d, valid);
}

VectorXd calculate_path_time_cpp(const VectorXd& x, const VectorXd& y, const VectorXd& v) {
    int N = x.size(); VectorXd t = VectorXd::Zero(N);
    for (int i = 1; i < N; ++i) {
        double dist = std::sqrt(std::pow(x(i)-x(i-1), 2) + std::pow(y(i)-y(i-1), 2));
        double vel = std::abs(v(i-1));
        if (vel < 1e-6) vel = 1e-6;
        t(i) = t(i-1) + dist / vel;
    }
    return t;
}

// ==========================================
// PART 4: PTC Solver (Robust Implementation)
// ==========================================

VectorXd g_ptc(const VectorXd& z, const VectorXd& b) {
    return z.cwiseMin(b);
}

VectorXd solve_qp_ptc_cpp(
    const MatrixXd& P, const VectorXd& q_cost,
    const MatrixXd& A, const VectorXd& l, const VectorXd& u
) {
    int n_vars = P.cols();
    int n_cons = A.rows();

    // 1. Filter Constraints
    std::vector<int> eq_indices;
    std::vector<int> ineq_indices; 
    std::vector<int> ineq_indices_lower;
    double eq_tol = 1e-5; 

    for (int i = 0; i < n_cons; ++i) {
        if (std::abs(l(i) - u(i)) < eq_tol) {
            eq_indices.push_back(i);
        } else {
            if (u(i) < INF_VAL) ineq_indices.push_back(i);
            if (l(i) > -INF_VAL) ineq_indices_lower.push_back(i);
        }
    }

    int n_eq = eq_indices.size();
    MatrixXd C = MatrixXd::Zero(n_eq, n_vars);
    VectorXd p = VectorXd::Zero(n_eq);
    for (int i = 0; i < n_eq; ++i) {
        C.row(i) = A.row(eq_indices[i]);
        p(i) = l(eq_indices[i]);
    }

    int n_ineq = ineq_indices.size() + ineq_indices_lower.size();
    MatrixXd D = MatrixXd::Zero(n_ineq, n_vars);
    VectorXd q_ineq = VectorXd::Zero(n_ineq);
    
    int current_row = 0;
    for (int idx : ineq_indices) {
        D.row(current_row) = A.row(idx);
        q_ineq(current_row) = u(idx);
        current_row++;
    }
    for (int idx : ineq_indices_lower) {
        D.row(current_row) = -A.row(idx);
        q_ineq(current_row) = -l(idx);
        current_row++;
    }

    // 2. Precompute Matrices
    // [Fix 1] Regularize P to ensure positive definiteness (Avoid LLT failure)
    MatrixXd P_reg = P + 1e-6 * MatrixXd::Identity(n_vars, n_vars);
    
    // [Fix 2] Use LDLT instead of LLT for better stability
    LDLT<MatrixXd> ldlt_P(P_reg);
    MatrixXd H_inv = ldlt_P.solve(MatrixXd::Identity(n_vars, n_vars));

    MatrixXd CHinvCT = C * H_inv * C.transpose();
    // Regularize CHinvCT too
    CHinvCT += 1e-8 * MatrixXd::Identity(n_eq, n_eq);
    LDLT<MatrixXd> ldlt_C(CHinvCT);
    MatrixXd CHinvCT_inv = ldlt_C.solve(MatrixXd::Identity(n_eq, n_eq));

    MatrixXd HinvCT = H_inv * C.transpose();
    MatrixXd term = HinvCT * CHinvCT_inv * HinvCT.transpose();
    MatrixXd G_prime = H_inv - term;

    VectorXd Hinv_c = H_inv * q_cost;
    VectorXd inner = C * Hinv_c + p;
    VectorXd h_prime = H_inv * (C.transpose() * (CHinvCT_inv * inner) - q_cost);

    MatrixXd M1 = D * G_prime * D.transpose();
    VectorXd m2 = D * h_prime;

    // 3. Integration (Euler Step)
    // [Fix 3] Smaller step size and more iterations for stability
    double beta = 10.0;
    double h_step = 0.05; // Reduced from 0.5 to 0.05
    int max_iter = 200;   // Increased iterations

    VectorXd lambda = VectorXd::Zero(n_ineq);

    for (int k = 0; k < max_iter; ++k) {
        VectorXd tmp1 = M1 * lambda;
        VectorXd inside = tmp1 + m2;
        VectorXd g_val = g_ptc(inside, q_ineq);
        
        VectorXd derivative = beta * (-tmp1 + g_val - m2);
        
        // Simple check for divergence
        if (lambda.hasNaN()) {
            // Fallback or reset if divergence detected
            lambda.setZero();
            break; 
        }

        lambda = lambda + h_step * derivative;
    }

    // 4. Recover Primal Solution
    VectorXd y_opt = G_prime * D.transpose() * lambda + h_prime;

    return y_opt;
}




PYBIND11_MODULE(mpc_builder, m) {
    m.doc() = "Ultimate MPC Builder & PTC Solver";
    m.def("build_qp_matrices", &build_qp_matrices, "Build MPC matrices");
    m.def("solve_quintic_spline", &solve_quintic_spline, "Solve quintic spline");
    m.def("gen_raw_traj_cpp", &gen_raw_traj_cpp, "Generate raw traj");
    m.def("calculate_path_time_cpp", &calculate_path_time_cpp, "Calc time");
    m.def("solve_qp_ptc_cpp", &solve_qp_ptc_cpp, "Solve QP using PTC");
}