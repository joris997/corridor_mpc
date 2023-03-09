from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np


class FreeFlyerKinematics(object):
    def __init__(self, dt=None, **kwargs):
        """
        Astrobee Robot kinematics class class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param dt: sampling time of the discrete system, defaults to 0.01
        :type dt: float, optional
        """

        # Model
        self.nonlinear_model = self.astrobee_dynamics
        self.model = None
        # state is [x,y,th] so really no non-linearities
        self.n = 3
        self.m = 3

        # Tracking
        self.total_trajectory_time = None

        # Control bounds
        self.max_v = 4 #3.75 #0.5
        self.max_w = 4 #3.75 #0.2
        self.ulb = [-self.max_v, -self.max_v, -self.max_w]
        self.uub = [self.max_v, self.max_v, self.max_w]

        # Trajectory reference
        vx = self.max_v * 0.1
        wz = self.max_w * 0.1
        self.constant_v_tracking = np.array([[vx, 0, wz]]).T

        self.set_casadi_options()
        self.set_system_constants()
        self.set_dynamics()

        self.eta1 = []
        self.h1 = []
        self.h1_ineq = []
        self.eta2 = []
        self.h2 = []
        self.h2_ineq = []

        # Initialize barrier and trajectory variables
        if "cbfs" in kwargs:
            self.cbfs = kwargs["cbfs"]
        if "trajs" in kwargs:
            self.trajs = kwargs["trajs"]

        if self.cbfs is None:
            self.set_barrier_functions()
        else:
            self.ncbfs = len(self.cbfs)
            self.set_barrier_functions_waypoints()

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def set_dynamics(self):
        """
        Helper function to populate Astrobee's dynamics.
        """

        self.model = self.rk4_integrator(self.astrobee_dynamics)

        return

    def psi(self, euler):
        """
        Body to Inertial Attitude jacoboian matrix

        :param euler: euler vector with (roll, pitch,  yaw)
        :type euler: ca.MX, ca.DM, np.ndarray
        :return: attitude jacobian matrix
        :rtype: ca.MX
        """
        phi = euler[0]
        varphi = euler[1]

        Psi = ca.MX.zeros((3, 3))
        Psi[0, 0] = 1
        Psi[0, 1] = ca.sin(phi) * ca.tan(varphi)
        Psi[0, 2] = ca.cos(phi) * ca.tan(varphi)

        Psi[1, 1] = ca.cos(phi)
        Psi[1, 2] = -ca.sin(phi)

        Psi[2, 1] = ca.sin(phi) / ca.cos(varphi)
        Psi[2, 2] = ca.cos(phi) / ca.cos(varphi)

        return Psi

    def astrobee_dynamics(self, x, u):
        """
        Pendulum nonlinear dynamics.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        t = x[3:]

        # 3D Linear velocity
        v = u[0:2]

        # 3D Angular velocity
        w = u[2:]

        # Model
        pdot = v
        tdot = w

        dxdt = [pdot, tdot]

        return ca.vertcat(*dxdt)

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check best way to propagate
        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def get_trajectory(self, t0, npoints):
        """
        Generate trajectory to be followed.

        :param x0: starting position
        :type x0: ca.DM
        :param t0: starting time
        :type t0: float
        :param npoints: number of trajectory points
        :type npoints: int
        :return: trajectory with shape (Nx, npoints)
        :rtype: np.array
        """

        if t0 == 0.0:
            print("Creating trajectory...", end="")
            # Trajectory params
            traj_points = int(self.trajectory_time / self.dt) + npoints

            # Generate u_r
            u_r = self.constant_v_tracking
            u_r = np.repeat(u_r, traj_points, 1)

            # Initial trajectory point
            xr0 = self.trajectory_start_point

            # Generate the reference trajectory from the system dynamics
            self.trajectory_set = np.empty((self.m, 0))
            self.trajectory_set = np.append(self.trajectory_set, xr0, axis=1)
            for i in range(traj_points - 1):
                x_ri = self.model(self.trajectory_set[:, -1].reshape((self.m, 1)), u_r[:, i])
                self.trajectory_set = np.append(self.trajectory_set, x_ri, axis=1)
            x_r = self.trajectory_set[:, 0:npoints]
            print(" Done")
        else:
            id_s = int(round(t0 / self.dt))
            id_e = int(round(t0 / self.dt)) + npoints
            x_r = self.trajectory_set[:, id_s:id_e]

        return x_r

    def get_constant_u_sp(self, npoints):
        """
        Generate constant velocity input for the system.

        :param npoints: number of trajectory points
        :type npoints: int
        :return: constant velocity input
        :rtype: np.array
        """
        u_r = np.repeat(self.constant_v_tracking, npoints, axis=1)
        return u_r
    
    def get_trajectory_waypoints(self, t0, npoints, trajs):
        if t0 == 0.0:
            print("Creating trajectory...", end="")
            traj_points = int(self.trajectory_time / self.dt) + npoints
            times = np.linspace(0,traj_points*self.dt,traj_points)

            self.trajectory_set = np.empty((self.m,0))
            self.velocity_set = np.empty((self.m,0))
            for ti in times:
                # find the index of the line we care about
                tj = []
                for idx,traj in reversed(list(enumerate(trajs))):
                    if ti >= traj[0][0]:
                        tj = traj
                        break

                # now interpolate the trajectory
                t0 = tj[0][0]
                tf = tj[0][1]
                x0 = tj[0][2]
                xf = tj[0][3]
                x_ri = self.interpolate_state(ti,t0,tf,x0,xf).reshape((self.m,1))
                x_ri_2 = self.interpolate_state(ti+self.dt,t0,tf,x0,xf).reshape((self.m,1))
                v_ri = (x_ri_2 - x_ri)/self.dt
                # v_ri = np.zeros((3,1))
                self.trajectory_set = np.append(self.trajectory_set, x_ri, axis=1)
                self.velocity_set = np.append(self.velocity_set, v_ri, axis=1)

            x_r = self.trajectory_set[:, 0:npoints]
            v_r = self.velocity_set[:, 0:npoints-1]
            print(" Done")
        else:
            id_s = int(round(t0 / self.dt))
            id_e = int(round(t0 / self.dt)) + npoints
            x_r = self.trajectory_set[:, id_s:id_e]
            v_r = self.velocity_set[:, id_s:id_e-1]

        return x_r, v_r
    
    def interpolate_state(self,t,t0,tf,x0,xf):
        x_r = np.zeros((self.m,))
        for i in range(self.m):
            x_r[i] = (1-(t-t0)/(tf-t0))*x0[i] + ((t-t0)/(tf-t0))*xf[i]
        return x_r

    def set_trajectory(self, length, start):
        """
        Set trajectory type to be followed

        """

        self.trajectory_time = length
        self.trajectory_start_point = start

    def set_system_constants(self, l1=None, l2=None):
        """
        Helper method to set the constants for the barriers h1 and h2
        """

        # Barrier properties - position
        self.dt_p = 0.01
        self.eps_p = 1.32
        self.lambda_1 = 1.0075
        self.rah_1 = 0.2397

        # Barrier properties - attitude
        self.dt_t = 0.01
        self.eps_t = 0.4338
        self.lambda_2 = 1.6249
        self.rah_2 = 0.0650

        # Get minimum dt
        self.dt = min(self.dt_p, self.dt_t)

    def set_barrier_functions(self):
        """
        Helper method to set the desired barrier functions.

        :param hp: position barrier, defaults to None
        :type hp: ca.MX, optional
        :param hpdt: time-derivative of hp, defaults to None
        :type hpdt: ca.MX, optional
        :param hq: attitude barrier, defaults to None
        :type hq: ca.MX, optional
        :param hqdt: time-derivative of hq, defaults to None
        :type hqdt: ca.MX, optional
        """

        # Paper Translation barrier
        u = ca.MX.sym("u", self.n, 1)
        u1 = u[0:2]
        u2 = u[2:]

        # Setup position barrier
        p = ca.MX.sym("p", 2, 1)
        pr = ca.MX.sym("pr", 2, 1)

        h1 = self.eps_p**2 - ca.norm_2(p - pr)**2
        h1_ineq = - 2 * (p - pr).T @ u1 + self.lambda_1 * h1 - self.rah_1

        self.h1 = ca.Function('h1', [p, pr], [h1], self.fun_options)
        self.h1_ineq = ca.Function('h1_ineq', [p, pr, u], [h1_ineq], self.fun_options)

        # Setup attitude barrier
        t = ca.MX.sym("t", 1, 1)
        tr = ca.MX.sym("tr", 1, 1)

        h2 = self.eps_t**2 - ca.norm_2(t - tr)**2
        h2_ineq = - 2 * (t - tr).T @ u2 + self.lambda_2 * h2 - self.rah_2

        self.h2 = ca.Function('h2', [t, tr], [h2], self.fun_options)
        self.h2_ineq = ca.Function('h2_ineq', [t, tr, u], [h2_ineq], self.fun_options)

    def set_barrier_functions_waypoints(self):
        ### TODO: add Lipschitz and bounds of time-explicit eta
        
        # Paper Translation barrier
        time_var = ca.MX.sym("time", 1, 1)
        u = ca.MX.sym("u", self.n, 1)
        u1 = u[0:2]
        u2 = u[2:]

        # for now, just take the first barrier
        for i in range(self.ncbfs):
            tj = self.cbfs[i]
            t0 = tj[0][0]
            tf = tj[0][1]
            x0 = tj[0][2]
            xf = tj[0][3]
            dt = tf-t0
            eps = self.eps_p
            # eps = 10
        
            gamma = 15
            time = -1/gamma * ca.log(ca.exp(-gamma*time_var) + np.exp(-gamma*tf))
            time = ca.fmin(time_var,tf)

            # Setup position barrier
            eta_range = [ca.norm_2(x0[0:2]-xf[0:2]) + eps, eps]
            eta = eta_range[0]*(1-(time-t0)/(dt)) + eta_range[1]*((time-t0)/(dt))
            eta_dt = -eta_range[0]*(1/dt) + eta_range[1]*(1/dt)

            p = ca.MX.sym("p", 2, 1)
            pr = ca.MX.sym("pr", 2, 1)

            get_h = eta**2 - ca.norm_2(p-xf[0:2])**2
            get_dhdt = 2*eta * eta_dt
            get_dhdx = -2*(p-xf[0:2]).T @ u1

            h1 = get_h
            h1_ineq = get_dhdt + get_dhdx + self.lambda_1 * h1 - self.rah_1
            
            self.eta1.append( ca.Function('eta', [time_var], [eta], self.fun_options) )
            self.h1.append( ca.Function('h1', [time_var, p, pr], [h1], self.fun_options) )
            self.h1_ineq.append( ca.Function('h1_ineq', [time_var, p, pr, u], [h1_ineq], self.fun_options) )

            # Setup attitude barrier
            eps = self.eps_t

            eta_range = [ca.norm_2(x0[2:]-xf[2:]) + eps, eps]
            eta = eta_range[0]*(1-(time-t0)/(dt)) + eta_range[1]*((time-t0)/(dt))
            eta_dt = -eta_range[0]*(1/dt) + eta_range[1]*(1/dt)

            p = ca.MX.sym("p", 1, 1)
            pr = ca.MX.sym("pr", 1, 1)

            get_h = eta**2 - ca.norm_2(p-xf[2:])**2
            get_dhdt = 2*eta * eta_dt
            get_dhdx = -2*(p-xf[2:]).T @ u2

            h2 = get_h
            h2_ineq = get_dhdt + get_dhdx + self.lambda_2 * h2 - self.rah_2

            self.eta2.append( ca.Function('eta', [time_var], [eta], self.fun_options) )
            self.h2.append( ca.Function('h2', [time_var, p, pr], [h2], self.fun_options) )
            self.h2_ineq.append( ca.Function('h2_ineq', [time_var, p, pr, u], [h2_ineq], self.fun_options) )

    def get_barrier_constraint_value(self, t0, x_t, x_r, u_t, cbf_idx=0):
        """
        Helper function to get the barrier function values.

        :param x_t: system state
        :type x_t: np.array or ca.MX
        :param x_r: reference state
        :type x_r: np.array or ca.MX
        :param u_t: system control input
        :type u_t: np.array or ca.MX
        :return: values for the position and attitude barrier conditions
        :rtype: float
        """

        p = x_t[0:2]
        pr = x_r[0:2]
        t = x_t[2:]
        tr = x_r[2:]
        u = u_t

        if self.cbfs is None:
            # use the corridor CBF from Pedro
            # hp_ineq >= 0
            h1_ineq = self.h1_ineq(p, pr, u)
            # hq_ineq >= 0
            h2_ineq = self.h2_ineq(t, tr, u)
        else:
            # use the timed-waypoints CBFs from Joris
            # hp_ineq >= 0
            h1_ineq = self.h1_ineq[cbf_idx](t0, p, pr, u)
            # hq_ineq >= 0
            h2_ineq = self.h2_ineq[cbf_idx](t0, t, tr, u)

        return h1_ineq, h2_ineq

    def get_barrier_value(self, t0, x_t, x_r, cbf_idx=0):
        """
        Helper function to get the barrier function values.

        :param x_t: system state
        :type x_t: np.array or ca.MX
        :param x_r: reference state
        :type x_r: np.array or ca.MX
        :return: values for the position and attitude barrier conditions
        :rtype: float
        """

        p = x_t[0:2]
        t = x_t[2:]

        pr = x_r[0:2]
        tr = x_r[2:]

        if self.cbfs is None:
            # self.hp(p, pr, v, vr) - self.eps_p - self.eps_v
            e1 = self.h1(p,pr)
            # self.hq(q, qr, w, wr) - self.eps_q - self.eps_w
            e2 = self.h2(p,pr)
        else:
            # self.hp(p, pr, v, vr) - self.eps_p - self.eps_v
            e1 = self.h1[cbf_idx](t0,p,pr)
            # self.hq(q, qr, w, wr) - self.eps_q - self.eps_w
            e2 = self.h2[cbf_idx](t0,t,tr)

        return e1, e2

    def get_epsilon_value(self, t0, cbf_idx=0):
        if self.cbfs is None:
            eps1 = self.eps_p
            eps2 = self.eps_t
        else:
            eps1 = self.eta1[cbf_idx](t0)
            eps2 = self.eta2[cbf_idx](t0)
        
        return eps1, eps2