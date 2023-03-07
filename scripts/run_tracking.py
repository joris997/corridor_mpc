#!/usr/bin/env python
import numpy as np
import pickle

from corridor_mpc.models.ff_kinematics import FreeFlyerKinematics
from corridor_mpc.controllers.corridor_mpc import CorridorMPC
from corridor_mpc.simulation_trajectory import EmbeddedSimEnvironment

# Sim and MPC Params
SIM_TIME = 15.0
# SIM_TIME = 0.1
Q = np.diag([100, 100, 50])
R = np.diag([50, 50, 30])
P = Q * 100

# Instantiante Model
abee = FreeFlyerKinematics()

# Instantiate controller (to track a velocity)
ctl = CorridorMPC(model=abee,
                  dynamics=abee.model,
                  horizon=.3,
                  solver_type='ipopt',
                  Q=Q, R=R, P=P,
                  ulb=abee.ulb,
                  uub=abee.uub,
                  set_zcbf=True)

# Trajectory from Robust STL
with open("mp_data/prob1_export.pkl",'rb') as f:
    cbfs,lines = pickle.load(f)

xr0 = np.zeros((3, 1))
abee.set_trajectory(length=SIM_TIME, start=xr0)
sim_env_full = EmbeddedSimEnvironment(model=abee,
                                      dynamics=abee.model,
                                      cmpc=ctl,
                                      noise={"pos": 0.1, "att": 0.1},
                                      time=SIM_TIME, collect=True,
                                      animate=False,
                                      cbfs=cbfs,
                                      trajectories=lines)
                                      
_, _, _, avg_ct = sim_env_full.run([0.0, 0.0, 0.0])

print("Average computational cost:", avg_ct)
