import numpy as np
import pandas as pd

from physics_engine import simulater

simulation_config = {
    "t0": 0,
    "r_cm__i": np.array([0,0,0]), # [m]
    "i_dot_r_cm__i": np.array([0,5,5]), # [m/s]
    "ypr": np.array([0,0,0]), # [deg]
    "omega_b_wrt_i__i": np.array([0, 3, 0]), # [deg/s]
    "m": 1,
    "J_cm__b": np.array([[1,0,0],[0,2,0],[0,0,3]]), # [kg*m^2]
    "step_size": 0.01, # [s]
    "sim_duration": 10 #[s]
}

sim_data = simulater.simulate(simulation_config)
modified_sim_data = simulater.post_process_sim_data(sim_data, simulation_config)
modified_sim_data.to_csv("sim_data.csv", index = False)