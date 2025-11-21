import numpy as np
import pandas as pd

from . import rk4
from . import dynamics
from . import rotation_utils

def simulate(simulation_config) -> pd.DataFrame:
    num_steps = int(simulation_config["sim_duration"]/simulation_config["step_size"])
    t = simulation_config["t0"]
    h = simulation_config["step_size"]
    T_i_to_b0 = rotation_utils.ypr_to_dcm(simulation_config["ypr"])

    time_array = np.zeros((num_steps+1,))
    state_array = np.zeros((num_steps+1, 13))

    #Extract initial states and append to state_array
    r_cm__i = simulation_config["r_cm__i"]
    i_dot_r_cm__i = simulation_config["i_dot_r_cm__i"]
    q_i_to_b__b = rotation_utils.ypr_to_rotation_quaternion(simulation_config["ypr"])
    omega_b_wrt_i__b = T_i_to_b0 @ simulation_config["omega_b_wrt_i__i"]

    initial_state_vector = np.concat((r_cm__i, i_dot_r_cm__i, q_i_to_b__b, omega_b_wrt_i__b))
    state_array[0, :] = initial_state_vector

    #Setup params array
    params = [simulation_config["m"], simulation_config["J_cm__b"]]
    
    for step in range(num_steps):
        t = time_array[step]
        state_vector = state_array[step,:]

        new_t, new_state_vector = rk4.rk4_step(t, state_vector, dynamics.compute_dot_state, h, params = params)
        time_array[step+1] = new_t
        state_array[step+1, :] = new_state_vector

    sim_data = pd.DataFrame({
        "t": time_array,
        "r_cm__i_x": state_array[:, 0], 
        "r_cm__i_y": state_array[:, 1],
        "r_cm__i_z": state_array[:, 2],
        "i_dot_r_cm__i_x": state_array[:, 3],
        "i_dot_r_cm__i_y": state_array[:, 4],
        "i_dot_r_cm__i_z": state_array[:, 5],
        "q_i_to_b__b_w": state_array[:, 6],
        "q_i_to_b__b_x": state_array[:, 7],
        "q_i_to_b__b_y": state_array[:, 8],
        "q_i_to_b__b_z": state_array[:, 9],
        "omega_b_wrt_i__b_x": state_array[:, 10],
        "omega_b_wrt_i__b_y": state_array[:, 11],
        "omega_b_wrt_i__b_z": state_array[:, 12],
    })

    return sim_data


def post_process_sim_data(sim_data, simulation_config):
    # The following section post-processes the simulation data and computes various variables which will be appeneded to the simulation dataframe

    # Compute T_i_to_b from q_i_to_b__b
    T_i_to_b = sim_data[["q_i_to_b__b_w", "q_i_to_b__b_x", "q_i_to_b__b_y", "q_i_to_b__b_z"]].apply(rotation_utils.rotation_quaternion_to_dcm, axis=1)
    sim_data["T_i_to_b"] = T_i_to_b

    #Compute KE_trans from m and i_dot_r_cm__b
    KE_trans =  0.5 * simulation_config["m"] * np.linalg.norm(sim_data[["i_dot_r_cm__i_x", "i_dot_r_cm__i_y", "i_dot_r_cm__i_z"]], axis = 1)
    sim_data["KE_trans"] = KE_trans

    #Compute KE_rot from J_cm__b and omega_b_wrt_i__b
    omega_b_wrt_i__b = sim_data[["omega_b_wrt_i__b_x", "omega_b_wrt_i__b_y", "omega_b_wrt_i__b_z"]].T
    KE_rot =  0.5 * np.sum(omega_b_wrt_i__b.to_numpy() * (simulation_config["J_cm__b"] @ omega_b_wrt_i__b), axis = 0)
    sim_data["KE_rot"] = KE_rot

    #Compute p_cm__i
    p_cm__i = simulation_config["m"] * sim_data[["i_dot_r_cm__i_x", "i_dot_r_cm__i_y", "i_dot_r_cm__i_z"]]
    sim_data[["p_cm__i_x", "p_cm__i_y", "p_cm__i_z"]] = p_cm__i 

    #Compute h_cm__i
    h_cm__b =  simulation_config["J_cm__b"] @ sim_data[["omega_b_wrt_i__b_x", "omega_b_wrt_i__b_y", "omega_b_wrt_i__b_z"]].to_numpy().T
    sim_data[["h_cm__b_x", "h_cm__b_y", "h_cm__b_z"]] = h_cm__b.T
    h_cm__i = np.empty_like(h_cm__b)

    for count, (T_i_to_b_count, h_cm__b_count) in enumerate(zip(T_i_to_b, h_cm__b.T)):
        h_cm__i_count = T_i_to_b_count.T @ h_cm__b_count
        h_cm__i[:, count] = h_cm__i_count
    sim_data[["h_cm__i_x", "h_cm__i_y", "h_cm__i_z"]] = h_cm__i.T

    return sim_data