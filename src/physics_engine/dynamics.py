import numpy as np

from .rotation_utils import quat_mult, quat_conj

def compute_i_dot2_r_cm__i(f__i, m) -> np.ndarray:
    """
    Computes the inertial linear acceleration of the body's CM in inertial coordinates.

    :param f__i: Net external forces acting on the body expressed in inertial coordinates [N]
    :param m: Mass of the body [kg]
    :return: Inertial linear acceleration of the object's CM expressed in inertial coordinates [m/s^2]
    :rtype: ndarray
    """
 
    return f__i/m

def compute_b_dot_omega_b_wrt_i__b(t_cm__b, omega_b_wrt_i__b, J_cm__b) -> np.ndarray:
    """
    Computes the body angular acceleration in body coordinates
    
    :param t_cm__b: Net external Torques acting on the body expressed in body coordinates [N*m]
    :param omega_b_wrt_i__b: Angular velocity of the body frame w.r.t to the interial frame expressed in body coordinates [rad/s]
    :param J_cm__b: Inertia Matrix of the body w.r.t to its CM expressed in body coordinates [kg*m^2]
    :return: Body angular acceleration of the object expressed in body coordinates [rad/s^2]
    :rtype: ndarray[Any, Any]
    """
    return np.linalg.inv(J_cm__b)@(t_cm__b - np.cross(omega_b_wrt_i__b, J_cm__b @ omega_b_wrt_i__b))

def compute_dot_q_i_to_b__b(omega_b_wrt_i__b, q_i_to_b__b) -> np.ndarray:
    """
    Computes the time derivative of the rotation quaternion from inertial into body
    
    :param omega_b_wrt_i__b: Angular velocity of the body frame w.r.t to the interial frame expressed in body coordinates [rad/s]
    :param q_i_to_b__b: Rotation quaternion to rotate from inertial to body frame expressed in body coordinates
    :return: Time derivative of the rotation quaternion from inertial into body [1/s]
    :rtype: ndarray
    """
  
    return 0.5 * quat_mult(q_i_to_b__b, np.array([0, *omega_b_wrt_i__b]))  
    #q_b_to_i = q_i_to_b__b
    #omega_b_wrt_i__i = quat_mult(q_b_to_i, quat_mult(np.array([0, *omega_b_wrt_i__b]), quat_conj(q_b_to_i)))[1:]
    #return 0.5 * quat_mult(np.array([0, *omega_b_wrt_i__i]), q_i_to_b__b)

def compute_f_grav__i(m) -> np.ndarray:
    """
    Assumes Gravity points straight down and acts at the object's CM.
    
    :param m: Mass of the object [kg]
    :return: Force due gravity expressed in inertial coordinates[N]
    :rtype: ndarray
    """
    g__i = np.array([0, 0, -9.81])
    return m*g__i

def compute_KE_trans(m, i_dot_r_cm__i):
    return 0.5 * m * np.linalg.norm(i_dot_r_cm__i)**2

def compute_KE_rot(omega_b_wrt_i__b, J_cm__b):
    return 0.5 * omega_b_wrt_i__b.T @ J_cm__b @ omega_b_wrt_i__b

def compute_dot_state(t, y, params) -> np.ndarray:
    """
    Computes time deriavtive of the state vector
    
    :param t: Current Sim Time [s]
    :param y: State Vector
    :param params: Parametes such as mass, MOI, etc
    :return: Derivative of State Vector
    :rtype: ndarray
    """
    # Compute i_dot_r_cm__i
    i_dot_r_cm__i = y[3:6]

    # Compute i_dot2_r_cm__i
    f__i = np.array([0,0,0])
    m = params[0]
    i_dot2_r_cm__i = compute_i_dot2_r_cm__i(f__i, m)

    # Compute b_dot_omega_b_wrt_i__b
    t_cm__b = np.array([0,0,0])
    J_cm__b = params[1]
    omega_b_wrt_i__b = y[10:13]
    b_dot_omega_b_wrt_i__b = compute_b_dot_omega_b_wrt_i__b(t_cm__b, omega_b_wrt_i__b, J_cm__b)

    # Compute dot_q_i_to_b__b
    q_i_to_b__b = y[6:10]
    dot_q_i_to_b__b = compute_dot_q_i_to_b__b(omega_b_wrt_i__b, q_i_to_b__b)

    state_dot = np.concat((
        i_dot_r_cm__i,
        i_dot2_r_cm__i,
        dot_q_i_to_b__b,
        b_dot_omega_b_wrt_i__b,
    ))

    return state_dot