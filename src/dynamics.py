import numpy as np

import quaternion

def compute_i_dot2_r_cm__b(f__b, m) -> np.ndarray:
    """
    Computes the inertial linear acceleration of the body's CM in body coordinates.

    :param f__b: Net external forces acting on the body expressed in body coordinates [N]
    :param m: Mass of the body [kg]
    :return: Inertial linear acceleration of the object's CM expressed in body coordinates [m/s^2]
    :rtype: ndarray
    """
    return f__b/m

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
    :param q_i_to_b__b: Rotation quaternion to rotate from inertial to body frame
    :return: Time derivative of the rotation quaternion from inertial into body [1/s]
    :rtype: ndarray
    """
    return 0.5 * quaternion.quat_mult(np.array([0, *omega_b_wrt_i__b]), q_i_to_b__b)