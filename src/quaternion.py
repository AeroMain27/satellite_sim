import numpy as np

def quat_mult(p, q) -> np.ndarray:
    """
    Multiplies together 2 quaternions. Scalar Part First!
    
    :param p: Left Quaternion
    :param q: Right Quaternion
    :return: Left Quaternion * Right Quaternion
    :rtype: ndarray
    """
    p0 = p[0]
    q0 = q[0]
    pv = p[1:4]
    qv = q[1:4]

    prod0 = p0 * q0 - np.dot(pv, qv)
    prodv = p0 * qv + q0 * pv + np.cross(pv, qv)
    return np.array([prod0, *prodv])