import numpy as np

def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Multiplies together 2 quaternions. Scalar Part First!
    
    :param p: Left Quaternion
    :type p: np.ndarray
    :param q: Right Quaternion
    :type q: np.ndarray
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

def quat_conj(p: np.ndarray) -> np.ndarray:
    """
    Returns the conjugate of the quaternion
    
    :param p: Quaternion
    :type p: np.ndarray
    :return: Conjugate of the quaternion
    :rtype: ndarray[Any, Any]
    """
    return np.array([p[0], *(-p[1:4])])


def ypr_to_dcm(ypr: np.ndarray) -> np.ndarray:
    """
    Converts a yaw-pitch-roll sequence into the corresponding DCM
    
    :param ypr: Yaw-pitch-roll sequence [deg]
    :type ypr: np.ndarray
    :return: Corresponding DCM
    :rtype: ndarray
    """
    yaw, pitch, roll = ypr*np.pi/180 # convert to radians
    T_yaw = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    T_pitch = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    T_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0,-np.sin(roll), np.cos(roll)]
    ])

    return T_roll @ T_pitch @ T_yaw

def ypr_to_rotation_quaternion(ypr: np.ndarray) -> np.ndarray:
    """
    Converts a yaw-pitch-roll sequence into the corresponding rotation quaternion
    
    :param ypr: Yaw-pitch-roll sequence [deg]
    :type ypr: np.ndarray 
    :return: Rotation Quaternion
    :rtype: ndarray
    """
    yaw, pitch, roll = ypr*np.pi/180
    q_yaw = np.array([np.cos(yaw/2), 0,0,np.sin(yaw/2)])
    q_pitch = np.array([np.cos(pitch/2), 0,np.sin(pitch/2), 0])
    q_roll = np.array([np.cos(roll/2), np.sin(roll/2), 0, 0])

    return quat_mult(q_yaw, quat_mult(q_pitch, q_roll))

def rotation_quaternion_to_dcm(q_a_to_b__a: np.ndarray) -> np.ndarray:
    """
    Coverts a rotation quaternion (q_a_to_b__a/q_a_to_b__b) into a DCM (T_a_to_b)
    
    :param rotation_quaternion: Rotation quaternion expressed in a-coordinates/b-coordinates (q_a_to_b__a/q_a_to_b__b)
    :type rotation_quaternion: np.ndarray
    :return: The corresponding DCM (T_a_to_b)
    :rtype: ndarray
    """

    q_w, q_x, q_y, q_z = q_a_to_b__a

    T_a_to_b = np.array([
        [q_w**2 + q_x**2 - q_y**2 - q_z**2, 2*(q_w*q_z + q_x*q_y), 2*(q_x*q_z - q_w*q_y)],
        [2*(q_x*q_y - q_w*q_z), q_w**2 - q_x**2 + q_y**2 - q_z**2, 2*(q_w*q_x + q_y*q_z)],
        [2*(q_w*q_y + q_x*q_z), 2*(q_y*q_z - q_w*q_x), q_w**2 - q_x**2 - q_y**2 + q_z**2]
    ])
    
    return T_a_to_b


def main():
    theta = np.pi
    n__a = np.array([0.707,0.707,0])
    q_a_to_b__a = np.array([np.cos(theta/2),*(np.sin(theta/2)*n__a)])
    v__a = np.array([0,0,1])
    v__b = rotation_quaternion_to_dcm(q_a_to_b__a) @ v__a
    print(v__b)

if __name__ == "__main__":
    main()