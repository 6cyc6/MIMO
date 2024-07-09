import math

import numpy as np
# import pymanopt.manifolds.sphere as sp
import open3d as o3d

from scipy.spatial.transform import Rotation as R

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def quat2eulers(q0: float, q1: float, q2: float, q3: float) -> tuple:
    """
    Compute yaw-pitch-roll Euler angles from a quaternion.

    Args
    ----
        q0: Scalar component of quaternion.
        q1, q2, q3: Vector components of quaternion.

    Returns
    -------
        (roll, pitch, yaw) (tuple): 321 Euler angles in radians
    """
    roll = math.atan2(
        2 * ((q2 * q3) + (q0 * q1)),
        q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2
    )  # radians
    pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
    yaw = math.atan2(
        2 * ((q1 * q2) + (q0 * q3)),
        q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
    )
    return roll, pitch, yaw


def matrix_from_pose(pos, quat) -> np.ndarray:
    """
    homogeneous transformation matrix from transformation and quaternion
    :param pos: (3, ) numpy array
    :param quat: (4, ) numpy array
    :return: (4, 4) numpy array
    """
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(quat)
    pos = pos.reshape((1, 3))
    t[0:3, 3] = pos

    return t


def pose_from_matrix(t, stack=False):
    """
    transformation and quaternion from homogeneous transformation matrix
    :param stack: if stack pose and quaternion
    :param t: 4x4 numpy array
    :return: pos: (3, ) numpy array
             quat: (4, ) numpy array
    """
    pos = t[0:3, -1]
    r_mat = t[0:3, 0:3]
    quat = mat_to_quat(r_mat)

    if stack:
        return np.hstack([pos, quat])
    else:
        return pos, quat


def get_transform(pose_target, pose_origin) -> np.ndarray:
    """
    Find transformation from original pose to target pose
    :param pose_target: (4, 4) numpy array
    :param pose_origin: (4, 4) numpy array
    :return pose: (7, ) numpy array
    """
    t = pose_target @ np.linalg.inv(pose_origin)
    pos, quat = pose_from_matrix(t)
    pose = np.hstack([pos, quat])

    return pose


def quat_to_mat(quat) -> np.ndarray:
    # first change the order to use scipy package: scalar-last (x, y, z, w) format
    # id_ord = [1, 2, 3, 0]
    # quat = quat[id_ord]
    r = R.from_quat(quat)

    return r.as_matrix()


def mat_to_quat(mat) -> np.ndarray:
    r = R.from_matrix(mat)
    quat = r.as_quat()
    # change order
    # id_ord = [3, 0, 1, 2]

    return quat
    # return quat[id_ord]


def quat2mat(quat) -> np.ndarray:
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)

    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def mat2euler(mat):
    """Convert Rotation Matrix to Euler Angles.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def quat2euler(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    return mat2euler(quat2mat(quat))


def euler2mat(euler):
    r = R.from_euler('xyz', angles=[euler[0], euler[1], euler[2]], degrees=False)

    return r.as_matrix()


def mat_to_euler(mat):
    if mat.shape[0] == 4:
        mat = mat[:3, :3]
    r = R.from_matrix(mat)
    euler = r.as_euler('xyz', degrees=False)

    return euler[0], euler[1], euler[2]


def add_random_approach(grasps, n, d=0.03, h=0.01, r=0.09) -> np.ndarray:
    """
    add randomness to original poses
    :param h:
    :param r:
    :param d:
    :param grasps: the original pose
    :param n: num of generated poses for each original pose
    :return: generated poses
    """
    num_grasps = grasps.shape[0]
    manifold = sp.Sphere(4)

    # trans_mat = [np.array([[1, 0, 0, 0],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 1, np.random.rand() * d * 2 - d],
    #                       [0, 0, 0, 1]]) for i in range(n * num_grasps)]
    grasps_mat = [matrix_from_pose(grasps[i, :3], grasps[i, 3:]) for i in range(num_grasps)]
    # pose = [(grasps_mat[i] @ trans_mat[i])[:3, -1][np.newaxis, :] for i in range(n * num_grasps)]
    # pos = np.concatenate(pose[:])
    pos = np.vstack(
        [np.vstack([(grasps_mat[j] @ np.array([[1, 0, 0, np.random.uniform(low=-1, high=1) * h],
                                               [0, 1, 0, np.random.uniform(low=-1, high=1) * h],
                                               [0, 0, 1, np.random.uniform(low=-1, high=0.25) * d],
                                               [0, 0, 0, 1]]))[:3, -1][np.newaxis, :]
                    for i in range(n)])
         for j in range(num_grasps)])

    quat = np.vstack(
        [np.vstack([manifold.exp(grasps[j, 3:],
                                 manifold.random_tangent_vector(point=grasps[j, 3:]) * np.random.rand(1) * r)
                    for i in range(n)])
         for j in range(num_grasps)])
    pose = np.hstack([pos, quat])

    n_grasps = pose.shape[0]
    print("Get " + str(n_grasps) + " grasps.")

    return pose


def add_random_approach_side(grasps, n, d=0.015, h=0.03, r=0.09) -> np.ndarray:
    """
    add randomness to original poses
    :param h:
    :param r:
    :param d:
    :param grasps: the original pose
    :param n: num of generated poses for each original pose
    :return: generated poses
    """
    num_grasps = grasps.shape[0]
    manifold = sp.Sphere(4)

    # trans_mat = [np.array([[1, 0, 0, 0],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 1, np.random.rand() * d * 2 - d],
    #                       [0, 0, 0, 1]]) for i in range(n * num_grasps)]
    grasps_mat = [matrix_from_pose(grasps[i, :3], grasps[i, 3:]) for i in range(num_grasps)]
    # pose = [(grasps_mat[i] @ trans_mat[i])[:3, -1][np.newaxis, :] for i in range(n * num_grasps)]
    # pos = np.concatenate(pose[:])
    pos = np.vstack(
        # [np.vstack([(grasps_mat[j] @ np.array([[1, 0, 0, 0],
        #                                       [0, 1, 0, np.random.rand() * h * 2 - h],
        #                                       [0, 0, 1, -np.random.rand() * d * 2 + 0.5 * d],
        #                                       [0, 0, 0, 1]]))[:3, -1][np.newaxis, :]
        [np.vstack([(grasps_mat[j] @ np.array([[1, 0, 0, 0],
                                               [0, 1, 0, np.random.uniform(low=-0.5, high=1) * h],
                                               [0, 0, 1, np.random.uniform(low=-1, high=0.3) * d],
                                               [0, 0, 0, 1]]))[:3, -1][np.newaxis, :]
                    for i in range(n)])
         for j in range(num_grasps)])

    quat = np.vstack(
        [np.vstack([manifold.exp(grasps[j, 3:],
                                 manifold.random_tangent_vector(point=grasps[j, 3:]) * np.random.rand(1) * r)
                    for i in range(n)])
         for j in range(num_grasps)])
    pose = np.hstack([pos, quat])

    n_grasps = pose.shape[0]
    print("Get " + str(n_grasps) + " grasps.")

    return pose


def add_random_pose(grasps, n, d=0.015, r=0.18) -> np.ndarray:
    """
    add randomness to original poses
    :param r:
    :param d:
    :param grasps: the original pose
    :param n: num of generated poses for each original pose
    :return: generated poses
    """
    num_grasps = grasps.shape[0]
    manifold = sp.Sphere(4)

    pos = np.vstack([grasps[i, :3] + np.random.rand(n, 3) * d * 2 - np.array([d, d, d]) for i in range(num_grasps)])
    # pos = np.vstack([grasps[i, :3] - np.random.rand(n, 3) * d * 2 for i in range(num_grasps)])
    quat = np.vstack(
        [np.vstack([manifold.exp(grasps[j, 3:],
                                 manifold.random_tangent_vector(point=grasps[j, 3:]) * np.random.rand(1) * r)
                    for i in range(n)])
         for j in range(num_grasps)])
    pose = np.hstack([pos, quat])

    n_grasps = pose.shape[0]
    print("Get " + str(n_grasps) + " grasps.")

    return pose


def square_distance(pts):
    """
    Calculate Euclid distance between each two points.
    Param:
        pts: source points, (n, 3)
    Return:
        dist: per-point square distance, (n, n)
    """
    dist = -2 * pts @ pts.T
    a = np.sum(pts ** 2, axis=-1, keepdims=True)
    b = np.sum(pts.T ** 2, axis=0, keepdims=True)
    dist = dist + a + b
    return dist


def fit_plane(pcd):
    """
    Fit a plane in the point cloud.
    Param:
        pcd: points, (n, 3)
    Return:
        plane_model: (4, ) (plane function: ax + by + cz + d = 0)
        pcd_plane: points on the plane (n_inliers, 3)
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    plane_model, inliers = pcd_o3d.segment_plane(distance_threshold=0.01,
                                                 ransac_n=100,
                                                 num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    pcd_plane = pcd[inliers]

    return plane_model, pcd_plane


def get_place_pos(pcd):
    """
        Choose a point for placement given a set of points on a plane.
        Param:
            pcd: points, (n, 3)
        Return:
            pos: placement point (3, )
    """
    np.random.shuffle(pcd)

    # randomly choose some points
    n_pts = min(pcd.shape[0], 5000)
    pcd = pcd[:n_pts]

    # calculate the distance and choose points with most neighbors within the threshold
    dist = square_distance(pcd)
    radius = 0.07
    n_filter_1 = int(0.8 * n_pts)
    idx_sorted = np.argsort(np.sum(np.where(dist < radius * radius, 1, 0), axis=-1))[-n_filter_1:]
    # place_pos = plane_pcd[idx]
    pcd_sorted = pcd[idx_sorted]

    radius = 0.3
    n_filter_2 = int(0.2 * n_pts)
    dist = square_distance(pcd_sorted)
    idx_sorted = np.argsort(np.sum(np.where(dist < radius * radius, 1, 0), axis=-1))[-n_filter_2:]
    # pcd_sorted = pcd_sorted[idx_sorted]
    idx = np.random.randint(0, n_filter_2, 1)

    pos = pcd_sorted[idx_sorted[idx]]

    return pos


def get_trans_plane(source_pose, plane_model, target_pos):
    """
        Calculate the transformation from source pose to target plane pose
        Param:
            source_pose: (4, 4) homogeneous matrix
            plane_model: (4, ) plane function
            target_pos: (3, ) target position
        Return:
            t: transformation matrix (4, 4)
    """
    # get source normal vector and the position of the frame
    z0 = np.array([0, 0, 0])
    z1 = np.array([0, 0, 1])
    vec_source = source_pose[:3, :3] @ z1 - source_pose[:3, :3] @ z0
    pos_source = source_pose[:3, -1]

    # get target vector
    vec_target = np.array([plane_model[0], plane_model[1], plane_model[2]])

    # unit vectors
    vec_source = vec_source / np.linalg.norm(vec_source)
    vec_target = vec_target / np.linalg.norm(vec_target)

    # calculate the rotation
    # dimension of the space and identity
    I = np.identity(3)
    # the cos angle between the vectors
    c = np.dot(vec_source, vec_target)
    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        rot = I
    elif np.abs(c + 1.0) < eps:
        rot = -I
    else:
        # the cross product matrix of a vector to rotate around
        K = np.outer(vec_target, vec_source) - np.outer(vec_source, vec_target)
        # Rodrigues' formula
        rot = I + K + (K @ K) / (1 + c)

    t_target = np.eye(4)
    t_target[:3, :3] = rot @ source_pose[:3, :3]
    t_target[:3, -1] = target_pos

    t = t_target @ np.linalg.inv(source_pose)

    return t
