import numpy as np
import polyscope as ps
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image

from robot_utils.math.transformations import quaternion_from_matrix, quaternion_matrix
from mfim.utils import common
from mfim.utils.path_util import get_gripper_mesh_path


def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels, out_file=None, show=False):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualise_projection(
        self, points, world_mat, camera_mat, img, output_file='out.png'):
    r''' Visualizes the transformation and projection to image plane.

        The first points of the batch are transformed and projected to the
        respective image. After performing the relevant transformations, the
        visualization is saved in the provided output_file path.

    Arguments:
        points (tensor): batch of point cloud points
        world_mat (tensor): batch of matrices to rotate pc to camera-based
                coordinates
        camera_mat (tensor): batch of camera matrices to project to 2D image
                plane
        img (tensor): tensor of batch GT image files
        output_file (string): where the output should be saved
    '''
    points_transformed = common.transform_points(points, world_mat)
    points_img = common.project_to_camera(points_transformed, camera_mat)
    pimg2 = points_img[0].detach().cpu().numpy()
    image = img[0].cpu().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.plot(
        (pimg2[:, 0] + 1)*image.shape[1]/2,
        (pimg2[:, 1] + 1) * image.shape[2]/2, 'x')
    plt.savefig(output_file)


def ps_plot_grasp_pose(pose, name="g", radius=0.005, length=0.15):
    gripper_path = get_gripper_mesh_path()
    gripper = trimesh.load_mesh(gripper_path, process=False)
    gripper.apply_transform(pose)
    pts = gripper.sample(5000)

    coords = np.array([[length, 0, 0, 0],
                       [0., length, 0, 0],
                       [0, 0, length, 0],
                       [1, 1, 1, 1]])
    coords = pose @ coords
    coords = coords[0:3, :]
    coords = coords.T
    nodes = coords

    ps.register_curve_network(f"edge_x_{name}", nodes[[0, 3]], np.array([[0, 1]]), enabled=True, radius=radius,
                              color=(1, 0, 0))
    ps.register_curve_network(f"edge_y_{name}", nodes[[1, 3]], np.array([[0, 1]]), enabled=True, radius=radius,
                              color=(0, 1, 0))
    ps.register_curve_network(f"edge_z_{name}", nodes[[2, 3]], np.array([[0, 1]]), enabled=True, radius=radius,
                              color=(0, 0, 1))

    ps.register_point_cloud(f"gripper_{name}", pts, enabled=True)


def plot_pcd_trans(pcd, trans_mat, name=""):
    ones = np.ones((pcd.shape[0], 1))
    pcd_ones = np.hstack([pcd, ones])

    pts = (trans_mat @ pcd_ones.T)[:3, :].T

    ps.register_point_cloud(f"pcd_{name}", pts, enabled=True)


def plot_trajetory(trajectory, name=""):
    for i in range(trajectory.shape[0]):
        coords = np.array([[0.05, 0, 0, 0],
                           [0., 0.05, 0, 0],
                           [0, 0, 0.05, 0],
                           [1, 1, 1, 1]])

        pose = trajectory[i, :]

        mat = quaternion_matrix(pose[3:])
        mat[:3, -1] = pose[:3]
        coords = mat @ coords
        coords = coords[0:3, :]
        coords = coords.T
        nodes = coords

        ps.register_curve_network(f"edge_x_{name}_{i}", nodes[[0, 3]], np.array([[0, 1]]), enabled=True, radius=0.005,
                                  color=(1, 0, 0))
        ps.register_curve_network(f"edge_y_{name}_{i}", nodes[[1, 3]], np.array([[0, 1]]), enabled=True, radius=0.005,
                                  color=(0, 1, 0))
        ps.register_curve_network(f"edge_z_{name}_{i}", nodes[[2, 3]], np.array([[0, 1]]), enabled=True, radius=0.005,
                                  color=(0, 0, 1))

