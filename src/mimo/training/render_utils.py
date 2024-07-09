import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pickle
import pyvista as pv
import numpy as np
from trimesh import transformations

def euler2quat(euler, axes='xyz'):
    r = R.from_euler(axes, euler)
    return r.as_quat()

def depth_to_point_cloud(depth,cam_in,mask=None):
    if len(cam_in.shape)==1:
        fx,fy,cx,cy = cam_in
    else:
        fx = cam_in[0,0]
        fy = cam_in[1,1]
        cx = cam_in[0,2]
        cy = cam_in[1,2]
    h,w = depth.shape
    ij = np.mgrid[0:h,0:w].transpose((1,2,0))
    xyz = np.zeros((h,w,3))
    xyz[...,0] = (ij[...,1]-cx)*depth/fx
    xyz[...,1] = (ij[...,0]-cy)*depth/fy
    xyz[...,2] = depth
    if mask is not None:
        xyz = xyz[mask]
    return xyz.reshape(-1,3)

def matrix_from_list(pose_list):
    trans = pose_list[0:3]
    quat = pose_list[3:7]
    T = np.zeros((4, 4))
    T[-1, -1] = 1
    r = R.from_quat(quat)
    T[:3, :3] = r.as_matrix()
    T[0:3, 3] = trans
    return T


def render(obj_class,shapenet_id,scale,pose):
    script_dir = f'/home/zeyu/dataset/NDF/scripts/mesh'
    cam_ex = np.array([[[ 9.39692584e-01,  1.44543973e-01, -3.09975584e-01,
            7.78978012e-01],
            [ 3.42020186e-01, -3.97131278e-01,  8.51650711e-01,
            -7.66485618e-01],
            [-7.65259864e-09, -9.06307804e-01, -4.22618281e-01,
            1.48035655e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]],

        [[-9.39692517e-01,  1.44543997e-01, -3.09975573e-01,
            7.78978044e-01],
            [ 3.42020176e-01,  3.97131236e-01, -8.51650708e-01,
            7.66485673e-01],
            [-2.11811607e-08, -9.06307739e-01, -4.22618294e-01,
            1.48035641e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]],

        [[-8.66025326e-01, -2.11309141e-01,  4.53153922e-01,
            9.21615170e-02],
            [-4.99999998e-01,  3.65998144e-01, -7.84885562e-01,
            7.06396925e-01],
            [-2.29993820e-10, -9.06307739e-01, -4.22618294e-01,
            1.48035642e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]],

        [[ 8.66025331e-01, -2.11309227e-01,  4.53154075e-01,
            9.21613871e-02],
            [-5.00000181e-01, -3.65998129e-01,  7.84885478e-01,
            -7.06396896e-01],
            [-1.91468588e-08, -9.06307810e-01, -4.22618268e-01,
            1.48035647e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]]])
    focal_point =  [
        [0.5, 0.0, 1.1], 
        [0.5, 0.0, 1.1], 
        [0.5, 0.0, 1.1], 
        [0.5, 0.0, 1.1]]
    cam_in = np.array([415.69219382,415.69219382,320,240])
    yaw =  [20, 160, 210, 330]
    pitch = [-25.0, -25.0, -25.0, -25.0]
    roll =  [0, 0, 0, 0]
    

    T = matrix_from_list(pose)
    mesh = pickle.load(open(f'{script_dir}/{obj_class}/{shapenet_id}.pkl','rb')) 
    mesh.apply_scale(scale)
    mesh.apply_transform(T)

    pcds = []
    for i in range(4):
        p = pv.Plotter(off_screen=True,window_size=(640,480))
        p.add_mesh(mesh,(255,0,0),lighting=False,reset_camera=False)
        p.camera.clipping_range = (0.01,10)
        p.camera.view_angle = 60.0
        p.camera.up = (0,0,1)
        p.camera.position = cam_ex[i][:3,3]
        p.camera.focal_point = focal_point[i]
        p.background_color = (0,0,0)
        p.store_image = True
        rgb = p.screenshot(transparent_background=True,return_img=True)
        rgb = p.image
        depth = np.abs(p.get_image_depth())
        pcd = depth_to_point_cloud(depth,cam_in,rgb[...,0]==255)
        pcd = transformations.transform_points(pcd,cam_ex[i])
        pcds.append(pcd)
        p.close()
    return pcds

def render_both(idx,obj_class,data):
    data_dir = f'/home/zeyu/dataset/NDF/data/training_data'
    if os.path.exists(f'{data_dir}/pcds/{obj_class}/{idx}.pkl'): 
        pcds = pickle.load(open(f'{data_dir}/pcds/{obj_class}/{idx}.pkl','rb'))
        return pcds
    shapenet_id = data['shapenet_id']
    scale_up = data['up_scale']
    pose_up = data['up_pose']
    pcds_up = render(obj_class,shapenet_id,scale_up,pose_up)

    scale_any = data['any_scale']
    pose_any = data['any_pose']
    pcds_any = render(obj_class,shapenet_id,scale_any,pose_any)

    pcds = {
        'obj_class': obj_class,
        'shapenet_id': shapenet_id,
        'scale_up': scale_up,
        'pose_up': pose_up,
        'pcds_up': pcds_up,
        'scale_any': scale_any,
        'pose_any': pose_any,
        'pcds_any': pcds_any
    }
    pickle.dump(pcds,open(f'{data_dir}/pcds/{obj_class}/{idx}.pkl','wb'))
    return pcds
