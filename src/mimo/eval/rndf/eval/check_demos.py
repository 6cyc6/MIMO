import os
import os.path as osp
import numpy as np
import polyscope as ps

from rndf_robot.utils import path_util

# demo_path = osp.join(path_util.get_rndf_data(), 'relation_demos', 'release_demos/bottle_in_container_relation')
demo_path = osp.join(path_util.get_rndf_data(), 'relation_demos', 'release_demos/bowl_on_mug_relation')
demo_files = [fn for fn in sorted(os.listdir(demo_path)) if fn.endswith('.npz') and 'recon' in fn]
demos = []
for f in demo_files:
    demo = np.load(demo_path + '/' + f, allow_pickle=True)
    demos.append(demo)

for idx, demo in enumerate(demos):
    print(idx)

    # pcd_s = demo["multi_obj_start_pcd"]
    # pcd_e = demo["multi_obj_final_pcd"]
    #
    # pcd_parent_s = pcd_s.item()["parent"]
    # pcd_child_s = pcd_s.item()["child"]
    #
    # pcd_parent_e = pcd_e.item()["parent"]
    # pcd_child_e = pcd_e.item()["child"]

    pcd_parent_e = demo["pcd_parent"]
    pcd_child_e = demo["pcd_child"]

    ps.init()
    ps.set_up_dir("z_up")

    ps.register_point_cloud("obj1", pcd_parent_e)
    ps.register_point_cloud("obj2", pcd_child_e)

    ps.show()
