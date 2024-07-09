import os, os.path as osp


def get_data_src():
    return os.environ["DATA_DIR"]


def get_ndf_data():
    return osp.join(get_data_src())


def get_mfim_src():
    return os.environ["MFIM_DIR"]


def get_mfim_model_src():
    return osp.join(get_mfim_src(), 'scripts/logging')


def get_ndf_demo_src(obj="mug"):
    return osp.join(get_mfim_src(), f"vis_descriptor_field/data/ndf_{obj}")


def get_model_weight_src(obj="mug", model="mfim", depth=False, full=False):
    if depth:
        return osp.join(get_mfim_src(), f"model_weights/{obj}/{obj}_{model}_dep.pth")
    else:
        return osp.join(get_mfim_src(), f"model_weights/{obj}/{obj}_{model}.pth")


def get_gripper_mesh_path():
    return osp.join(get_mfim_src(), "mesh/armar6_right_hand.obj")


def get_demo_mesh_path(obj):
    return osp.join(get_mfim_src(), f"demo/data/recordings/ref_{obj}/demo.obj")


def get_canonical_mesh_path(obj):
    return osp.join(get_mfim_src(), f"demo/data/canonical/{obj}/{obj}.obj")


def set_gmm_weights_save_dir(n_components, task, obj="mug"):
    return osp.join(get_mfim_src(), f"distribution/weights", obj, task, str(n_components))


def get_gmm_weights_save_path(n_component, task, obj="mug"):
    return osp.join(get_mfim_src(), f"distribution/weights", obj, task, str(n_component), "weights.npz")


def get_gmm_data_dir(task, obj="mug"):
    return osp.join(get_mfim_src(), f"distribution/weights", obj, task, "grasps.npz")


def set_gmm_sample_dir(n_components, task, obj="mug"):
    return osp.join(get_mfim_src(), f"distribution/weights", obj, task, str(n_components))


def get_can_mesh_path(obj):
    return osp.join(get_mfim_src(), f"mesh/canonical_{obj}.obj")


def get_grasp_evaluation_path(file_name):
    return os.path.join(get_mfim_src(), "grasp_eval_net/weights", file_name)


def get_ndf_exp_cfg_path():
    return os.path.join(get_mfim_src(), "eval/config", "ndf_exp.yaml")
