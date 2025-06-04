import sys
sys.path.append('.')

import argparse
import os
import time
import tempfile
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
import trimesh
import sapien
from sapien.utils import Viewer
from manotorch.manolayer import ready_arguments
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.nn.parallel import DataParallel as DP
from trimesh import Trimesh

# 先导入模型和转换器，确保它们被注册
from lib.models import *
from lib.models.transforms import *
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils.config import CN
from lib.utils.logger import logger
from lib.viztools.utils import ColorsMap as CMap
from lib.viztools.viz_o3d_utils import VizContext
from pytransform3d import rotations

from dex_retargeting.constants import (
    HandType, 
    RobotName, 
    RetargetingType,
    get_default_config_path
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting import yourdfpy as urdf

from scripts.simple_retarget import SimpleRetargeter

# 最后导入 builder，因为它依赖于上面的导入
from lib.utils import builder


GrabNetConfig = dict(
    DATA_PRESET=dict(
        CENTER_IDX=9,
        N_RESAMPLED_OBJ_POINTS=4096,
    ),
    MODEL=dict(
        TYPE='GrabNet',
        COARSE_NET=dict(
            TYPE='CoarseNet',
            LATENTD=16,
            KL_COEF=0.005,
            VPE_PATH='assets/GrabNet/verts_per_edge.npy',
            C_WEIGHT_PATH='assets/GrabNet/rhand_weight.npy',
            PRETRAINED='checkpoints/grabnet_oishape/CoarseNet.pth.tar',
        ),
        REFINE_NET=dict(
            TYPE='RefineNet',
            KL_COEF=0.005,
            VPE_PATH='assets/GrabNet/verts_per_edge.npy',
            C_WEIGHT_PATH='assets/GrabNet/rhand_weight.npy',
            PRETRAINED='checkpoints/grabnet_oishape/RefineNet.pth.tar',
        ),
    ),
    TRANSFORM=dict(
        TYPE="GrabNetTransformObject",
        RAND_ROT=False,
        USE_ORIGINAL_OBJ_ROT=True,
        BPS_BASIS_PATH="assets/GrabNet/bps.npz",
        BPS_FEAT_TYPE="dists",
    ),
)


def load_obj_models(obj_path: str, n_sample_verts=10000, rescale=False):
    obj_trimesh: Trimesh = trimesh.load(obj_path, process=False)
    obj_verts = np.asarray(obj_trimesh.vertices, dtype=np.float32)
    obj_faces = np.asarray(obj_trimesh.faces, dtype=np.int32)

    # 计算物体的中心点，用于后续坐标系转换
    maximum = obj_verts.max(0, keepdims=True)
    minimum = obj_verts.min(0, keepdims=True)
    obj_center = (maximum + minimum).squeeze(0) / 2  # 确保形状是 (3,)
    obj_verts = obj_verts - obj_center  # 将物体居中

    if rescale:
        scale = (obj_verts.max() - obj_verts.min()) / 2
        obj_verts = obj_verts / scale
        obj_verts = obj_verts * 0.1

    obj_rotmat = np.eye(3, dtype=np.float32)

    mesh = Meshes(verts=torch.from_numpy(obj_verts).unsqueeze(0), faces=torch.from_numpy(obj_faces).unsqueeze(0))
    obj_verts_ds, obj_normals_ds = sample_points_from_meshes(mesh, n_sample_verts, return_normals=True)

    res = {
        Queries.SAMPLE_IDENTIFIER: obj_path,
        Queries.OBJ_ID: obj_path,
        Queries.OBJ_VERTS_OBJ: torch.from_numpy(obj_verts),
        Queries.OBJ_FACES: torch.from_numpy(obj_faces),
        Queries.OBJ_ROTMAT: torch.from_numpy(obj_rotmat),
        Queries.OBJ_VERTS_OBJ_DS: obj_verts_ds.squeeze(0),
        Queries.OBJ_NORMALS_OBJ_DS: obj_normals_ds.squeeze(0),
        'obj_center': obj_center,  # 保存物体中心点
        'obj_scale': scale if rescale else 1.0,  # 保存缩放因子
    }

    return res


def compute_smooth_shading_normal_np(vertices, indices):
    """计算顶点法线"""
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


def grasp_new_obj(arg: Namespace, exp_time):
    rank = 0
    cfg = CN(GrabNetConfig)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET)
    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)

    # 初始化 SAPIEN 场景
    scene = sapien.Scene()
    scene.set_timestep(1 / 240.0)

    # 设置光照
    scene.set_environment_map(
        sapien.asset.create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_directional_light([1, -1, -1], [2, 2, 2], shadow=True)
    scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
    scene.set_ambient_light([0.2, 0.2, 0.2])

    # 添加相机
    camera = scene.add_camera(
        name='camera',
        width=1920,
        height=1080,
        fovy=np.deg2rad(35),
        near=0.1,
        far=100,
    )
    camera.set_local_pose(sapien.Pose([1.5, 0, 1.0], [0, 0.389418, 0, -0.921061]))

    # 加载机器人手
    config_path = get_default_config_path(
        RobotName[arg.robots], 
        RetargetingType.position, 
        HandType[arg.hand_type]
    )
    config = RetargetingConfig.load_from_file(config_path, override={'add_dummy_free_joint': True})
    retargeting = config.build()
    
    # 加载 URDF
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = Path(config.urdf_path)
    robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
    temp_dir = tempfile.mkdtemp(prefix='dex_retargeting-')
    temp_path = f'{temp_dir}/{urdf_path.name}'
    robot_urdf.write_xml_file(temp_path)
    robot = loader.load(temp_path)

    # 获取机器人关节限制
    robot_joint_limits = []
    for joint in robot.get_active_joints():
        limits = joint.get_limits()
        if limits is None:
            robot_joint_limits.append((-np.pi, np.pi))
        else:
            limits = np.array(limits)
            robot_joint_limits.append((float(limits.min()), float(limits.max())))
    robot_joint_limits = np.array(robot_joint_limits)

    # 初始化 GrabNet 模型
    model = DP(model).to(device=rank)
    transform = DP(transform).to(device=rank)
    model.eval()

    # 加载物体
    obj_data = load_obj_models(arg.obj_path, rescale=arg.rescale)
    obj_center = obj_data['obj_center']  # 获取物体中心点
    
    for k, v in obj_data.items():
        if isinstance(v, torch.Tensor):
            obj_data[k] = v.unsqueeze(0).to(rank)
        if isinstance(v, np.ndarray):
            obj_data[k] = torch.from_numpy(v).unsqueeze(0).to(rank)

    # 创建物体网格
    obj_verts = obj_data[Queries.OBJ_VERTS_OBJ][0].detach().cpu().numpy()
    obj_faces = obj_data[Queries.OBJ_FACES][0].detach().cpu().numpy()
    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
    
    # 设置物体材质
    obj_material = sapien.render.RenderMaterial()
    obj_material.set_base_color(np.array([0.2, 0.8, 0.2, 1]))  # 绿色
    obj_material.set_roughness(0.7)
    obj_material.set_metallic(0.0)
    obj_material.set_specular(0.04)
    
    # 保存临时文件
    temp_dir = tempfile.mkdtemp(prefix='obj-')
    temp_obj_path = os.path.join(temp_dir, 'object.obj')
    obj_mesh.export(temp_obj_path)
    
    # 从文件加载物体，并设置到点云中心位置
    obj_actor = scene.create_actor_builder().add_visual_from_file(temp_obj_path, material=obj_material).build_static()
    obj_actor.set_pose(sapien.Pose(obj_center))  # 物体放在点云中心

    # 创建查看器
    viewer = Viewer()
    viewer.set_scene(scene)
    # 从更远的地方看原点
    viewer.set_camera_xyz(0, 0.5, 0.1)  # 相机位置：在后方更远，高度适中
    viewer.set_camera_rpy(0, -0.3, 1.57)  # 相机朝向：往前看，稍微往下
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=0.9)  # 视场角
    viewer.control_window.toggle_origin_frame(False)

    print('正在生成抓取姿势...')
    print('按空格键继续下一个抓取姿势，按 ESC 退出')

    # 创建 MANO 手的材质
    mat_hand = sapien.render.RenderMaterial()
    mat_hand.set_base_color(np.array([0.96, 0.75, 0.69, 1]))  # 肉色
    mat_hand.set_roughness(0.8)
    mat_hand.set_metallic(0.0)
    mat_hand.set_specular(0.04)

    # 存储 MANO 手的 actor
    mano_actor = None
    temp_dir = tempfile.mkdtemp(prefix='mano-')

    # 获取标准MANO面片
    mano_rhand_path = os.path.join(arg.mano_path, "models", "MANO_RIGHT.pkl")
    mano_data = ready_arguments(mano_rhand_path)
    mano_faces = np.array(mano_data["f"]).astype(np.int32)

    # update_mano_hand 只用hand_verts和标准faces渲染
    def update_mano_hand(hand_verts, idx):
        nonlocal mano_actor
        if mano_actor is not None:
            scene.remove_actor(mano_actor)
        temp_path = os.path.join(temp_dir, f'mano_{idx}.obj')
        temp_mesh = trimesh.Trimesh(vertices=hand_verts, faces=mano_faces)
        temp_mesh.export(temp_path)
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(temp_path, material=mat_hand)
        mano_actor = builder.build_static(name=f"mano_{idx}")

    try:
        for i in range(arg.n_grasps):
            obj_data = transform(obj_data)
            prd, _ = model(inp=obj_data, step_idx=0, mode='test')
            mano_joints = prd['Refine.joints_rhand'][0].detach().cpu().numpy()  # (21, 3)
            mano_pose = torch.cat([
                prd['Refine.global_orient'][0],
                prd['Refine.hand_pose'][0]
            ]).detach().cpu().numpy()  # (48,)
            mano_transl = prd['Refine.transl'][0].detach().cpu().numpy()  # (3,)
            hand_verts = prd['Refine.hand_verts'][0].detach().cpu().numpy()

            # 直接用hand_verts渲染
            update_mano_hand(hand_verts, i)

            # wrist_quat = Rotation.from_rotvec(mano_pose[:3]).as_quat()
            wrist_quat = rotations.quaternion_from_compact_axis_angle(
                mano_pose[0:3])
            
            # 先进行 warm_start 初始化机器人手的位姿
            retargeting.warm_start(
                mano_transl,  # 使用 MANO 手的关节位置
                wrist_quat,         # 使用 MANO 手的全局旋转
                hand_type=HandType[arg.hand_type],
                is_mano_convention=True
            )
            
            # 然后进行重映射
            indices = retargeting.optimizer.target_link_human_indices
            ref_value = mano_joints[indices, :]
            fixed_qpos = np.zeros(arg.fixed_joints_num)
            
            # 获取关节角度并设置
            qpos = retargeting.retarget(ref_value, fixed_qpos)
            qpos = np.clip(qpos, robot_joint_limits[:, 0], robot_joint_limits[:, 1])
            robot.set_qpos(qpos)
            
            # 设置机器人的全局位姿
            robot.set_pose(sapien.Pose(mano_transl, wrist_quat))

            scene.step()
            scene.update_render()
            viewer.render()

            space_pressed = False
            while not space_pressed:
                viewer.render()
                if viewer.window.key_down('escape'):
                    viewer.close()
                    return
                elif viewer.window.key_down('space'):
                    space_pressed = True
                    break
            while viewer.window.key_down('space'):
                viewer.render()

    finally:
        # 清理 MANO 手的 actor 和临时文件
        if mano_actor is not None:
            scene.remove_actor(mano_actor)
        import shutil
        shutil.rmtree(temp_dir)

    viewer.close()

    if arg.save:
        ts = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(exp_time))
        save_path = os.path.join("user", "grasps", ts)
        os.makedirs(save_path, exist_ok=True)
        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        obj_mesh.export(os.path.join(save_path, "object.obj"))


if __name__ == '__main__':
    exp_time = time.time()
    arg, _ = parse_exp_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    # 设置 URDF 目录
    robot_dir = Path(__file__).parent.parent / 'assets' / 'robots' / 'hands'
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--obj_path", type=str, required=True, help='The path to the 3D object Mesh')
    parser.add_argument("--mano_path", type=str, default="assets/mano_v1_2", help='The path to MANO models')
    parser.add_argument("--n_grasps", type=int, required=False, default=1, help='how many grasps to generate')
    parser.add_argument("--rescale",
                        action="store_true",
                        default=False,
                        help='rescale the object to fit in radius=0.1m sphere')
    parser.add_argument("--save", action="store_true", default=False, help='save the grasps to file')
    parser.add_argument("--robots", type=str, default="ALLEGRO", help='robot hand name for retargeting (e.g. ALLEGRO, SHADOW, etc.)')
    parser.add_argument("--hand_type", type=str, default="left", help='robot hand type for retargeting (e.g. left, right, etc.)')
    parser.add_argument("--fixed_joints_num", type=int, default=2, help='fixed joints number for retargeting')

    arg_extra, _ = parser.parse_known_args()
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    grasp_new_obj(arg, exp_time)