from abc import abstractmethod
from typing import List, Optional, Union

import nlopt
import numpy as np
import torch

from dex_retargeting.kinematics_adaptor import (
    KinematicAdaptor,
    MimicJointKinematicAdaptor,
)
from dex_retargeting.robot_wrapper import RobotWrapper


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )

        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1) #计算指尖向量的欧式距离
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True # 如果指尖向量距离小于project_dist，则进行投影
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False # 如果指尖向量距离大于escape_dist，则不进行投影
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        ) # 如果指尖向量距离小于project_dist，则进行投影
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        ) # 再进一步判断指尖向量是否投影 target_vec_dist[len_s1:len_proj]是人手对应的目标向量

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        ) # 拼接了一段长度为手指数量 数值为len_proj+手指数量 的向量

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3) or (10, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3) or (10, 3)

        # Compute final reference vector
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3) or (10, 3)
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3) or (15, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :] # 获取origin_link_indices对应的link位置
            task_link_pos = torch_body_pos[self.task_link_indices, :] # 获取task_link_indices对应的link位置
            robot_vec = task_link_pos - origin_link_pos # 计算task_link_pos和origin_link_pos之间的向量

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            print("=== 向量距离检查 ===")
            for i, (origin_name, task_name, dist) in enumerate(zip(self.origin_link_names, self.task_link_names, vec_dist)):
                print(f"{origin_name} -> {task_name}: {dist.item():.4f}")

            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()

            result = huber_distance.cpu().detach().item()
            print("最终损失:", huber_distance)
            
            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class HybridOptimizer(Optimizer):
    """结合 Position 和 DexPilot 的混合优化器
    
    特点：
    1. 同时优化指尖位置和向量关系
    2. 保留 DexPilot 的投影机制
    3. 使用动态权重平衡两种目标
    """
    retargeting_type = "HYBRID"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],  # 用于位置损失的所有连杆
        finger_tip_link_names: List[str],  # 用于向量损失的指尖连杆
        target_link_human_indices: np.ndarray,
        wrist_link_name: str,
        target_link_human_indices_vec: Optional[np.ndarray] = None,  # 用于dex损失的人类关节索引
        # 优化参数
        huber_delta_pos=0.02,
        huber_delta_vec=0.03,
        norm_delta=4e-3,
        # DexPilot 参数
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
        # 混合权重
        position_weight=0.5,    # 位置损失权重
        vector_weight=0.5,      # 向量损失权重
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"HYBRID optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)
        
        origin_link_index, task_link_index = DexPilotOptimizer.generate_link_indices(
            self.num_fingers
        )

        if target_link_human_indices_vec is None:
            # 生成 DexPilot 风格的索引
            self.target_link_human_indices_vec = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]
        
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.target_link_indices_pos = self.get_link_indices(target_link_names)

        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss_pos = torch.nn.SmoothL1Loss(beta=huber_delta_pos)
        self.huber_loss_vec = torch.nn.SmoothL1Loss(beta=huber_delta_vec, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2
        self.position_weight = position_weight
        self.vector_weight = vector_weight
        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = DexPilotOptimizer.set_dexpilot_cache(self.num_fingers, eta1, eta2)


    def get_objective_function(self, target_data: dict, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        """目标函数，严格模仿 PositionOptimizer 和 DexPilotOptimizer 的写法"""
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        
        # 准备目标数据
        torch_target_pos = torch.as_tensor(target_data['position'])
        target_vec = torch.as_tensor(target_data['vector'])  # 这个就是人手关节位置算出来的向量


        # DexPilot 的向量计算逻辑
        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # 更新投影状态
        target_vec_dist = np.linalg.norm(target_vec[:len_proj].cpu().numpy(), axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # 更新权重向量
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        ) # 拼接了一段长度为手指数量 数值为len_proj+手指数量 的向量

        # 计算参考向量
        normal_vec = target_vec * self.scaling
        dir_vec = target_vec[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3) or (10, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3) or (10, 3)

        # Compute final reference vector
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3) or (10, 3)
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3) or (15, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)
        torch_target_pos.requires_grad_(False)


        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # 前向运动学
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]
            self.robot.compute_forward_kinematics(qpos)
            # ----------- 1. Position Loss-----------
            target_link_poses_pos = [
                self.robot.get_link_pose(index) for index in self.target_link_indices_pos]
            body_pos_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses_pos], axis=0
            )
            torch_body_pos_pos = torch.as_tensor(body_pos_pos)
            torch_body_pos_pos.requires_grad_()
            # Loss term for kinematics retargeting based on 3D position error
            huber_distance_pos = self.huber_loss_pos(torch_body_pos_pos, torch_target_pos)

            # ----------- 2. Dexpolit Loss-----------
            target_link_poses_vec = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices]
            body_pos_vec = np.array([pose[:3, 3] for pose in target_link_poses_vec])
            torch_body_pos_vec = torch.as_tensor(body_pos_vec)
            torch_body_pos_vec.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos_vec[self.origin_link_indices, :] # 获取origin_link_indices对应的link位置
            task_link_pos = torch_body_pos_vec[self.task_link_indices, :] # 获取task_link_indices对应的link位置
            robot_vec = task_link_pos - origin_link_pos # 计算task_link_pos和origin_link_pos之间的向量

            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance_vec = (
                self.huber_loss_vec(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            huber_distance_vec = huber_distance_vec.sum()

            # ----------- 3. 总损失加权 -----------
            total_loss = self.position_weight * huber_distance_pos + self.vector_weight * huber_distance_vec
            # 平滑项
            result = total_loss.cpu().detach().item()

            # ----------- 4. 梯度 -----------
            if grad.size > 0:
                jacobians_pos = []
                for i, index in enumerate(self.target_link_indices_pos):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses_pos[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians_pos.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians_pos= np.stack(jacobians_pos, axis=0)
                huber_distance_pos.backward()
                grad_pos_pos= torch_body_pos_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians_pos = self.adaptor.backward_jacobian(jacobians_pos)
                else:
                    jacobians_pos = jacobians_pos[..., self.idx_pin2target]

                # dexpolit损失梯度
                jacobians_vec = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index)[:3, ...]                    
                    link_pose = target_link_poses_vec[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians_vec.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians_vec = np.stack(jacobians_vec, axis=0)
                huber_distance_vec.backward()
                grad_pos_vec = torch_body_pos_vec.grad.cpu().numpy()[:, None, :]

                if self.adaptor is not None:
                    jacobians_vec = self.adaptor.backward_jacobian(jacobians_vec)
                else:
                    jacobians_vec = jacobians_vec[..., self.idx_pin2target]

                grad_qpos_pos = np.matmul(grad_pos_pos, np.array(jacobians_pos))
                grad_qpos_pos = grad_qpos_pos.mean(1).sum(0)
                
                grad_qpos_vec = np.matmul(grad_pos_vec, np.array(jacobians_vec))
                grad_qpos_vec = grad_qpos_vec.mean(1).sum(0)

                grad_qpos = self.position_weight * grad_qpos_pos + self.vector_weight * grad_qpos_vec
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)
                grad[:] = grad_qpos[:]

            return result

        return objective


class PositionPinchOptimizer(Optimizer):
    """结合 Position 和 DexPilot 的优化器，根据 DexPilot 的投影逻辑自动切换
    
    特点：
    1. 使用 DexPilot 的投影逻辑判断是否捏合
    2. 捏合时使用 DexPilot 优化器
    3. 非捏合时使用 Position 优化器
    """
    retargeting_type = "POSITION_PINCH"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],  # 用于位置损失的所有连杆
        finger_tip_link_names: List[str],  # 用于向量损失的指尖连杆
        target_link_human_indices: np.ndarray,
        wrist_link_name: str,
        target_link_human_indices_vec: Optional[np.ndarray] = None,  # 用于dex损失的人类关节索引
        # 优化参数
        huber_delta_pos=0.02,
        huber_delta_vec=0.03,
        norm_delta=4e-3,
        # DexPilot 参数
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"PositionPinch optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)
        
        # 生成 DexPilot 风格的索引
        origin_link_index, task_link_index = DexPilotOptimizer.generate_link_indices(
            self.num_fingers
        )

        # 保存为类的属性
        if target_link_human_indices_vec is None:
            self.target_link_human_indices_vec = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        else:
            self.target_link_human_indices_vec = target_link_human_indices_vec

        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]
        
        super().__init__(robot, target_joint_names, target_link_human_indices)
        
        # 创建两个优化器
        self.position_optimizer = PositionOptimizer(
            robot,
            target_joint_names,
            target_link_names=target_link_names,
            target_link_human_indices=target_link_human_indices,
            norm_delta=norm_delta,
            huber_delta=huber_delta_pos,
        )
        self.dexpilot_optimizer = DexPilotOptimizer(
            robot,
            target_joint_names,
            finger_tip_link_names=finger_tip_link_names,
            wrist_link_name=wrist_link_name,
            target_link_human_indices=self.target_link_human_indices_vec,  # 使用类的属性
            huber_delta=huber_delta_vec,
            norm_delta=norm_delta,
            project_dist=project_dist,
            escape_dist=escape_dist,
            eta1=eta1,
            eta2=eta2,
            scaling=scaling,
        )

        # DexPilot 缓存，用于判断是否捏合
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = DexPilotOptimizer.set_dexpilot_cache(self.num_fingers, eta1, eta2)

        # 缓存用于判断捏合的连杆索引
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

    def is_pinching(self, target_vec: np.ndarray) -> bool:
        """使用 DexPilot 的投影逻辑判断是否捏合"""
        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # 计算目标向量距离
        target_vec_dist = np.linalg.norm(target_vec[:len_proj], axis=1)
        
        # 更新投影状态
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.dexpilot_optimizer.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.dexpilot_optimizer.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # 如果有任何一个向量被投影，就认为是捏合状态
        return np.any(self.projected)

    def retarget(
        self,
        ref_value: Union[np.ndarray, dict],
        fixed_qpos: np.ndarray = np.array([]),
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        根据指尖是否捏合选择优化方法
        """
        if last_qpos is None:
            last_qpos = self.robot.q0[self.idx_pin2target]

        # 判断是否捏合
        if isinstance(ref_value, dict) and "target_vec" in ref_value:
            is_pinching = self.is_pinching(ref_value["target_vec"])
        else:
            is_pinching = False

        if is_pinching:
            # 捏合时用 dexpilot，直接传递 target_vec
            return self.dexpilot_optimizer.retarget(ref_value["target_vec"], fixed_qpos, last_qpos)
        else:
            # 非捏合时用 position
            if isinstance(ref_value, dict):
                ref_value = ref_value["target_pos"]
            return self.position_optimizer.retarget(ref_value, fixed_qpos, last_qpos)

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        """设置关节极限，并同步设置子优化器的关节极限"""
        # 先设置主优化器的关节极限
        super().set_joint_limit(joint_limits, epsilon)
        
        # 同步设置子优化器的关节极限
        # 注意：子优化器的 idx_pin2target 和主优化器是一样的
        self.position_optimizer.set_joint_limit(joint_limits, epsilon)
        self.dexpilot_optimizer.set_joint_limit(joint_limits, epsilon)
        
        # 打印一下关节极限，方便调试
        print("=== 关节极限设置 ===")
        print("主优化器关节极限:", self.opt.get_lower_bounds(), self.opt.get_upper_bounds())
        print("Position优化器关节极限:", self.position_optimizer.opt.get_lower_bounds(), self.position_optimizer.opt.get_upper_bounds())
        print("DexPilot优化器关节极限:", self.dexpilot_optimizer.opt.get_lower_bounds(), self.dexpilot_optimizer.opt.get_upper_bounds())
