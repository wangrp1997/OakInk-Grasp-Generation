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
        
        origin_link_index, task_link_index = self.generate_link_indices(
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
            body_pos_pos = np.array(
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