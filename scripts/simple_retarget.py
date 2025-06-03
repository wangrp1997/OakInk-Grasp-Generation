import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from pytransform3d import rotations


def get_valid_robot_names() -> List[str]:
    """获取所有有效的机器人名称列表（不区分大小写）."""
    return [name.name.lower() for name in RobotName]

def get_robot_name_enum(robot_name: str) -> RobotName:
    """根据机器人名称（不区分大小写）获取对应的枚举值.
    
    Args:
        robot_name: 机器人名称（不区分大小写）
        
    Returns:
        RobotName: 对应的枚举值
        
    Raises:
        ValueError: 当机器人名称无效时抛出
    """
    try:
        return next(name for name in RobotName if name.name.lower() == robot_name.lower())
    except StopIteration:
        raise ValueError(
            f'无效的机器人名称: {robot_name}. '
            f'有效的名称包括: {", ".join(get_valid_robot_names())}'
        )

class SimpleRetargeter:
    def __init__(
        self,
        robot_name: str,
        hand_type: HandType = HandType.right,
        fixed_joints_num: int = 0,
    ):
        """初始化简化版重映射器
        
        Args:
            robot_name: 机器人手名称（不区分大小写，如 'allegro', 'shadow' 等）
            hand_type: 手部类型（左/右）
            fixed_joints_num: 固定关节数量
            
        Raises:
            ValueError: 当机器人名称无效时抛出
        """
        # 自动设置URDF默认目录（如果外部未设置）
        if RetargetingConfig._DEFAULT_URDF_DIR == "./":
            # 自动推断项目根目录
            project_root = Path(__file__).absolute().parent.parent
            default_dir = project_root / 'assets' / 'robots' / 'hands'
            RetargetingConfig.set_default_urdf_dir(default_dir)
        self.robot_name = get_robot_name_enum(robot_name)
        self.hand_type = hand_type
        self.fixed_joints_num = fixed_joints_num
        
        # 加载重映射配置
        config_path = get_default_config_path(
            self.robot_name, 
            RetargetingType.position, 
            self.hand_type
        )
        print(f"config_path: {config_path}")
        override = dict(add_dummy_free_joint=True)
        config = RetargetingConfig.load_from_file(config_path, override=override)
        self.retargeting = config.build()
        
    def retarget(
        self, 
        mano_joints: np.ndarray, 
        mano_pose: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """将 MANO 手部姿态重映射到机器人关节角度
        
        Args:
            mano_joints: MANO 手部关键点位置 (21, 3)
            mano_pose: MANO 手部姿态参数 (48,)
            
        Returns:
            Tuple[np.ndarray, Dict]: 
                - 机器人关节角度
                - 包含重映射信息的字典
        """
        # 提取手腕旋转四元数
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            mano_pose[0:3]
        )
        
        # 使用重映射器进行转换
        fixed_qpos = np.zeros(self.fixed_joints_num)
        indices = self.retargeting.optimizer.target_link_human_indices
        ref_value = mano_joints[indices, :]
        
        # 获取机器人关节角度
        qpos = self.retargeting.retarget(ref_value, fixed_qpos)
        
        # 收集重映射信息
        info = {
            'robot_name': self.robot_name.value,
            'joint_names': self.retargeting.joint_names,
            'target_indices': indices.tolist(),
            'wrist_quat': wrist_quat.tolist()
        }
        
        return qpos, info


def main():
    # 示例使用
    retargeter = SimpleRetargeter(
        robot_name='botyard',
        hand_type=HandType.right
    )
    
    # 这里应该从 GrabNet 的输出中获取
    # mano_joints = ...  # (21, 3) 的手部关键点
    # mano_pose = ...    # (48,) 的手部姿态参数
    
    # qpos, info = retargeter.retarget(mano_joints, mano_pose)
    # print(f"机器人关节角度: {qpos}")
    # print(f"重映射信息: {info}")


if __name__ == '__main__':
    main() 