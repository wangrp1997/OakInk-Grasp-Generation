from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int, hand_type: str, data_id: int, fixed_joints_num: int):
    dataset = DexYCBVideoDataset(data_root, hand_type=hand_type)
    if robots is None:
        viewer = HandDatasetSAPIENViewer(hand_type=hand_type, headless=False)
    else:
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType[hand_type], headless=False, 
            fixed_joints_num=fixed_joints_num
        )

    # Data ID, feel free to change it to visualize different trajectory

    sampled_data = dataset[data_id]

    for key, value in sampled_data.items():
        # if "pose" not in key:
        print(f"{key}: {value}")
    viewer.load_object_hand(sampled_data)
    viewer.render_dexycb_data(sampled_data, fps, data_id=data_id)


def main(dexycb_dir: str,
        robots: Optional[List[RobotName]] = None,
        fps: int = 10,
        hand_type: str = "right",
        data_id: int = 0,
        fixed_joints_num: int = 2):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory
        hand_type: The type of hand to render, either "left" or "right"
        data_id: The index of the data to render
        fixed_joints_num: The number of fixed joints for the robot hands

    """
    data_root = Path(dexycb_dir).absolute()
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    viz_hand_object(robots, data_root, fps, hand_type, data_id, fixed_joints_num)


if __name__ == "__main__":
    tyro.cli(main)
