import os
import argparse
import numpy as np
import open3d as o3d

def get_partial_point_cloud_by_view(full_pc, view_point, max_angle=60, view_direction=None):
    """从完整点云中获取从某个视角可见的部分点云"""
    vectors = full_pc - view_point
    # 使用指定的视角方向，默认为z轴正方向
    if view_direction is None:
        view_direction = np.array([0, 0, 1])
    view_direction = view_direction / np.linalg.norm(view_direction)  # 归一化
    
    # 计算每个点到视角点的向量与视角方向的夹角
    dot_products = np.dot(vectors, view_direction)
    vector_norms = np.linalg.norm(vectors, axis=1)
    # 避免除以零
    vector_norms[vector_norms < 1e-10] = 1e-10
    cos_angles = dot_products / vector_norms
    # 确保cos_angles在[-1, 1]范围内
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    mask = angles < np.radians(max_angle)
    
    print(f'视角点: {view_point}')
    print(f'视角方向: {view_direction}')
    print(f'点云范围: X[{full_pc[:,0].min():.3f}, {full_pc[:,0].max():.3f}], '
          f'Y[{full_pc[:,1].min():.3f}, {full_pc[:,1].max():.3f}], '
          f'Z[{full_pc[:,2].min():.3f}, {full_pc[:,2].max():.3f}]')
    print(f'选中点数: {np.sum(mask)}/{len(full_pc)}')
    return full_pc[mask]

def get_partial_point_cloud_by_distance(full_pc, center_point, max_distance):
    """通过距离阈值获取局部点云"""
    distances = np.linalg.norm(full_pc - center_point, axis=1)
    mask = distances < max_distance
    print(f'中心点: {center_point}')
    print(f'点云范围: X[{full_pc[:,0].min():.3f}, {full_pc[:,0].max():.3f}], '
          f'Y[{full_pc[:,1].min():.3f}, {full_pc[:,1].max():.3f}], '
          f'Z[{full_pc[:,2].min():.3f}, {full_pc[:,2].max():.3f}]')
    print(f'选中点数: {np.sum(mask)}/{len(full_pc)}')
    return full_pc[mask]

def get_partial_point_cloud_by_voxel(full_pc, center_point, voxel_size=0.01, num_voxels=10):
    """通过体素网格获取局部点云"""
    min_bound = center_point - num_voxels * voxel_size / 2
    max_bound = center_point + num_voxels * voxel_size / 2
    mask = np.all((full_pc >= min_bound) & (full_pc <= max_bound), axis=1)
    print(f'中心点: {center_point}')
    print(f'体素范围: X[{min_bound[0]:.3f}, {max_bound[0]:.3f}], '
          f'Y[{min_bound[1]:.3f}, {max_bound[1]:.3f}], '
          f'Z[{min_bound[2]:.3f}, {max_bound[2]:.3f}]')
    print(f'选中点数: {np.sum(mask)}/{len(full_pc)}')
    return full_pc[mask]

def process_point_cloud(input_path, output_path, method='distance', **kwargs):
    """处理单个点云文件."""
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        raise ValueError(f'点云文件 {input_path} 为空或无法读取')
    
    full_pc = np.asarray(pcd.points)
    print(f'原始点云点数: {len(full_pc)}')
    
    # 计算点云中心
    center_point = np.mean(full_pc, axis=0)
    
    # 根据方法生成局部点云
    if method == 'view':
        # 修改视角点位置，从侧面观察，距离更远
        x_offset = (np.max(full_pc[:, 0]) - np.min(full_pc[:, 0])) * 2.0  # 使用点云宽度的2倍作为偏移
        view_point = center_point + np.array([x_offset, 0, 0])  # 从x轴正方向观察
        print(f'点云宽度: {np.max(full_pc[:, 0]) - np.min(full_pc[:, 0]):.3f}')
        print(f'X轴偏移: {x_offset:.3f}')
        # 修改视角方向为x轴正方向
        partial_pc = get_partial_point_cloud_by_view(
            full_pc, view_point, kwargs.get('max_angle', 90),
            view_direction=np.array([1, 0, 0])  # 从x轴正方向观察
        )
    elif method == 'distance':
        partial_pc = get_partial_point_cloud_by_distance(
            full_pc, center_point, kwargs.get('max_distance', 0.05)
        )
    elif method == 'voxel':
        partial_pc = get_partial_point_cloud_by_voxel(
            full_pc, center_point,
            kwargs.get('voxel_size', 0.01),
            kwargs.get('num_voxels', 10)
        )
    else:
        raise ValueError(f'Unknown method: {method}')
    
    if len(partial_pc) == 0:
        raise ValueError(f'生成的局部点云为空，请调整参数')
    
    # 创建输出点云
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial_pc)
    
    # 估计法向量
    pcd_partial.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    
    # 使用泊松重建生成网格
    print('正在生成网格...')
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_partial, depth=8, width=0, scale=1.1, linear_fit=False
    )
    
    # 保存为带面信息的PLY格式
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f'已保存带面信息的点云到: {output_path}')
    
    return output_path

def main():
    """主函数：处理单个点云文件并生成局部点云和网格模型."""
    parser = argparse.ArgumentParser(description='从完整点云生成局部点云和网格模型')
    parser.add_argument('--input_file', type=str, required=True,
                      help='输入点云文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                      help='输出点云文件路径')
    parser.add_argument('--method', type=str, default='distance',
                      choices=['view', 'distance', 'voxel'],
                      help='采样方法：view(视角采样), distance(距离采样), '
                           'voxel(体素采样)')
    parser.add_argument('--max_angle', type=float, default=90,
                      help='视角采样时的最大视角范围（度）')
    parser.add_argument('--max_distance', type=float, default=0.05,
                      help='距离采样时的最大距离阈值')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                      help='体素采样时的体素大小')
    parser.add_argument('--num_voxels', type=int, default=10,
                      help='体素采样时每个方向上的体素数量')
    parser.add_argument('--poisson_depth', type=int, default=8,
                      help='泊松重建的深度参数')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    try:
        output_path = process_point_cloud(
            args.input_file,
            args.output_file,
            method=args.method,
            max_angle=args.max_angle,
            max_distance=args.max_distance,
            voxel_size=args.voxel_size,
            num_voxels=args.num_voxels
        )
        print(f'已处理: {args.input_file} -> {output_path}')
    except Exception as e:
        print(f'处理文件时出错: {str(e)}')

if __name__ == '__main__':
    main() 