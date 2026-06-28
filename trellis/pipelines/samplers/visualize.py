
# import utils3d
import os
# import open3d as o3d
# import open3d.visualization.rendering as rendering

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from PIL import Image, ImageDraw, ImageFont, ImageOps
import re
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa

import numpy as np
import torch
import open3d as o3d
from utils3d.torch.transforms import axis_angle_to_matrix, euler_angles_to_matrix


def save_loss_curve(loss_list, output_path, filename="loss_curve.png", title="Loss Curve Over Time"):
    """
    保存损失曲线图到指定路径。

    参数:
    - loss_list: list of float，loss 每一步的数值。
    - output_path: str，图像保存目录。
    - filename: str，图像文件名，默认 "loss_curve.png"。
    - title: str，图像标题。
    """
    if not loss_list:
        print("⚠️ loss_list is empty, skipping loss curve plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, marker='o', label="Dice Loss ×100")
    plt.xlabel("Timestep index")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved loss curve to {save_path}")



def save_loss_curves(loss_list1, loss_list2, loss_list3, output_path, filename="loss_curve.png", title="Loss Curve Over Time"):
    """
    保存损失曲线图到指定路径。

    参数:
    - loss_list: list of float，loss 每一步的数值。
    - output_path: str，图像保存目录。
    - filename: str，图像文件名，默认 "loss_curve.png"。
    - title: str，图像标题。
    """
    # if not loss_list:
    #     print("⚠️ loss_list is empty, skipping loss curve plot.")
    #     return

    plt.figure(figsize=(8, 4))

    # 绘制曲线
    plt.plot(loss_list1, marker='o', markevery=5, label="BCE loss", color='r')
    plt.plot(loss_list2, marker='s', markevery=5, label="Dice loss", color='g')
    plt.plot(loss_list3, marker='^', markevery=5, label="Sum loss", color='b')
    
    plt.xlabel("Timestep index")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # 避免标题或标签被截断

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved loss curve to {save_path}")

def coords_to_positions(coords: torch.Tensor, resolution: int = 64):
    """
    Recover original positions from coords.

    Args:
        coords: [N, 4] tensor (batch_idx, x, y, z)
        resolution: voxel resolution used in quantization (default: 64)

    Returns:
        positions: [N, 3] float tensor ∈ [-0.5, 0.5]
    """
    indices = coords[:, 1:4].float()
    # positions = indices / resolution - 0.5
    positions = (indices + 0.5) / resolution - 0.5
    return positions


def save_pointcloud_matplotlib(positions_np, save_path, color=[0.5, 0.5, 0.5]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], c=[color], s=1)
    ax.set_axis_off()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

# def save_pointcloud_matplotlib_multiview(positions_np, save_path, color=[0.5, 0.5, 0.5]):
#     fig = plt.figure(figsize=(12, 3))  # 横向排布四个子图

#     elev = 0  # 默认仰角
#     views = [
#         {"title": "Front",     "azim": 0},
#         {"title": "Side",      "azim": 90},
#         {"title": "Top",       "azim": 0,  "elev": 90},
#         {"title": "Isometric", "azim": 45, "elev": 30},
#     ]

#     # 判断是否为空点云
#     if positions_np.shape[0] == 0:
#         # print("⚠️ positions_np is empty, saving blank views.")
#         for i, view in enumerate(views):
#             ax = fig.add_subplot(1, 4, i + 1, projection='3d')
#             ax.set_title(view["title"], fontsize=10)
#             ax.set_axis_off()
#     else:
#         # 坐标中心和最大跨度用于三轴等比缩放
#         center = positions_np.mean(axis=0)
#         max_range = np.ptp(positions_np, axis=0).max()

#         for i, view in enumerate(views):
#             ax = fig.add_subplot(1, 4, i + 1, projection='3d')
#             ax.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], c=[color], s=1)
#             ax.set_title(view["title"], fontsize=10)
#             ax.set_axis_off()

#             azim = view.get("azim", 0)
#             elev_i = view.get("elev", elev)
#             ax.view_init(elev=elev_i, azim=azim)

#             # 设置等比缩放
#             ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
#             ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
#             ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

def save_pointcloud_matplotlib_multiview(positions_np, save_path, color=[0.5, 0.5, 0.5]):
    fig = plt.figure(figsize=(12, 3))  # 横向排布四个子图

    elev = 0  # 默认仰角
    views = [
        {"title": "Front",     "azim": 0},
        {"title": "Side",      "azim": 90},
        {"title": "Top",       "azim": 0,  "elev": 90},
        {"title": "Isometric", "azim": 45, "elev": 30},
    ]

    # 判断是否为空点云
    if positions_np.shape[0] == 0:
        # print("⚠️ positions_np is empty, saving blank views.")
        for i, view in enumerate(views):
            ax = fig.add_subplot(1, 4, i + 1, projection='3d')
            ax.set_title(view["title"], fontsize=10)
            ax.set_axis_off()
    else:
        # # 坐标中心和最大跨度用于三轴等比缩放
        # # 计算点云的范围
        # min_vals = positions_np.min(axis=0)
        # max_vals = positions_np.max(axis=0)
        # # 计算范围
        # scale = np.ptp(positions_np, axis=0).max()  # 计算最大的跨度

        for i, view in enumerate(views):
            ax = fig.add_subplot(1, 4, i + 1, projection='3d')
            ax.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], c=[color], s=1)
            ax.set_title(view["title"], fontsize=10)
            ax.set_axis_off()
            ax.title.set_position([0.5, 1.05])  # 设置标题位置，0.5 是水平位置，1.05 是垂直位置


            azim = view.get("azim", 0)
            elev_i = view.get("elev", elev)
            ax.view_init(elev=elev_i, azim=azim)


            # # 设置统一的比例缩放，确保 X, Y, Z 轴范围一致
            # max_range = np.max([np.ptp(positions_np[:, 0]), np.ptp(positions_np[:, 1]), np.ptp(positions_np[:, 2])])

            # # 设置每个轴的范围，确保它们对称
            # ax.set_xlim(min_vals[0], min_vals[0] + max_range)
            # ax.set_ylim(min_vals[1], min_vals[1] + max_range)
            # ax.set_zlim(min_vals[2], min_vals[2] + max_range)
          
            # 设置每个轴的范围，确保它们对称
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            ax.set_zlim(-0.55, 0.55)

            # # 设置等比缩放
            # ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
            # ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
            # ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def vis_step_voxel(t, coords, output_path=None, out_name=None):
    positions = coords_to_positions(coords, resolution=64)
    positions_np = positions.cpu().numpy()

    # 可设置颜色渐变（红→蓝）
    # color = [t / num_steps, 0.2, 1.0 - t / num_steps]
    if t>1:
        color = [t / 2, 0.2, 1.0 - t / 2]
    else:
        color = [t / 1, 0.2, 1.0 - t / 1]


    if output_path is None:
        if out_name is None:
            save_path =os.path.join('./tmp',  f"step_{t:06f}.png")
            # utils3d.io.write_ply(os.path.join('./tmp', 
            #                         f'voxels_out-{t}.ply'), positions.cpu().numpy())
        else:
            save_path =os.path.join('./tmp',  f"{out_name}-step_{t:06f}.png")
    else:
        if out_name is None:
            save_path =os.path.join(output_path,  f"step_{t:06f}.png")
            # utils3d.io.write_ply(os.path.join(output_path, 
            #                         f'voxels_out-{t}.ply'), positions.cpu().numpy()
        else:
            save_path =os.path.join(output_path,  f"{out_name}-step_{t:06f}.png")
    save_pointcloud_matplotlib_multiview(positions_np, save_path, color)



def concat_images_from_folder(folder_path, 
                                images_per_row=3,
                                pattern=r"step_(\d+\.\d+)\.png",
                                out_name=None
                                ):
    """
    将某文件夹内所有 step_xx_xx_multiview.png 按顺序拼接成一张大图
    :param folder_path: 输入图像的文件夹
    :param output_path: 输出合图路径
    :param direction: "horizontal" 或 "vertical"
    :param pattern: 正则模式，用于提取排序编号
    """
    
    # 收集匹配文件（按浮点时间步降序排序）
    files = sorted([
        f for f in sorted(os.listdir(folder_path)) if re.match(pattern, f)
    ], key=lambda x: float(re.findall(pattern, x)[0]), reverse=True)

    if not files:
        print("❌ No matching files found.")
        return

    # 加载图像
    images = []
    for f in files:
        img = Image.open(os.path.join(folder_path, f)).convert("RGB")

        # 缩小图像为原来的1/4大小
        new_size = (img.width // 2 // 2, img.height // 2 // 2)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(img)
        label = os.path.splitext(f)[0]  # 文件名（无扩展名）
        
        # 默认字体；如需自定义字体，可使用 truetype 字体
        draw.text((40, 40), label, fill=(255, 0, 0))  # 红色文字

        images.append(img)

    img_width, img_height = images[0].size
    total_images = len(images)
    num_rows = math.ceil(total_images / images_per_row)

    # 创建拼图画布
    grid_width = images_per_row * img_width
    grid_height = num_rows * img_height
    final_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    # 粘贴带文字的图像
    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * img_width
        y = row * img_height
        final_image.paste(img, (x, y))

    # 保存大图
    if out_name is None:
        output_path = os.path.join(folder_path, 'voxel_timesteps_multiview_concat.png')
    else:
        output_path = os.path.join(folder_path, f'{out_name}-voxel_timesteps_multiview_concat.png')
    final_image.save(output_path)
    print(f"✅ Saved grid image to {output_path}")

    # 删除原图
    for f in files:
        if 'step_2.00000' not in f:
            os.remove(os.path.join(folder_path, f))
    print(f"🧹 Deleted {len(files)} individual images.")



def rotate_voxel(coords: torch.Tensor, rotation_angle: int, resolution=64) -> torch.Tensor:

    # Calculate the centroid of the coordinates
    # centroid = torch.mean(coords.float(), dim=0)
    centroid = torch.mean(coords[:, 1:4].float(), dim=0)
    
    # Translate the coordinates so that the centroid is at the origin
    # coords_centered = coords.float() - centroid
    coords_centered = coords[:, 1:4].float() - centroid

    device=coords.device
    # Define rotation matrices for 90, 180, and 270 degrees
    if rotation_angle == 90:
        rotation_matrix = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]],device=device)
    elif rotation_angle == 180:
        rotation_matrix = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],device=device)
    elif rotation_angle == 270:
        rotation_matrix = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]],device=device)
    else:
        raise ValueError("Rotation angle must be one of [90, 180, 270]")

    # Apply rotation to x, y coordinates
    # coords_rotated = coords[:, 1:4].float()
    # coords_rotated = coords
    rotated_coords_centered  = torch.matmul(coords_centered, rotation_matrix.T.float())
    
    # Translate the coordinates back to their original position
    rotated_coords = rotated_coords_centered + centroid
    
    # Combine batch_idx with the rotated coordinates
    rotated_coords = torch.cat([coords[:, :1], rotated_coords], dim=1) # torch.Size([17076, 4])
    # rotated_coords = torch.cat([coords[:, :0], rotated_coords], dim=1)
    
    rotated_coords = torch.round(rotated_coords).to(torch.long)
    rotated_coords = torch.clamp(rotated_coords, 0, resolution-1)

    return rotated_coords

def rotate_voxel_returen_new_voxel(coords: torch.Tensor, rotation_angle: int, resolution=64) -> torch.Tensor:

    # coords = coords.squeeze(1)
    # Calculate the centroid of the coordinates
    # centroid = torch.mean(coords.float(), dim=0)
    centroid = torch.mean(coords[:, 1:4].float(), dim=0)
    
    # Translate the coordinates so that the centroid is at the origin
    # coords_centered = coords.float() - centroid
    coords_centered = coords[:, 1:4].float() - centroid

    device=coords.device
    # Define rotation matrices for 90, 180, and 270 degrees
    if rotation_angle == 90:
        rotation_matrix = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]],device=device)
    elif rotation_angle == 180:
        rotation_matrix = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],device=device)
    elif rotation_angle == 270:
        rotation_matrix = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]],device=device)
    else:
        raise ValueError("Rotation angle must be one of [90, 180, 270]")

    # Apply rotation to x, y coordinates
    # coords_rotated = coords[:, 1:4].float()
    # coords_rotated = coords
    rotated_coords_centered  = torch.matmul(coords_centered, rotation_matrix.T.float())
    
    # Translate the coordinates back to their original position
    rotated_coords = rotated_coords_centered + centroid
    
    # # Combine batch_idx with the rotated coordinates
    # rotated_coords = torch.cat([coords[:, :1], rotated_coords], dim=1) # torch.Size([17076, 4])
    # # rotated_coords = torch.cat([coords[:, :0], rotated_coords], dim=1)
    
    rotated_coords = torch.round(rotated_coords).to(torch.long)
    rotated_coords = torch.clamp(rotated_coords, 0, resolution-1)

    rotated_voxels = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    rotated_voxels[:, rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]] = 1
    rotated_voxels = rotated_voxels.cuda().float()

    # return rotated_voxels.unsqueeze(1)
    return rotated_voxels



def save_ply_from_tensor(positions: torch.Tensor, save_path: str):
    """
    Save a tensor of 3D positions to a .ply file.

    Args:
        positions: [N, 3] tensor of 3D positions
        save_path: Path to save the .ply file
    """
    # Convert the tensor to numpy
    positions_np = positions.cpu().numpy()

    # Create a PointCloud object in Open3D
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions_np)

    # Save to .ply file
    o3d.io.write_point_cloud(save_path, point_cloud)
    print(f"✅ Saved point cloud to {save_path}")


def symmetry_voxel(coords: torch.Tensor,
                        resolution: int = 64, 
                        save_path: str = "./tmp"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # # Recover positions
    # # indices = coords[:, 1:4].float()
    # # positions = indices / resolution - 0.5
    # positions = coords_to_positions(coords, resolution=resolution)
    # # Save the original positions to .ply
    # # save_ply_from_tensor(positions, os.path.join(save_path, 'original_positions.ply'))

    # Apply rotations (90, 180, 270 degrees)
    rotated_coords_90 = rotate_voxel(coords, 90, resolution=resolution)
    rotated_coords_180 = rotate_voxel(coords, 180, resolution=resolution)
    rotated_coords_270 = rotate_voxel(coords, 270, resolution=resolution)
    
    all_coords = torch.cat([coords, rotated_coords_90, rotated_coords_180, rotated_coords_270], dim=0)
    all_positions = coords_to_positions(all_coords, resolution=resolution)
    save_ply_from_tensor(all_positions, os.path.join(save_path, 'symmetric_voxel_t.ply'))

    # all_coords = torch.round(all_coords).to(torch.long)
    # all_coords = torch.clamp(all_coords, 0, resolution-1)

    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = 1
    ss = ss.cuda().float()

    # # # Convert rotated coords to positions
    # # rotated_positions_90 = (rotated_positions_90[:, 1:4].float() / resolution) - 0.5
    # # rotated_positions_180 = (rotated_positions_180[:, 1:4].float() / resolution) - 0.5
    # # rotated_positions_270 = (rotated_positions_270[:, 1:4].float() / resolution) - 0.5

    # # Convert rotated coords to positions
    # rotated_positions_90 = (rotated_coords_90.float() / resolution) - 0.5
    # rotated_positions_180 = (rotated_coords_180.float() / resolution) - 0.5
    # rotated_positions_270 = (rotated_coords_270.float() / resolution) - 0.5


    # # # Save rotated results to .ply
    # # save_ply_from_tensor(rotated_positions_90, os.path.join(save_path, 'rotated_positions_90.ply'))
    # # save_ply_from_tensor(rotated_positions_180, os.path.join(save_path, 'rotated_positions_180.ply'))
    # # save_ply_from_tensor(rotated_positions_270, os.path.join(save_path, 'rotated_positions_270.ply'))

    # # Combine original and rotated positions
    # all_positions = torch.cat([positions, rotated_positions_90, rotated_positions_180, rotated_positions_270], dim=0)
    # save_ply_from_tensor(all_positions, os.path.join(save_path, 'symmetric_voxel_t.ply'))

    return ss.unsqueeze(1), all_coords


# def voxelize(position, resolution=64):

#     coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
#     ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
#     ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
#     ss = ss.cuda().float()


def apply_rotation_and_translation_axis_angle(coords, axis_angle, translation_vector = torch.tensor([0.0, 0.0, 0.0])):
    # axis_angle: shape (3,)
    R = axis_angle_to_matrix(axis_angle.unsqueeze(0))[0]  # shape (3,3)
    # R = axis_angle_to_matrix(axis_angle)
    centroid = coords.mean(dim=0)
    coords_centered = coords - centroid
    rotated_coords = torch.matmul(coords_centered, R.T.float()) + translation_vector
    rotated_coords = rotated_coords + centroid
    return rotated_coords


# def apply_rotation_and_translation_axis_angle_v2(coords, axis_angle, translation_vector = torch.tensor([0.0, 0.0, 0.0])):
#     # axis_angle: shape (3,)
#     R = axis_angle_to_matrix(axis_angle.unsqueeze(0))[0]  # shape (3,3)
#     # # R = axis_angle_to_matrix(axis_angle)
#     # centroid = coords.mean(dim=0)
#     # coords_centered = coords - centroid
#     rotated_coords = torch.matmul(coords, R.T.float()) + translation_vector
#     # rotated_coords = torch.matmul(coords_centered, R.T.float()) + translation_vector
#     # rotated_coords = rotated_coords + centroid
#     return rotated_coords


def apply_rotation_and_translation_euler_angle(coords, euler_angle, translation_vector = torch.tensor([0.0, 0.0, 0.0])):
    # axis_angle: shape (3,)
    R = euler_angles_to_matrix(euler_angle.unsqueeze(0))[0]  # shape (3,3)
    # R = axis_angle_to_matrix(axis_angle)
    centroid = coords.mean(dim=0)
    coords_centered = coords - centroid
    rotated_coords = torch.matmul(coords_centered, R.T.float()) + translation_vector
    rotated_coords = rotated_coords + centroid
    return rotated_coords


