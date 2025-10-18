#!/usr/bin/env python3
"""
Rotation utilities to replace pytorch3d dependencies.
Implements common rotation conversions and transformations.
"""

import torch
import numpy as np

# def rotation_6d_to_matrix(rotation_6d):
#     """
#     Convert 6D rotation representation to rotation matrix using Gram-Schmidt orthogonalization.
    
#     This is compatible with pytorch3d's rotation_6d_to_matrix function.
    
#     Args:
#         rotation_6d: tensor of shape [..., 6] - 6D rotation representation
#                     First 3 elements represent the first column of rotation matrix
#                     Last 3 elements represent the second column of rotation matrix
    
#     Returns:
#         rotation matrix of shape [..., 3, 3]
#     """
#     if isinstance(rotation_6d, np.ndarray):
#         return rotation_6d_to_matrix_numpy(rotation_6d)
    
#     batch_shape = rotation_6d.shape[:-1]
#     rotation_6d = rotation_6d.view(-1, 6)
    
#     # Extract first two columns
#     col1 = rotation_6d[:, :3]  # First column
#     col2 = rotation_6d[:, 3:]  # Second column
    
#     # Normalize first column
#     col1 = col1 / torch.norm(col1, dim=-1, keepdim=True)
    
#     # Gram-Schmidt orthogonalization for second column
#     col2 = col2 - torch.sum(col1 * col2, dim=-1, keepdim=True) * col1
#     col2 = col2 / torch.norm(col2, dim=-1, keepdim=True)
    
#     # Third column is cross product
#     col3 = torch.cross(col1, col2, dim=-1)
    
#     # Stack columns to form rotation matrix
#     matrix = torch.stack([col1, col2, col3], dim=-1)
    
#     return matrix.view(*batch_shape, 3, 3)

def rotation_6d_to_matrix_numpy(rotation_6d):
    """
    NumPy version of rotation_6d_to_matrix.
    
    Args:
        rotation_6d: numpy array of shape [..., 6] - 6D rotation representation
    
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2)



# def matrix_to_rotation_6d(matrix):
#     """
#     Convert rotation matrix to 6D rotation representation.
    
#     Args:
#         matrix: tensor of shape [..., 3, 3] - rotation matrices
    
#     Returns:
#         6D rotation representation of shape [..., 6]
#     """
#     if isinstance(matrix, np.ndarray):
#         return matrix_to_rotation_6d_numpy(matrix)
    
#     batch_shape = matrix.shape[:-2]
#     matrix = matrix.view(-1, 3, 3)
    
#     # Extract first two columns
#     col1 = matrix[:, :, 0]  # First column
#     col2 = matrix[:, :, 1]  # Second column
    
#     # Concatenate to form 6D representation
#     rotation_6d = torch.cat([col1, col2], dim=-1)
    
#     return rotation_6d.view(*batch_shape, 6)



def mat_to_9d_numpy(matrix):
    rot = matrix[..., :3, :3]
    rot_6d = matrix_to_rotation_6d_numpy(rot)
    trans = matrix[...,:3, 3]
    return np.concatenate([trans, rot_6d], axis=-1)

def matrix_to_rotation_6d_numpy(matrix):
    """
    NumPy version of matrix_to_rotation_6d.
    
    Args:
        matrix: numpy array of shape [..., 3, 3] - rotation matrices
    
    Returns:
        6D rotation representation of shape [..., 6]
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))

def quaternion_to_matrix(quaternion):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion: tensor of shape [..., 4] - quaternion [w, x, y, z]
    
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    if isinstance(quaternion, np.ndarray):
        return quaternion_to_matrix_numpy(quaternion)
    
    batch_shape = quaternion.shape[:-1]
    quaternion = quaternion.view(-1, 4)
    
    w, x, y, z = quaternion.unbind(-1)
    
    # Compute rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    matrix = torch.stack([
        torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
    ], dim=-2)
    
    return matrix.view(*batch_shape, 3, 3)


def quaternion_to_matrix_numpy(quaternion):
    """
    NumPy version of quaternion_to_matrix.
    
    Args:
        quaternion: numpy array of shape [..., 4] - quaternion [w, x, y, z]
    
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    if quaternion.ndim == 1:
        # Single quaternion
        w, x, y, z = quaternion
        
        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        matrix = np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        ])
        
        return matrix
    else:
        # Batch of quaternions
        batch_shape = quaternion.shape[:-1]
        quaternion = quaternion.reshape(-1, 4)
        
        matrices = []
        for quat in quaternion:
            matrices.append(quaternion_to_matrix_numpy(quat))
        
        result = np.stack(matrices, axis=0)
        return result.reshape(*batch_shape, 3, 3)


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    
    Args:
        axis_angle: tensor of shape [..., 3] - axis-angle representation
    
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    if isinstance(axis_angle, np.ndarray):
        return axis_angle_to_matrix_numpy(axis_angle)
    
    batch_shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.view(-1, 3)
    
    # Compute angle (magnitude) and axis (normalized direction)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Rodrigues' formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Cross product matrix
    K = torch.zeros(axis_angle.shape[0], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    # Identity matrix
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(K)
    
    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    matrix = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)
    
    return matrix.view(*batch_shape, 3, 3)


def axis_angle_to_matrix_numpy(axis_angle):
    """
    NumPy version of axis_angle_to_matrix.
    
    Args:
        axis_angle: numpy array of shape [..., 3] - axis-angle representation
    
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    if axis_angle.ndim == 1:
        # Single axis-angle
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-8:
            return np.eye(3)
        
        axis = axis_angle / angle
        
        # Rodrigues' formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Cross product matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return matrix
    else:
        # Batch of axis-angles
        batch_shape = axis_angle.shape[:-1]
        axis_angle = axis_angle.reshape(-1, 3)
        
        matrices = []
        for aa in axis_angle:
            matrices.append(axis_angle_to_matrix_numpy(aa))
        
        result = np.stack(matrices, axis=0)
        return result.reshape(*batch_shape, 3, 3)


def test():
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

    rotation_6d = torch.randn(10, 6)
    matrix = rotation_6d_to_matrix(rotation_6d)
    rotation_6d_numpy = rotation_6d.numpy()
    matrix_numpy = rotation_6d_to_matrix_numpy(rotation_6d_numpy)
    print(matrix.numpy() - matrix_numpy)

    rotation_6d_numpy_2 = matrix_to_rotation_6d_numpy(matrix.numpy())
    rotation_6d_2 = matrix_to_rotation_6d(matrix)

    print(rotation_6d_numpy_2 - rotation_6d_2.numpy())


