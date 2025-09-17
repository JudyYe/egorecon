#!/usr/bin/env python3
"""Quaternion and rotation conversion utilities for hand-to-object training"""

import torch
import math


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    Args:
        axis_angle: tensor of shape [..., 3] - axis-angle representation
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    # Handle batch dimensions
    batch_shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.view(-1, 3)
    
    # Compute angle (magnitude of axis-angle vector)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    
    # Handle case where angle is zero (identity rotation)
    small_angle = angle < 1e-8
    angle = torch.where(small_angle, torch.ones_like(angle), angle)
    
    # Normalize axis
    axis = axis_angle / angle
    
    # Rodrigues' rotation formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Cross product matrix [axis]_x
    axis_x = axis[:, 0]
    axis_y = axis[:, 1] 
    axis_z = axis[:, 2]
    
    zeros = torch.zeros_like(axis_x)
    
    K = torch.stack([
        torch.stack([zeros, -axis_z, axis_y], dim=-1),
        torch.stack([axis_z, zeros, -axis_x], dim=-1),
        torch.stack([-axis_y, axis_x, zeros], dim=-1)
    ], dim=-2)
    
    # Identity matrix
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(axis_angle.shape[0], -1, -1)
    
    # Rodrigues' formula: R = I + sin(θ) * K + (1 - cos(θ)) * K^2
    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)
    
    # Handle small angles (use identity matrix)
    small_angle_expanded = small_angle.unsqueeze(-1).expand(-1, 3, 3)
    R = torch.where(small_angle_expanded, I, R)
    
    return R.view(*batch_shape, 3, 3)


def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to axis-angle representation.
    Args:
        matrix: tensor of shape [..., 3, 3] - rotation matrices
    Returns:
        axis-angle representation of shape [..., 3]
    """
    batch_shape = matrix.shape[:-2]
    matrix = matrix.view(-1, 3, 3)
    
    # Compute angle from trace
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # Handle small angles
    small_angle = angle < 1e-6
    
    # For small angles, axis doesn't matter much, use [1, 0, 0]
    axis = torch.zeros_like(matrix[:, :, 0])
    axis[:, 0] = 1.0
    
    # For non-small angles, compute axis from skew-symmetric part
    valid_mask = ~small_angle
    if valid_mask.any():
        sin_angle = torch.sin(angle[valid_mask])
        axis[valid_mask, 0] = (matrix[valid_mask, 2, 1] - matrix[valid_mask, 1, 2]) / (2 * sin_angle)
        axis[valid_mask, 1] = (matrix[valid_mask, 0, 2] - matrix[valid_mask, 2, 0]) / (2 * sin_angle)
        axis[valid_mask, 2] = (matrix[valid_mask, 1, 0] - matrix[valid_mask, 0, 1]) / (2 * sin_angle)
    
    # Scale by angle
    axis_angle = axis * angle.unsqueeze(-1)
    
    return axis_angle.view(*batch_shape, 3)


def matrix_to_quaternion(matrix):
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    Args:
        matrix: tensor of shape [..., 3, 3] - rotation matrices
    Returns:
        quaternion of shape [..., 4] - (w, x, y, z)
    """
    batch_shape = matrix.shape[:-2]
    matrix = matrix.view(-1, 3, 3)
    
    # Shepperd's method for numerical stability
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
    qw1 = 0.25 * s1
    qx1 = (matrix[mask1, 2, 1] - matrix[mask1, 1, 2]) / s1
    qy1 = (matrix[mask1, 0, 2] - matrix[mask1, 2, 0]) / s1
    qz1 = (matrix[mask1, 1, 0] - matrix[mask1, 0, 1]) / s1
    
    # Initialize output
    quat = torch.zeros(matrix.shape[0], 4, device=matrix.device, dtype=matrix.dtype)
    quat[mask1] = torch.stack([qw1, qx1, qy1, qz1], dim=-1)
    
    # Case 2: m00 > m11 and m00 > m22
    mask2 = ~mask1 & (matrix[:, 0, 0] > matrix[:, 1, 1]) & (matrix[:, 0, 0] > matrix[:, 2, 2])
    s2 = torch.sqrt(1.0 + matrix[mask2, 0, 0] - matrix[mask2, 1, 1] - matrix[mask2, 2, 2]) * 2
    qw2 = (matrix[mask2, 2, 1] - matrix[mask2, 1, 2]) / s2
    qx2 = 0.25 * s2
    qy2 = (matrix[mask2, 0, 1] + matrix[mask2, 1, 0]) / s2
    qz2 = (matrix[mask2, 0, 2] + matrix[mask2, 2, 0]) / s2
    quat[mask2] = torch.stack([qw2, qx2, qy2, qz2], dim=-1)
    
    # Case 3: m11 > m22
    mask3 = ~mask1 & ~mask2 & (matrix[:, 1, 1] > matrix[:, 2, 2])
    s3 = torch.sqrt(1.0 + matrix[mask3, 1, 1] - matrix[mask3, 0, 0] - matrix[mask3, 2, 2]) * 2
    qw3 = (matrix[mask3, 0, 2] - matrix[mask3, 2, 0]) / s3
    qx3 = (matrix[mask3, 0, 1] + matrix[mask3, 1, 0]) / s3
    qy3 = 0.25 * s3
    qz3 = (matrix[mask3, 1, 2] + matrix[mask3, 2, 1]) / s3
    quat[mask3] = torch.stack([qw3, qx3, qy3, qz3], dim=-1)
    
    # Case 4: else
    mask4 = ~mask1 & ~mask2 & ~mask3
    s4 = torch.sqrt(1.0 + matrix[mask4, 2, 2] - matrix[mask4, 0, 0] - matrix[mask4, 1, 1]) * 2
    qw4 = (matrix[mask4, 1, 0] - matrix[mask4, 0, 1]) / s4
    qx4 = (matrix[mask4, 0, 2] + matrix[mask4, 2, 0]) / s4
    qy4 = (matrix[mask4, 1, 2] + matrix[mask4, 2, 1]) / s4
    qz4 = 0.25 * s4
    quat[mask4] = torch.stack([qw4, qx4, qy4, qz4], dim=-1)
    
    return quat.view(*batch_shape, 4)


def matrix_to_rotation_6d(matrix):
    """
    Convert rotation matrix to 6D rotation representation.
    Args:
        matrix: tensor of shape [..., 3, 3] - rotation matrices
    Returns:
        6D rotation representation of shape [..., 6] (first two columns of rotation matrix)
    """
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def quaternion_to_matrix(quat):
    """
    Convert quaternion to rotation matrix.
    Args:
        quat: tensor of shape [..., 4] - quaternion (w, x, y, z)
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    batch_shape = quat.shape[:-1]
    quat = quat.view(-1, 4)
    
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Compute rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    matrix = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)
    
    return matrix.view(*batch_shape, 3, 3)


def rotation_6d_to_matrix(rotation_6d):
    """
    Convert 6D rotation representation to rotation matrix using Gram-Schmidt orthogonalization.
    Args:
        rotation_6d: tensor of shape [..., 6] - 6D rotation representation
    Returns:
        rotation matrix of shape [..., 3, 3]
    """
    batch_shape = rotation_6d.shape[:-1]
    rotation_6d = rotation_6d.view(-1, 6)
    
    # Extract first two columns
    col1 = rotation_6d[:, :3]
    col2 = rotation_6d[:, 3:]
    
    # Normalize first column
    col1 = col1 / torch.norm(col1, dim=-1, keepdim=True)
    
    # Gram-Schmidt orthogonalization for second column
    col2 = col2 - torch.sum(col1 * col2, dim=-1, keepdim=True) * col1
    col2 = col2 / torch.norm(col2, dim=-1, keepdim=True)
    
    # Third column is cross product
    col3 = torch.cross(col1, col2, dim=-1)
    
    # Stack columns to form rotation matrix
    matrix = torch.stack([col1, col2, col3], dim=-1)
    
    return matrix.view(*batch_shape, 3, 3)


def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 4)
    :param y: quaternion tensor (N, S, J, 4)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """
    len = torch.sum(x * y, dim=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y

    return res
