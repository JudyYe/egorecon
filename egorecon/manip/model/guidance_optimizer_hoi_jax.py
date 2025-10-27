"""Optimize SE3 trajectories using Levenberg-Marquardt."""

from __future__ import annotations
from jaxlie import SO3

from fire import Fire
import os
import pickle

# Need to play nice with PyTorch!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from functools import partial
from typing import Literal, cast

import jax
import jax_dataclasses as jdc
import jaxls
import numpy as np
import numpy as onp
import torch
from jax import numpy as jnp
from jutils import geom_utils, mesh_utils, plot_utils

# from jaxtyping import Float  # Not used in this minimal implementation
from torch import Tensor
from typing_extensions import assert_never

from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer
from egorecon.visualization.rerun_visualizer import RerunVisualizer
import jaxlie



class _SmplhSingleHandPosesVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.concatenate(
        [jnp.ones((15, 1)), jnp.zeros((15, 3))], axis=-1
    ),
    retract_fn=lambda val, delta: (
        jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(15, 3))
    ).wxyz,
    tangent_dim=15 * 3,
):
    """Variable containing local joint poses for one hand of a SMPL-H human."""


class _SE3TrajectoryVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ),  # Identity quaternion + zero translation
    retract_fn=lambda val, delta: (
        jaxlie.SE3(val) @ jaxlie.SE3.exp(delta)
    ).wxyz_xyz,  # SE3 retraction
    tangent_dim=6,  # 6D tangent space (3 for translation + 3 for rotation)
):
    """Variable containing SE3 trajectory as wxyz_xyz format [qw, qx, qy, qz, x, y, z]."""


def se3_to_wxyz_xyz(se3):
    tsl, rot6d = torch.split(se3, [3, 6], dim=-1)
    mat = geom_utils.rotation_6d_to_matrix(rot6d)
    wxyz = geom_utils.rot_cvt.matrix_to_quaternion(mat)

    return torch.cat([wxyz, tsl], dim=-1)

def wxyz_xyz_to_se3(wxyz_xyz):
    wxyz, tsl = torch.split(wxyz_xyz, [4, 3], dim=-1)
    mat = geom_utils.rot_cvt.quaternion_to_matrix(wxyz)
    return geom_utils.matrix_to_se3_v2(geom_utils.rt_to_homo(mat, tsl))


from . import fncmano_jax
from pathlib import Path
from egorecon.utils.motion_repr import HandWrapper


def do_forward_kinematics_mano(
    mano_model: fncmano_jax.MANOModel,
    global_orient: jnp.ndarray,  # (T, 3)
    transl: jnp.ndarray,  # (T, 3)
    hand_pose: jnp.ndarray,  # (T, 15) - PCA components
    betas: jnp.ndarray,  # (T, 10) or (1, 10)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform forward kinematics on MANO model.
    
    Args:
        mano_model: MANO model
        global_orient: Root rotation in axis-angle (..., 3)
        transl: Translation vector (..., 3)
        hand_pose: Hand pose in PCA space (..., 15)
        betas: Shape parameters (..., 10)
        
    Returns:
        vertices: (..., 778, 3) mesh vertices
        joints: (..., 16, 3) joint positions
        faces: (F, 3) face indices
    """
    # Apply shape
    shaped = mano_model.with_shape(betas)
    
    # Convert to quaternions - assume hand_pose is already rotation vectors

    # hand_quats = SO3.exp(hand_pose).wxyz
    
    # Apply pose
    posed = shaped.with_pose(
        global_orient=global_orient,
        transl=transl,
        # local_quats=hand_quats,
        pca=hand_pose,
        add_mean=True,
    )
    
    # Apply LBS
    mesh = posed.lbs()
    
    return mesh.verts, mesh.joints, mesh.faces


def draw_hand_joints_with_numbers(joints, name, verts, faces):
    ## plot them, display numbers {i} besides joints

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], joint[2], f"{i}")
        ax.scatter(joint[0], joint[1], joint[2], color='red')

    # draw meshes in the same figure using matplotlib
    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()

    # plt.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], faces)
    
    plt.show()
    plt.savefig(f"outputs/hand_joints_{name}.png")
    
    

from jutils import hand_utils, model_utils
def test_fk():
    save = pickle.load(open("outputs/tmp.pkl", "rb"))
    batch = save["sample"]
    batch = model_utils.to_cuda(batch, "cpu")

    device = batch["right_hand_params"].device
    hand_wrapper = HandWrapper(Path("assets/mano"))
    wrapper = hand_utils.ManopthWrapper().to(device)

    # golden standard
    # right_params = batch["right_hand_params"]  # (BS, T, 3+3+15+10)
    side = "left"
    right_params = batch[f"{side}_hand_params"] 
    B, T, D = right_params.shape
    # Parse parameters
    global_orient = right_params[:, :, :3]
    transl = right_params[:, :, 3:6]
    hand_pose = right_params[:, :, 6:21]
    pca = right_params[:, :, 6:21]
    hA = wrapper.pca_to_pose(pca[0])    # (BS, T, 45)
    hA = hA.reshape(B, T, 15, 3)
    betas = right_params[:, :, 21:31]

    # PyTorch forward pass
    right_verts, right_faces, right_joints = hand_wrapper.hand_para2verts_faces_joints(
        right_params[0, 0:1], side=side, my_order=True,
    )
    draw_hand_joints_with_numbers(right_joints.reshape(21, 3), 'smplx', right_verts, right_faces)  # (21, 3)

    # my_meshes, right_joints_my = wrapper(None, hA[0, 0:1].reshape(1, 45), global_orient[0, 0:1], transl[0, 0:1], th_betas=betas[0, 0:1])
    # print(right_joints_my - right_joints)
    # print(right_verts - my_meshes.verts_padded())
    # draw_hand_joints_with_numbers(right_joints_my.reshape(21, 3), 'my', my_meshes.verts_padded(), my_meshes.faces_padded())  # (21, 3)
    
    # Load JAX MANO model
    mano_model_jax = fncmano_jax.MANOModel.load(
        mano_dir=Path("assets/mano"),
        side=side,
        )
    
    # Convert to numpy
    global_orient_np = global_orient[0].numpy()
    transl_np = transl[0].numpy()
    hand_pose_np = hand_pose[0].numpy()
    betas_np = betas[0].numpy()

    # JAX forward pass
    # vmap over T-axis
    jax_verts, jax_joints, jax_faces = do_forward_kinematics_mano(
        mano_model_jax,
        global_orient=global_orient_np[0],
        transl=transl_np[0],
        hand_pose=hand_pose_np[0],
        betas=betas_np[0],
    )
    
    # Compare results
    print(f"PyTorch vertices shape: {right_verts.shape}")
    print(f"JAX vertices shape: {jax_verts.shape}")
    print(f"PyTorch joints shape: {right_joints.shape}")
    print(f"JAX joints shape: {jax_joints.shape}")
    
    # Check if results are close
    vert_diff = np.abs(right_verts.numpy()[0] - jax_verts)
    joint_diff = np.abs(right_joints.numpy()[0] - jax_joints)
    
    print(f"\nMax vertex difference: {vert_diff.max():.6f}")
    print(f"Mean vertex difference: {vert_diff.mean():.6f}")
    print(f"Max joint difference: {joint_diff.max():.6f}")
    print(f"Mean joint difference: {joint_diff.mean():.6f}")
    



if __name__ == "__main__":
    # Fire(test_optimization)
    Fire(test_fk)
