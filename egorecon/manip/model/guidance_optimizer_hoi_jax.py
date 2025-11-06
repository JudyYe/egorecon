"""Optimize SE3+hands trajectories using Levenberg-Marquardt."""

from __future__ import annotations

from fire import Fire
from pathlib import Path
from omegaconf import OmegaConf
from jutils import model_utils
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
from . import fncmano_jax

ndc = True


def se3_to_wxyz_xyz(se3):
    tsl, rot6d = torch.split(se3, [3, 6], dim=-1)
    mat = geom_utils.rotation_6d_to_matrix(rot6d)
    wxyz = geom_utils.rot_cvt.matrix_to_quaternion(mat)

    return torch.cat([wxyz, tsl], dim=-1)


def wxyz_xyz_to_se3(wxyz_xyz):
    wxyz, tsl = torch.split(wxyz_xyz, [4, 3], dim=-1)
    mat = geom_utils.rot_cvt.quaternion_to_matrix(wxyz)
    return geom_utils.matrix_to_se3_v2(geom_utils.rt_to_homo(mat, tsl))


# Modes for guidance
GuidanceMode = Literal[
    "fp", 
    "hoi_contact",  # reprojection + HOI contact loss
    "hoi_fp",  # reprojection + HOI FP loss
    "debug",
    "hand_only",
    "object_only",
    "contact_only",
    "fp_simple",
    "fp_full",
]


@jdc.pytree_dataclass
class JaxGuidanceParams:
    # Smoothness weights
    use_fp_rot: jdc.Static[bool] = False
    fp_rot_weight: float = 1e-3

    use_fp: jdc.Static[bool] = False
    fp_weight: float = 0.001

    use_j3d: jdc.Static[bool] = False
    j3d_weight: float = 10.0

    use_j2d: jdc.Static[bool] = True
    j2d_weight: float = 10.0

    use_contact_obs: jdc.Static[bool] = True
    contact_obs_weight: float = 1

    use_static: jdc.Static[bool] = True
    static_weight: float = 1.0

    use_obj_smoothness: jdc.Static[bool] = True
    obj_vel_weight: float = 1.0
    obj_acc_weight: float = 1.0

    use_hand_smoothness: jdc.Static[bool] = True
    hand_vel_weight: float = 1.0
    hand_acc_weight: float = 1.0

    use_delta_wTo: jdc.Static[bool] = True
    delta_wTo_weight: float = 1.0

    # Reprojection cost weights
    reproj_weight: float = 1.0
    use_reproj_cd: jdc.Static[bool] = True
    use_reproj_cd_one_way: jdc.Static[bool] = False

    use_kp2d: jdc.Static[bool] = False
    kp2d_weight: float = 1.0

    use_reproj_com: jdc.Static[bool] = False

    use_abs_contact: jdc.Static[bool] = True
    use_rel_contact: jdc.Static[bool] = True
    contact_weight: float = 1.0
    rel_contact_weight: float = 1.0

    use_hand_local: jdc.Static[bool] = True
    hand_local_weight: float = 1.0

    # smoothness weights
    tsl_weight: float = 1.0
    quat_weight: float = 1.0

    contact_th: float = 0.5

    # Optimization parameters
    lambda_initial: float = .1 # 1e4  # Increased for better convergence
    # lambda_initial: float = 1e-4  
    step_quality_min: float = 1e-4
    # step_quality_min: float = 1e-5  
    max_iters: jdc.Static[int] = 50  # Increased for better convergence

    @staticmethod
    def defaults(
        mode: GuidanceMode,
        phase: Literal["inner", "post", "fp", "inner-init"],
    ) -> JaxGuidanceParams:
        if mode == "hoi_contact":
            if phase == "inner":
                params = JaxGuidanceParams( 
                    use_j2d=True,
                    j2d_weight=20.,
                    use_hand_local=True,
                    hand_local_weight=.1,
                    use_hand_smoothness=False,
                    use_j3d=False,
                    use_contact_obs=True,
                    use_static=True,
                    static_weight=1.,
                    use_reproj_com=False,
                    use_reproj_cd=False,
                    use_reproj_cd_one_way=True,
                    use_abs_contact=False,
                    use_rel_contact=False,
                    use_obj_smoothness=False,
                    use_delta_wTo=False,
                    max_iters=5,
                    lambda_initial=1e-2,
                )                

            elif phase == "post":
                params = JaxGuidanceParams( 
                    use_j2d=True,
                    j2d_weight=20.,
                    use_hand_smoothness=True,
                    hand_acc_weight=1.,
                    use_j3d=False,
                    use_contact_obs=True,
                    use_static=True,
                    static_weight=100.,
                    use_reproj_com=False,
                    use_reproj_cd=False,
                    use_reproj_cd_one_way=True,
                    reproj_weight=10.,
                    use_abs_contact=True,
                    use_rel_contact=True,
                    rel_contact_weight=1,
                    use_obj_smoothness=True,
                    use_delta_wTo=True,
                    max_iters=50,
                )         
            return params        
        elif mode == "hoi_fp":
            if phase == "inner":
                params = JaxGuidanceParams( 
                    use_fp=True,
                    use_j2d=True,
                    j2d_weight=20.,
                    use_hand_local=True,
                    hand_local_weight=.1,
                    use_hand_smoothness=False,
                    use_j3d=False,
                    use_contact_obs=True,
                    use_static=True,
                    static_weight=1.,
                    use_reproj_com=False,
                    use_reproj_cd=False,
                    use_reproj_cd_one_way=True,
                    use_abs_contact=False,
                    use_rel_contact=False,
                    use_obj_smoothness=False,
                    use_delta_wTo=False,
                    max_iters=5,
                    lambda_initial=1e-2,
                )                

            elif phase == "post":
                params = JaxGuidanceParams( 
                    use_fp=True,
                    use_j2d=True,
                    j2d_weight=20.,
                    use_hand_smoothness=True,
                    hand_acc_weight=1.,
                    use_j3d=False,
                    use_contact_obs=True,
                    use_static=True,
                    static_weight=100.,
                    use_reproj_com=False,
                    use_reproj_cd=False,
                    use_reproj_cd_one_way=True,
                    reproj_weight=10.,
                    use_abs_contact=True,
                    use_rel_contact=True,
                    rel_contact_weight=1,
                    use_obj_smoothness=True,
                    use_delta_wTo=True,
                    max_iters=50,
                )         
            return params        
        if mode == "fp_simple":
            return JaxGuidanceParams( 
                    use_j2d=True,
                    j2d_weight=1.,
                    use_hand_smoothness=True,
                    hand_acc_weight=1.,
                    use_j3d=False,
                    use_contact_obs=False,
                    use_static=False,
                    static_weight=.1,
                    use_reproj_com=False,
                    use_reproj_cd=False,
                    use_reproj_cd_one_way=True,
                    reproj_weight=100.,
                    use_abs_contact=False,
                    use_rel_contact=False,
                    rel_contact_weight=1,
                    use_obj_smoothness=True,
                    use_delta_wTo=True,
                    max_iters=50,
                )                     
        if mode == "fp_full":
            return JaxGuidanceParams( 
                    use_j2d=True,
                    j2d_weight=1.,
                    use_hand_smoothness=True,
                    hand_acc_weight=1.,
                    use_j3d=False,
                    use_contact_obs=True,
                    use_static=True,
                    static_weight=.1,
                    use_reproj_com=False,
                    use_reproj_cd=False,
                    use_reproj_cd_one_way=True,
                    reproj_weight=100.,
                    use_abs_contact=True,
                    use_rel_contact=True,
                    rel_contact_weight=1,
                    use_obj_smoothness=True,
                    use_delta_wTo=True,
                    max_iters=50,
                )                      
            
                              
        if mode == "debug":
            params = JaxGuidanceParams( 
                use_fp_rot=True,
                use_fp=True,
                use_j2d=False,
                j2d_weight=20.,
                use_hand_local=False,
                hand_local_weight=.1,
                use_hand_smoothness=False,
                use_j3d=False,
                use_contact_obs=False,
                use_static=False,
                static_weight=1.,
                use_reproj_com=False,
                use_reproj_cd=False,
                use_reproj_cd_one_way=True,
                use_abs_contact=False,
                use_rel_contact=False,
                use_obj_smoothness=False,
                use_delta_wTo=False,
                max_iters=5,
                lambda_initial=1e-2,
            )          
            return params      

        if mode == "contact_only":
            max_iters = 5 if phase == "inner" else 50
            return JaxGuidanceParams(
                use_j2d=False,
                use_hand_smoothness=True,
                use_j3d=False,
                use_contact_obs=False,
                use_static=False,
                use_reproj_com=True,
                use_reproj_cd=False,
                use_abs_contact=False,
                use_rel_contact=False,
                use_obj_smoothness=True,
                max_iters=max_iters,
            )            
        elif mode == "object_only":
            max_iters = 5 if phase == "inner" else 50
            return JaxGuidanceParams(
                use_j2d=False,
                use_j3d=False,
                use_contact_obs=False,
                use_static=False,
                use_reproj_com=False,
                use_reproj_cd=True,
                use_abs_contact=False,
                use_rel_contact=False,
                use_obj_smoothness=False,
                max_iters=max_iters,
            )            
        elif mode == "hand_only":
            max_iters = 5 if phase == "inner" else 50
            return JaxGuidanceParams(
                use_j2d=True,
                use_hand_smoothness=False,
                use_j3d=False,
                use_contact_obs=False,
                use_static=False,
                use_reproj_com=False,
                use_reproj_cd=False,
                use_abs_contact=False,
                use_rel_contact=False,
                use_obj_smoothness=False,
                max_iters=max_iters,
            )

                
        else:
            assert_never(mode)


class _ContactVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros((2,)),  # left and right hand contact flags
    retract_fn=lambda val, delta: val + delta,  # Simple addition for free parameters
    tangent_dim=2,
):
    """Variable containing contact flags."""


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


class _LeftHandParamsVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros(
        (31,)
    ),  # global_orient(3) + transl(3) + hand_pose(15) + betas(10)
    retract_fn=lambda val, delta: val + delta,  # Simple addition for free parameters
    tangent_dim=31,
):
    """Variable containing hand parameters."""


class _RightHandParamsVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros(
        (31,)
    ),  # global_orient(3) + transl(3) + hand_pose(15) + betas(10)
    retract_fn=lambda val, delta: val + delta,  # Simple addition for free parameters
    tangent_dim=31,
):
    """Variable containing hand parameters."""


def do_guidance_optimization(
    pred_dict: dict,
    obs: dict,
    left_mano_model: fncmano_jax.MANOModel,
    right_mano_model: fncmano_jax.MANOModel,
    guidance_mode: GuidanceMode,
    phase: Literal["inner", "post"],
    verbose: bool,
    guidance_params: JaxGuidanceParams = None,
) -> tuple[Tensor, dict]:
    """Run an optimizer to optimize SE3 trajectory.

    Args:
        pred_dict: Dictionary containing predictions, e.g., {'wTo': wTo_pred, 'left_hand_params': left_hand_params, 'right_hand_params': right_hand_params, 'contact': contact}
        obs: Dictionary containing observations/targets, e.g., {'x': gt_trajectory}
        guidance_mode: Guidance mode
        phase: Optimization phase
        verbose: Whether to print optimization progress

    Returns:
        Tuple of (optimized_trajectory, debug_info)
    """
    start_time = time.time()
    if guidance_params is None:
        guidance_params = JaxGuidanceParams.defaults(guidance_mode, phase)

    # Extract prediction
    wTo_pred = se3_to_wxyz_xyz(pred_dict["wTo"])
    left_hand_params = pred_dict["left_hand_params"]  # (B, T, 31)
    right_hand_params = pred_dict["right_hand_params"]  # (B, T, 31)
    device = wTo_pred.device

    # Convert to JAX arrays and optimize
    wTo_pred_jax = cast(jax.Array, wTo_pred.numpy(force=True))
    left_hand_params_jax = cast(jax.Array, left_hand_params.numpy(force=True))
    right_hand_params_jax = cast(jax.Array, right_hand_params.numpy(force=True))
    contact_jax = cast(jax.Array, pred_dict["contact"].numpy(force=True))

    # Extract observation
    obs_dict = {}
    keep_keys = [
        "newPoints",
        "wTc",
        "intr",
        "x2d",
        "j3d",
        "j2d",
        "contact",
        "x2d_vis",
        "j2d_vis",
        "kp3d",
        "kp2d",
        "kp2d_vis",
        "wTo",
        "wTo_vis",
    ]
    for key in keep_keys:
        if key in obs:
            obs_dict["target_" + key] = cast(jax.Array, obs[key].numpy(force=True))

    # Optimize using vmapped function
    optimized_traj, debug_info = _optimize_vmapped(
        wTo=wTo_pred_jax,
        left_hand_params=left_hand_params_jax,
        right_hand_params=right_hand_params_jax,
        contact=contact_jax,
        **obs_dict,
        left_mano_model=left_mano_model,
        right_mano_model=right_mano_model,
        guidance_params=guidance_params,
        verbose=verbose,
    )

    # Convert debug_info JAX arrays to Python floats (outside traced context)
    # Handle batch dimension from vmap
    debug_info_converted = {}
    for key, value in debug_info.items():
        if isinstance(value, jax.Array):
            # Materialize and convert to float
            value_np = onp.array(value)
            # If batched (from vmap), take first element or mean
            debug_info_converted[key] = value_np
        else:
            debug_info_converted[key] = value
    debug_info = debug_info_converted

    # Convert back to torch tensor
    wTo = wxyz_xyz_to_se3(torch.from_numpy(onp.array(optimized_traj["wTo"])).to(device))
    left_hand_params = torch.from_numpy(
        onp.array(optimized_traj["left_hand_params"])
    ).to(device)
    right_hand_params = torch.from_numpy(
        onp.array(optimized_traj["right_hand_params"])
    ).to(device)
    contact = torch.from_numpy(onp.array(optimized_traj["contact"])).to(device)

    optimized_traj_torch = {
        "wTo": wTo,
        "left_hand_params": left_hand_params,
        "right_hand_params": right_hand_params,
        "contact": contact,
    }

    print(f"SE3 trajectory optimization finished in {time.time() - start_time}sec")

    return optimized_traj_torch, debug_info


@jdc.jit
def _optimize_vmapped(
    wTo: jax.Array,
    left_hand_params: jax.Array,
    right_hand_params: jax.Array,
    contact: jax.Array,
    left_mano_model: fncmano_jax.MANOModel,
    right_mano_model: fncmano_jax.MANOModel,
    guidance_params: JaxGuidanceParams,
    verbose: jdc.Static[bool],
    target_newPoints: jax.Array = None,
    target_wTc: jax.Array = None,
    target_intr: jax.Array = None,
    target_x2d: jax.Array = None,
    target_j3d: jax.Array = None,
    target_j2d: jax.Array = None,
    target_contact: jax.Array = None,
    target_x2d_vis: jax.Array = None,
    target_j2d_vis: jax.Array = None,
    target_kp3d: jax.Array = None,
    target_kp2d: jax.Array = None,
    target_kp2d_vis: jax.Array = None,
    target_wTo: jax.Array = None,
    target_wTo_vis: jax.Array = None,
) -> tuple[dict, dict]:
    return jax.vmap(
        partial(
            _optimize,
            left_mano_model=left_mano_model,
            right_mano_model=right_mano_model,
            guidance_params=guidance_params,
            verbose=verbose,
        )
    )(
        wTo=wTo,
        left_hand_params=left_hand_params,
        right_hand_params=right_hand_params,
        contact=contact,
        target_newPoints=target_newPoints,
        target_wTc=target_wTc,
        target_intr=target_intr,
        target_x2d=target_x2d,
        target_j3d=target_j3d,
        target_j2d=target_j2d,
        target_contact=target_contact,
        target_x2d_vis=target_x2d_vis,
        target_j2d_vis=target_j2d_vis,
        target_kp3d=target_kp3d,
        target_kp2d=target_kp2d,
        target_kp2d_vis=target_kp2d_vis,
        target_wTo=target_wTo,
        target_wTo_vis=target_wTo_vis,
    )


def _optimize(
    wTo: jax.Array,
    left_hand_params: jax.Array,
    right_hand_params: jax.Array,
    contact: jax.Array,
    left_mano_model: fncmano_jax.MANOModel,
    right_mano_model: fncmano_jax.MANOModel,
    guidance_params: JaxGuidanceParams,
    verbose: jdc.Static[bool],
    target_newPoints: jax.Array = None,
    target_wTc: jax.Array = None,
    target_intr: jax.Array = None,
    target_x2d: jax.Array = None,
    target_j3d: jax.Array = None,
    target_j2d: jax.Array = None,
    target_contact: jax.Array = None,
    target_x2d_vis: jax.Array = None,
    target_j2d_vis: jax.Array = None,
    target_kp3d: jax.Array = None,
    target_kp2d: jax.Array = None,
    target_kp2d_vis: jax.Array = None,
    target_wTo: jax.Array = None,
    target_wTo_vis: jax.Array = None,
) -> jax.Array:
    timesteps = wTo.shape[0]
    assert wTo.shape == (timesteps, 7)
    assert target_intr.shape == (3, 3)
    assert target_newPoints.shape == (5000, 3)
    assert target_x2d.shape[2] == 2
    assert target_wTc.shape[-1] == 7


    left_betas = left_hand_params[..., -10:]
    right_betas = right_hand_params[..., -10:]
    left_shaped = left_mano_model.with_shape(jnp.mean(left_betas, axis=0))
    right_shaped = right_mano_model.with_shape(jnp.mean(right_betas, axis=0))

    initial_left_params = left_hand_params
    initial_right_params = right_hand_params

    initial_wTo = wTo


    def do_forward_kinematics_two_hands(
        vals: jaxls.VarValues,
        left_params: _LeftHandParamsVar,
        right_params: _RightHandParamsVar,
    ):
        """Helper to compute forward kinematics from variable values."""
        left_params = vals[left_params]
        right_params = vals[right_params]
        left_mesh = left_shaped.with_pose(
            global_orient=left_params[..., :3],
            transl=left_params[..., 3:6],
            pca=left_params[..., 6:21],
            add_mean=True,
        )

        right_mesh = right_shaped.with_pose(
            global_orient=right_params[..., :3],
            transl=right_params[..., 3:6],
            pca=right_params[..., 6:21],
            add_mean=True,
        )

        return left_mesh, right_mesh

    def dist_residual(weight, delta: jaxlie.SE3):
        res = delta.log()
        quat, tsl = res[:4], res[4:]
        res = weight * jnp.concatenate(
            [quat * guidance_params.quat_weight, tsl * guidance_params.tsl_weight],
            axis=0,
        )
        return res.flatten()

    init_quats = wTo

    # We'll populate a list of factors (cost terms)
    factors = list[jaxls.Cost]()

    def cost_with_args(*args):
        """Decorator for appending to the factor list."""

        def inner(cost_func):
            factors.append(jaxls.Cost(cost_func, args))
            return cost_func

        return inner

    # Create a single variable for the entire trajectory (much more efficient!)
    var_traj = _SE3TrajectoryVar(jnp.arange(timesteps))
    var_left_params = _LeftHandParamsVar(jnp.arange(timesteps))
    var_right_params = _RightHandParamsVar(jnp.arange(timesteps))
    var_contact = _ContactVar(jnp.arange(timesteps))

    if guidance_params.use_fp:
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps)),
            target_wTo,
            target_wTo_vis,
        )
        def fp_cost(
            vals: jaxls.VarValues,
            var_traj: _SE3TrajectoryVar,
            target_wTo: jax.Array,
            target_wTo_vis: jax.Array,
        ) -> jax.Array:
            wTo = vals[var_traj]
            wTo = jaxlie.SE3(wTo)

            wTo_target = jaxlie.SE3(target_wTo)

            res = dist_residual(guidance_params.fp_weight, wTo.inverse() @ wTo_target)
            res = res * target_wTo_vis.flatten()
            return res


    if guidance_params.use_fp_rot:
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps)),
            target_wTo,
            target_wTo_vis,
        )
        def fp_rot_cost(
            vals: jaxls.VarValues,
            var_traj: _SE3TrajectoryVar,
            target_wTo: jax.Array,
            target_wTo_vis: jax.Array,
        ) -> jax.Array:
            wTo = vals[var_traj]
            wTo = jaxlie.SE3(wTo)

            wTo_target = jaxlie.SE3(target_wTo)

            wTo_rot = wTo.rotation() 
            wTo_target_rot = wTo_target.rotation()

            delta = wTo_rot.inverse() @ wTo_target_rot

            res = target_wTo_vis * delta.log()
            
            res = guidance_params.fp_rot_weight * res.flatten()
            return res
            

    if guidance_params.use_j2d:
        @cost_with_args(
            _LeftHandParamsVar(jnp.arange(timesteps)),
            _RightHandParamsVar(jnp.arange(timesteps)),
            target_j2d,
            target_wTc,
            target_intr[None],
            target_j2d_vis,
        )
        def j2d_cost(
            vals: jaxls.VarValues,
            left_params: _LeftHandParamsVar,
            right_params: _RightHandParamsVar,
            target_j2d: jax.Array,
            target_wTc: jax.Array,
            target_intr: jax.Array,
            target_j2d_vis: jax.Array,
        ) -> jax.Array:
            left_posed, right_posed = do_forward_kinematics_two_hands(
                vals, left_params, right_params
            )
            left_joints = left_posed.joint_tip_transl
            right_joints = right_posed.joint_tip_transl

            joints_traj = jnp.concatenate([left_joints, right_joints], axis=0)
            wJoints_traj = joints_traj.reshape(-1, 3)

            j2d_pred = project_jax_pinhole(wJoints_traj, target_wTc, target_intr, ndc=ndc)
            diff = (j2d_pred - target_j2d) / j2d_pred.shape[0]  # (J, 2)
            diff = diff * target_j2d_vis.reshape(-1, 1)
            diff = guidance_params.j2d_weight * diff.flatten()
            return diff
       
    if guidance_params.use_contact_obs:

        @cost_with_args(
            _ContactVar(jnp.arange(timesteps)),
            target_contact,
        )
        def contact_obs_cost(
            vals: jaxls.VarValues,
            contact: _ContactVar,
            target_contact: jax.Array,
        ) -> jax.Array:
            contact = vals[contact]
            diff = contact - target_contact
            diff = guidance_params.contact_weight * diff.flatten()
            return diff

    if guidance_params.use_hand_local:
        @cost_with_args(
            _LeftHandParamsVar(jnp.arange(timesteps)),
            _RightHandParamsVar(jnp.arange(timesteps)),
            initial_left_params,
            initial_right_params,
        )
        def delta_hand_local_cost(
            vals: jaxls.VarValues,
            left_params: _LeftHandParamsVar,
            right_params: _RightHandParamsVar,
            initial_left_params: jax.Array,
            initial_right_params: jax.Array,
        ) -> jax.Array:
            left_params = vals[left_params]
            right_params = vals[right_params]
            
            left_delta = left_params[..., 6:] - initial_left_params[..., 6:]
            right_delta = right_params[..., 6:] - initial_right_params[..., 6:]
            diff = jnp.concatenate([left_delta, right_delta], axis=0)
            diff = guidance_params.hand_local_weight * diff.flatten()
            return diff

    if guidance_params.use_hand_smoothness:
    #     def hand_delta_smoothness_cost(
    #         vals: jaxls.VarValues,
    #         left_cur: _LeftHandParamsVar,
    #         left_next: _LeftHandParamsVar,
    #         right_cur: _RightHandParamsVar,
    #         right_next: _RightHandParamsVar,
    #     ) -> jax.Array:
    #         l_mesh_cur, r_mesh_cur = do_forward_kinematics_two_hands(
    #             vals, left_cur, right_cur
    #         )
    #         l_joints_cur = l_mesh_cur.Ts_world_joint[..., 4:7]
    #         r_joints_cur = r_mesh_cur.Ts_world_joint[..., 4:7]

    #         l_mesh_next, r_mesh_next = do_forward_kinematics_two_hands(
    #             vals, left_next, right_next
    #         )
    #         l_joints_next = l_mesh_next.Ts_world_joint[..., 4:7]
    #         r_joints_next = r_mesh_next.Ts_world_joint[..., 4:7]

    #         l_delta = l_joints_next - l_joints_cur
    #         r_delta = r_joints_next - r_joints_cur
    #         diff = jnp.concatenate([l_delta, r_delta], axis=0)
    #         diff = guidance_params.hand_vel_weight * diff.flatten()
    #         return diff

        @cost_with_args(
            _LeftHandParamsVar(jnp.arange(timesteps - 2)),
            _LeftHandParamsVar(jnp.arange(1, timesteps - 1)),
            _LeftHandParamsVar(jnp.arange(2, timesteps)),
            _RightHandParamsVar(jnp.arange(timesteps - 2)),
            _RightHandParamsVar(jnp.arange(1, timesteps - 1)),
            _RightHandParamsVar(jnp.arange(2, timesteps)),
        )
        def hand_vel_smoothness_cost(
            vals: jaxls.VarValues,
            left_t0: _LeftHandParamsVar,
            left_t1: _LeftHandParamsVar,
            left_t2: _LeftHandParamsVar,
            right_t0: _RightHandParamsVar,
            right_t1: _RightHandParamsVar,
            right_t2: _RightHandParamsVar,
        ) -> jax.Array:
            left_mesh_t0, right_mesh_t0 = do_forward_kinematics_two_hands(
                vals, left_t0, right_t0
            )
            left_joints_t0 = left_mesh_t0.joint_tip_transl
            right_joints_t0 = right_mesh_t0.joint_tip_transl

            left_mesh_t1, right_mesh_t1 = do_forward_kinematics_two_hands(
                vals, left_t1, right_t1
            )
            left_joints_t1 = left_mesh_t1.joint_tip_transl
            right_joints_t1 = right_mesh_t1.joint_tip_transl

            left_mesh_t2, right_mesh_t2 = do_forward_kinematics_two_hands(
                vals, left_t2, right_t2
            )
            left_joints_t2 = left_mesh_t2.joint_tip_transl
            right_joints_t2 = right_mesh_t2.joint_tip_transl

            left_vel_01 = left_joints_t1 - left_joints_t0
            right_vel_01 = right_joints_t1 - right_joints_t0
            left_vel_12 = left_joints_t2 - left_joints_t1
            right_vel_12 = right_joints_t2 - right_joints_t1

            left_acc_012 = left_vel_12 - left_vel_01
            right_acc_012 = right_vel_12 - right_vel_01

            diff = jnp.concatenate([left_acc_012, right_acc_012], axis=0)
            diff = guidance_params.hand_acc_weight * diff.flatten()
            return diff

    if guidance_params.use_delta_wTo:
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(0, timesteps)),
            initial_wTo,
        )
        def delta_wTo_cost(
            vals: jaxls.VarValues,
            wTo: _SE3TrajectoryVar,
            initial_wTo: jax.Array,
        ) -> jax.Array:
            ids = wTo.id
            wTo = jaxlie.SE3(vals[wTo])
            init_wTo = jaxlie.SE3(initial_wTo)
            mask = ids == 0
            diff = mask *dist_residual(
                guidance_params.delta_wTo_weight,
                wTo.inverse() @ init_wTo,
            )
            return diff.flatten()

    if guidance_params.use_obj_smoothness:
        # smoothness on
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps - 1)),
            _SE3TrajectoryVar(jnp.arange(1, timesteps)),
        )
        def delta_smoothness_cost(
            vals: jaxls.VarValues,
            current: _SE3TrajectoryVar,
            next: _SE3TrajectoryVar,
        ) -> jax.Array:
            dist2 = dist_residual(
                guidance_params.obj_vel_weight,
                jaxlie.SE3(vals[current]).inverse() @ jaxlie.SE3(vals[next]),
            )
            return jnp.concatenate([dist2])

        # smoothness on velocity
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps - 2)),
            _SE3TrajectoryVar(jnp.arange(1, timesteps - 1)),
            _SE3TrajectoryVar(jnp.arange(2, timesteps)),
        )
        def vel_smoothness_cost(
            vals: jaxls.VarValues,
            t0: _SE3TrajectoryVar,
            t1: _SE3TrajectoryVar,
            t2: _SE3TrajectoryVar,
        ) -> jax.Array:
            curdelt = jaxlie.SE3(vals[t0]).inverse() @ jaxlie.SE3(vals[t1])
            nexdelt = jaxlie.SE3(vals[t1]).inverse() @ jaxlie.SE3(vals[t2])
            return dist_residual(
                guidance_params.obj_acc_weight,
                curdelt.inverse() @ nexdelt,
            )

    if guidance_params.use_abs_contact:
        # find the first frame where contact is true
        # current_contact =

        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps)),
            _LeftHandParamsVar(jnp.arange(timesteps)),
            _RightHandParamsVar(jnp.arange(timesteps)),
            target_newPoints[None],
            target_contact,
        )
        def abs_contact_cost(
            vals: jaxls.VarValues,
            cur: _SE3TrajectoryVar,
            left_params: _LeftHandParamsVar,
            right_params: _RightHandParamsVar,
            target_newPoints: jax.Array,
            target_contact: jax.Array,
        ) -> jax.Array:
            left_posed, right_posed = do_forward_kinematics_two_hands(
                vals, left_params, right_params
            )
            left_joints = left_posed.joint_tip_transl
            right_joints = right_posed.joint_tip_transl
            joints_traj = jnp.concatenate([left_joints, right_joints], axis=0)

            current_traj = jaxlie.SE3(vals[cur])
            current_points = current_traj @ target_newPoints
            joints_traj = joints_traj.reshape(-1, 3)
            J = joints_traj.shape[0]

            dist, nn_idx, nn_points = knn_jax(joints_traj, current_points, k=1, )
            res = current_points[nn_idx.squeeze(1)] - joints_traj  # (J*2, 3)
            # print('res', res.shape)
            
            dist = dist.reshape(2, J // 2)
            res = res.reshape(2, J // 2, 3)
            dist = dist[..., -5:]  # finger tips  # (2, J//2)
            res = res[..., -5:, :]  # (2, J//2, 3)
            # print(res.shape)  # (2, 5, 3)
            
            # dist = dist.min(axis=-1, keepdims=True)  # (2, J//2) -> (2, 1)
            dist_idx = jnp.argmin(dist, axis=-1)  # (2, )
            # print('dist_idx', dist_idx.shape,)  
            # (2, 5, 3) -> (2, 1, 3)
            res_min = jnp.take_along_axis(res, dist_idx[..., None, None], axis=-2)  # (2, 1, 3)
            res_min = res_min.squeeze(1)  # (2, 3)

            # print('res_min', res_min.shape)
            
            res_min = residual_with_threshold(res_min, 0.005)  # half a cm
            contact_cost = res_min * guidance_params.contact_weight
            contact_cost = contact_cost.reshape(2, 3)
            D = contact_cost.shape[1]

            # dist = jnp.linalg.norm(
            #     current_points[:, None] - joints_traj[None, :, :], axis=-1
            # )  # (P, J)
            # dist = dist.reshape(-1, 2, J // 2)
            # contact_cost = (
            #     dist.min(axis=0) * guidance_params.contact_weight
            # )  # (2, J//2)
            # contact_cost = -jax.lax.top_k(-contact_cost, k=5, axis=-1)[0]
            # contact_cost = contact_cost.min(axis=-1, keepdims=True)

            # target_contact = jnp.min(dist, axis=-1) < guidance_params.contact_th
            # target_contact = jnp.array([0, 1])
            non_contact_cost = (
                jnp.zeros((2, D)) * guidance_params.contact_weight
            )  # (2,)

            cost = jnp.where(
                target_contact.reshape(2, 1), contact_cost, non_contact_cost
            )
            return cost.flatten()

    if guidance_params.use_rel_contact:
        # approximate contact label
        # smoothness on 
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps - 1)),
            _SE3TrajectoryVar(jnp.arange(1, timesteps)),
            _LeftHandParamsVar(jnp.arange(timesteps - 1)),
            _RightHandParamsVar(jnp.arange(timesteps - 1)),
            _LeftHandParamsVar(jnp.arange(1, timesteps)),
            _RightHandParamsVar(jnp.arange(1, timesteps)),
            target_newPoints[None],
            _ContactVar(jnp.arange(timesteps - 1)),
            _ContactVar(jnp.arange(1, timesteps)),
        )
        def relative_contact_cost(
            vals: jaxls.VarValues,
            wTo_cur: _SE3TrajectoryVar,  
            wTo_next: _SE3TrajectoryVar,
            left_params_cur: _LeftHandParamsVar,
            right_params_cur: _RightHandParamsVar,
            left_params_next: _LeftHandParamsVar,
            right_params_next: _RightHandParamsVar,
            target_newPoints: jax.Array,
            contact_cur: _ContactVar,
            contact_next: _ContactVar,
        ) -> jax.Array:
            wTo_cur = jaxlie.SE3(vals[wTo_cur])
            wTo_next = jaxlie.SE3(vals[wTo_next]) 
            contact_cur = vals[contact_cur]
            contact_next = vals[contact_next]
            # left_params_cur = vals[left_params_cur]
            # right_params_cur = vals[right_params_cur]
            # left_params_next = vals[left_params_next]
            # right_params_next = vals[right_params_next]

            # forward to get joints
            left_mesh, right_mesh = do_forward_kinematics_two_hands(
                vals, left_params_cur, right_params_cur
            )
            left_joints = left_mesh.joint_tip_transl
            right_joints = right_mesh.joint_tip_transl
            joints_traj_cur = jnp.concatenate([left_joints, right_joints], axis=0)
            joints_traj_cur = joints_traj_cur.reshape(-1, 3)
                
            left_mesh, right_mesh = do_forward_kinematics_two_hands(
                vals, left_params_next, right_params_next
            )
            left_joints = left_mesh.joint_tip_transl
            right_joints = right_mesh.joint_tip_transl
            joints_traj_next = jnp.concatenate([left_joints, right_joints], axis=0)
            joints_traj_next = joints_traj_next.reshape(-1, 3)

            current_points = wTo_cur @ target_newPoints
            next_points = wTo_next @ target_newPoints  # (P, 3)

            J = joints_traj_cur.shape[0]

            # minimal pairwise distance 
            dist_cur = jnp.linalg.norm(current_points[:, None] - joints_traj_cur[None, :, :], axis=-1)  # (P, J)

            # is_contact = (contact_cur < th) & (contact_next < th)  # (2,)
            # array((0, 1))
            is_contact = (contact_cur > 0.5) & (contact_next > 0.5)

            # p_near is the nearest point on the object to the current each joint
            p_near_ind = jnp.argmin(dist_cur, axis=0)  # (P, J) ->  (J,)
            p_near = current_points[p_near_ind] - joints_traj_cur # (J, 3)  # relative position to the joint
            p_near_next = next_points[p_near_ind] - joints_traj_next # (J, 3)  # relative position to the joint

            current_rot = wTo_cur.rotation()
            next_rot = wTo_next.rotation()

            res = p_near_next - next_rot @ current_rot.inverse() @ p_near

            # dist_prox = jnp.linalg.norm(res, axis=-1)    # (J,)
            dist_prox = res.reshape(2, J//2 * 3)
            D = dist_prox.shape[1]
            
            contact_cost = guidance_params.rel_contact_weight * dist_prox

            no_contact_cost = jnp.ones((2, D)) * 0 * guidance_params.rel_contact_weight  # (2, D)
            cost = jnp.where(is_contact.reshape(2, 1), contact_cost, no_contact_cost)

            # is_static = (((contact_cur <= 0.5) & (contact_cur <= 0.5)).sum(-1) >= 2) # (1, 2)
            # res = wTo_next.inverse() @  wTo_cur
            # res_cost = is_static * dist_residual(guidance_params.rel_contact_weight, res)

            # cost = jnp.concatenate([cost.flatten(), res_cost.flatten()])
            return cost.flatten()

    if guidance_params.use_static:
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps - 1)),
            _SE3TrajectoryVar(jnp.arange(1, timesteps)),
            _ContactVar(jnp.arange(timesteps - 1)),
            _ContactVar(jnp.arange(1, timesteps)),
            target_newPoints[None],
        )
        def static_cost(
            vals: jaxls.VarValues,
            wTo_cur: _SE3TrajectoryVar,
            wTo_next: _SE3TrajectoryVar,
            contact_cur: _ContactVar,
            contact_next: _ContactVar,
            target_newPoints: jax.Array,
        ) -> jax.Array:
            wTo_cur =   jaxlie.SE3(vals[wTo_cur])
            # wTo_next = jax.lax.stop_gradient(jaxlie.SE3(vals[wTo_next]))
            # contact_cur = jax.lax.stop_gradient(vals[contact_cur])
            # contact_next = jax.lax.stop_gradient(vals[contact_next])

            wTo_next = jaxlie.SE3(vals[wTo_next])
            contact_cur = vals[contact_cur]
            contact_next = vals[contact_next]

            # current_points = wTo_cur @ target_newPoints
            # next_points = wTo_next @ target_newPoints

            # wPoints_diff = (next_points - current_points)

            static_mask = (contact_cur < 0.5) | (contact_next < 0.5)
            static_mask = static_mask[0] & static_mask[1]
            static_cost = dist_residual(guidance_params.static_weight, wTo_next.inverse() @ wTo_cur)
            static_cost = static_cost * static_mask.flatten()
            # static_cost = guidance_params.static_weight * (static_mask * wPoints_diff).flatten()    
            return static_cost

    if guidance_params.use_reproj_com:
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps)),
            target_x2d,
            target_wTc,
            target_newPoints[None],
            target_intr[None],
            target_x2d_vis,
        )
        def reprojection_cost_com(
            vals: jaxls.VarValues,
            var_traj: _SE3TrajectoryVar,
            target_x2d: jax.Array,
            target_wTc: jax.Array,
            target_newPoints: jax.Array,
            target_intr: jax.Array,
            target_x2d_vis: jax.Array,
        ) -> jax.Array:
            """Reprojection cost: project object points and compare with 2D observations.
            but target_x2d is not in correspondence, you need to find the closest point in target_x2d to the projected point.
            """
            current_traj = vals[var_traj]  # (7, ) - wxyz_xyz format

            # inside (7,) (5000, 2) (7,) (5000, 3) (3, 3)
            # Project object points using current trajectory
            projected_2d = project_jax(
                target_newPoints, current_traj, target_wTc, target_intr, ndc=ndc
            )  # (P, 2)

            pred_com = projected_2d.mean(axis=0)  # (2,)
            target_com = target_x2d.mean(axis=0)  # (2,)
            diff = (pred_com - target_com) * target_x2d_vis.reshape(1)

            return guidance_params.reproj_weight * diff.flatten()


    if guidance_params.use_kp2d:
        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps)),
            target_kp2d,
            target_wTc,
            target_kp3d[None],
            target_intr[None],
            target_kp2d_vis,
        )
        def reprojection_cost_kp(
            vals: jaxls.VarValues,
            var_traj: _SE3TrajectoryVar,
            target_kp2d: jax.Array,
            target_wTc: jax.Array,
            target_kp3d: jax.Array,
            target_intr: jax.Array,
            target_kp2d_vis: jax.Array,
        ) -> jax.Array:
            """Reprojection cost: project object points and compare with 2D observations.
            but target_kp2d is not in correspondence, you need to find the closest point in target_kp2d to the projected point.
            """
            current_traj = vals[var_traj]  # (7, ) - wxyz_xyz format

            # inside (7,) (5000, 2) (7,) (5000, 3) (3, 3)
            # Project object points using current trajectory
            projected_2d = project_jax(
                target_kp3d, current_traj, target_wTc, target_intr, ndc=ndc
            )

            diff = projected_2d - target_kp2d  # (num_kp, 2)
            diff = diff * target_kp2d_vis.reshape(-1, 1)

            return guidance_params.kp2d_weight * diff.flatten()

    if guidance_params.use_reproj_cd or guidance_params.use_reproj_cd_one_way:
        assert (
            target_newPoints is not None
            and target_wTc is not None
            and target_intr is not None
            and target_x2d is not None
        )

        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps)),
            target_x2d,
            target_wTc,
            target_newPoints[None],
            target_intr[None],
            target_x2d_vis,
        )
        def reprojection_cost_cd(
            vals: jaxls.VarValues,
            var_traj: _SE3TrajectoryVar,
            target_x2d: jax.Array,
            target_wTc: jax.Array,
            target_newPoints: jax.Array,
            target_intr: jax.Array,
            target_x2d_vis: jax.Array,
        ) -> jax.Array:
            """Reprojection cost: project object points and compare with 2D observations.
            but target_x2d is not in correspondence, you need to find the closest point in target_x2d to the projected point.
            """
            current_traj = vals[var_traj]  # (7, ) - wxyz_xyz format

            # inside (7,) (5000, 2) (7,) (5000, 3) (3, 3)
            # Project object points using current trajectory
            projected_2d = project_jax(
                target_newPoints, current_traj, target_wTc, target_intr, ndc=ndc
            )
            P = projected_2d.shape[0]
            dist, nn_idx, x2d_nn = knn_jax(projected_2d, target_x2d, k=1)
            dist2, nn_idx2, oPoints_nn = knn_jax(target_x2d, projected_2d, k=1)

            dist1 = (projected_2d - x2d_nn.squeeze(1)) / projected_2d.shape[0]
            dist2 = (target_x2d - oPoints_nn.squeeze(1)) / target_x2d.shape[0]

            if guidance_params.use_reproj_cd_one_way:
                diff = dist2
            else:
                diff = jnp.concatenate([dist1, dist2], axis=0)  # residual  # (P1+P2, 2)
            diff = diff * target_x2d_vis.reshape(1, 1)

            return guidance_params.reproj_weight * diff.flatten()

            # Set up the optimization problem

    graph = jaxls.LeastSquaresProblem(
        costs=factors, variables=[var_traj, var_left_params, var_right_params, var_contact]
    ).analyze()

    # Solve using Levenberg-Marquardt
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                var_traj.with_value(wTo),
                var_left_params.with_value(left_hand_params),
                var_right_params.with_value(right_hand_params),
                var_contact.with_value(contact),
            ]
        ),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(
            lambda_initial=guidance_params.lambda_initial,
            step_quality_min=guidance_params.step_quality_min
        ),
        termination=jaxls.TerminationConfig(max_iterations=guidance_params.max_iters),
        verbose=verbose,
    )

    # Extract optimized trajectory
    optimized_traj = solutions[var_traj]
    optimized_left_params = solutions[var_left_params]
    optimized_right_params = solutions[var_right_params]
    optimized_contact = solutions[var_contact]



    # Compute cost values for all factors
    debug_info = get_cost_values(graph, solutions, timesteps, factors)
    return {
        "wTo": optimized_traj,
        "left_hand_params": optimized_left_params,
        "right_hand_params": optimized_right_params,
        "contact": optimized_contact,
    }, debug_info


def get_cost_values(problem, solution, timesteps, factors):
    """Get per-timestep cost values for all costs in the problem.
    
    Args:
        problem: The AnalyzedLeastSquaresProblem (result of jaxls.LeastSquaresProblem.analyze())
        solution: VarValues solution
        timesteps: Number of timesteps to pad to
        factors: Original list of Cost objects (before analysis)
        
    Returns:
        dict: {cost_name: cost_values} where cost_values is shape (timesteps,)
    """
    cost_dict = {}
    
    # Iterate over original costs (factors) to get per-timestep evaluation
    for cost in factors:
        cost_name = cost._get_name()
        
        # Get variables from the cost to determine timestep structure
        def get_variables_from_args(args):
            """Recursively extract Var objects from cost args."""
            variables = []
            if isinstance(args, (list, tuple)):
                for arg in args:
                    variables.extend(get_variables_from_args(arg))
            elif isinstance(args, jaxls.Var):
                variables.append(args)
            return variables
        
        # Extract variables from cost args
        cost_vars = get_variables_from_args(cost.args)
        
        # Detect cost type by examining timestep variable IDs
        # Note: We check shapes and array identity (not values) to avoid traced boolean conversions
        timestep_vars_info = []
        for var in cost_vars:
            if isinstance(var.id, jax.Array) and var.id.ndim > 0 and var.id.shape[0] > 1:
                timestep_vars_info.append((var.id, var))
        
        # Determine cost type based on timestep variable structure
        # Use shape and array identity checks (not value comparisons) to avoid traced booleans
        if len(timestep_vars_info) == 0:
            cost_type = "standard"
        elif len(timestep_vars_info) == 1:
            cost_type = "standard"
        elif len(timestep_vars_info) == 2:
            ids1, var1 = timestep_vars_info[0]
            ids2, var2 = timestep_vars_info[1]
            # Check if same variable (identity check) or same shape
            if var1 is var2 or ids1.shape[0] == ids2.shape[0]:
                # Same variable or same shape - check if they're actually the same object
                if var1 is var2:
                    cost_type = "standard"
                else:
                    # Different variables with same shape - likely first-order offset
                    # We can't check values in JIT, so assume first-order if different objects
                    cost_type = "first_order"
            else:
                # Different shapes - first-order
                cost_type = "first_order"
        elif len(timestep_vars_info) >= 3:
            ids1, var1 = timestep_vars_info[0]
            ids2, var2 = timestep_vars_info[1]
            ids3, var3 = timestep_vars_info[2]
            # Check if all same variable or all same shape
            if var1 is var2 is var3:
                cost_type = "standard"
            elif ids1.shape[0] == ids2.shape[0] == ids3.shape[0]:
                # Same shape but different variables - likely second-order offset
                cost_type = "second_order"
            else:
                # Different shapes - second-order
                cost_type = "second_order"
        else:
            cost_type = "standard"
        
        # Helper to evaluate cost at a single timestep (for standard costs)
        def eval_cost_standard(t: int):
            # Create VarValues for this timestep by extracting per-timestep variables
            timestep_vars = []
            timestep_args = []
            
            def extract_timestep_args(args, t_val):
                """Recursively extract timestep-specific args."""
                if isinstance(args, (list, tuple)):
                    return type(args)(extract_timestep_args(arg, t_val) for arg in args)
                elif isinstance(args, jaxls.Var):
                    if isinstance(args.id, jax.Array) and args.id.ndim > 0 and args.id.shape[0] > 1:
                        timestep_var = type(args)(t_val)
                        timestep_vars.append(timestep_var.with_value(solution[args][t_val]))
                        return timestep_var
                    else:
                        timestep_vars.append(args.with_value(solution[args]))
                        return args
                elif isinstance(args, jax.Array) and args.ndim > 1:
                    return args[t_val]
                else:
                    return args
            
            timestep_args = extract_timestep_args(cost.args, t)
            timestep_vals = jaxls.VarValues.make(timestep_vars)
            
            result = cost.compute_residual(timestep_vals, *timestep_args)
            if isinstance(result, tuple):
                residual = result[0]
            else:
                residual = result
            return jnp.sum(residual ** 2)
        
        # Helper to evaluate first-order cost at timestep pair (t1, t2)
        def eval_cost_first_order(i: int):
            t1, t2 = i, i + 1
            timestep_vars = []
            
            def extract_timestep_args_first_order(args, var_idx_ref):
                """Extract args with offset timesteps for first-order."""
                if isinstance(args, (list, tuple)):
                    return type(args)(extract_timestep_args_first_order(arg, var_idx_ref) for arg in args)
                elif isinstance(args, jaxls.Var):
                    if isinstance(args.id, jax.Array) and args.id.ndim > 0 and args.id.shape[0] > 1:
                        var_idx_ref[0] += 1
                        t_use = t1 if var_idx_ref[0] == 1 else t2
                        timestep_var = type(args)(t_use)
                        timestep_vars.append(timestep_var.with_value(solution[args][t_use]))
                        return timestep_var
                    else:
                        timestep_vars.append(args.with_value(solution[args]))
                        return args
                elif isinstance(args, jax.Array) and args.ndim > 1:
                    return args[t1]
                else:
                    return args
            
            var_idx_ref = [0]
            timestep_args = extract_timestep_args_first_order(cost.args, var_idx_ref)
            timestep_vals = jaxls.VarValues.make(timestep_vars)
            
            result = cost.compute_residual(timestep_vals, *timestep_args)
            if isinstance(result, tuple):
                residual = result[0]
            else:
                residual = result
            return jnp.sum(residual ** 2)
        
        # Helper to evaluate second-order cost at timestep triplet (t1, t2, t3)
        def eval_cost_second_order(i: int):
            t1, t2, t3 = i, i + 1, i + 2
            timestep_vars = []
            
            def extract_timestep_args_second_order(args, var_idx_ref):
                """Extract args with offset timesteps for second-order."""
                if isinstance(args, (list, tuple)):
                    return type(args)(extract_timestep_args_second_order(arg, var_idx_ref) for arg in args)
                elif isinstance(args, jaxls.Var):
                    if isinstance(args.id, jax.Array) and args.id.ndim > 0 and args.id.shape[0] > 1:
                        var_idx_ref[0] += 1
                        if var_idx_ref[0] == 1:
                            t_use = t1
                        elif var_idx_ref[0] == 2:
                            t_use = t2
                        else:
                            t_use = t3
                        timestep_var = type(args)(t_use)
                        timestep_vars.append(timestep_var.with_value(solution[args][t_use]))
                        return timestep_var
                    else:
                        timestep_vars.append(args.with_value(solution[args]))
                        return args
                elif isinstance(args, jax.Array) and args.ndim > 1:
                    return args[t1]
                else:
                    return args
            
            var_idx_ref = [0]
            timestep_args = extract_timestep_args_second_order(cost.args, var_idx_ref)
            timestep_vals = jaxls.VarValues.make(timestep_vars)
            
            result = cost.compute_residual(timestep_vals, *timestep_args)
            if isinstance(result, tuple):
                residual = result[0]
            else:
                residual = result
            return jnp.sum(residual ** 2)
        
        # Evaluate based on cost type and pad to T timesteps
        if cost_type == "standard":
            # Standard: t=0:T
            cost_values = jax.vmap(eval_cost_standard)(jnp.arange(timesteps))  # (T,)
            cost_values_per_timestep = cost_values
        elif cost_type == "first_order":
            # First-order: (t1=0:T-1, t2=1:T) -> T-1 values
            cost_values = jax.vmap(eval_cost_first_order)(jnp.arange(timesteps - 1))  # (T-1,)
            # Pad with 0 at the end
            cost_values_per_timestep = jnp.concatenate([cost_values, jnp.array([0.0])])  # (T,)
        elif cost_type == "second_order":
            # Second-order: (t1=0:T-2, t2=1:T-1, t3=2:T) -> T-2 values
            cost_values = jax.vmap(eval_cost_second_order)(jnp.arange(timesteps - 2))  # (T-2,)
            # Pad with 0 at the end
            cost_values_per_timestep = jnp.concatenate([cost_values, jnp.zeros(2)])  # (T,)
        else:
            # Fallback to standard
            cost_values = jax.vmap(eval_cost_standard)(jnp.arange(timesteps))
            cost_values_per_timestep = cost_values
        
        cost_dict[cost_name] = cost_values_per_timestep
    
    return cost_dict


def knn_jax(xPoints, yPoints, k=1):
    """JAX version of k-nearest neighbor.
    Args:
        xPoints: Points to find nearest neighbors for (P, D)
        yPoints: Points to find nearest neighbors from (Q, D)
        k: Number of nearest neighbors to find
    Returns:
        dist: Distance to nearest neighbors (P, k)
        nn_idx: Index of nearest neighbors (P, k)
        yPoints_nn: Nearest neighbors (P, k, D)
    """
    # Compute pairwise squared distances: (P, Q)
    # xPoints: (P, D), yPoints: (Q, D)
    # distances[i, j] = ||xPoints[i] - yPoints[j]||^2
    distances = jnp.sum(
        (xPoints[:, None, :] - yPoints[None, :, :]) ** 2, axis=-1
    )  # (P, Q)

    # Find k nearest neighbors using top_k (more efficient than argsort)
    # Note: top_k returns largest values, so we need to negate distances to get smallest
    if k == 1:
        nn_idx = jnp.argmin(distances, axis=1, keepdims=True)  # (P, 1)
        dist = jnp.min(distances, axis=1, keepdims=True)  # (P, 1)
    else:
        dist, nn_idx = jax.lax.top_k(-distances, k)  # (P, k)
        dist = -dist  # Convert back to positive distances

    # Get the actual nearest neighbor points
    yPoints_nn = yPoints[nn_idx]  # (P, k, D)

    return dist, nn_idx, yPoints_nn


def project_jax_pinhole(wJoints, wTc, intr, ndc=False, H=None, W=None):
    """JAX version of projection function for optimization.
    Args:
        wJoints: World joints (J, 3)
        wTc: Camera pose in world frame (7, ) - wxyz_xyz format
        intr: Camera intrinsics (3, 3)
    Returns:
        Projected 2D points (J, 2)
    """
    wTc_se3 = jaxlie.SE3(wTc)
    cTw_se3 = wTc_se3.inverse()
    
    cJoints = cTw_se3 @ wJoints

    iJoints = cJoints @ intr.T
    iJoints = iJoints[:, :2] / iJoints[:, 2:3]
    if ndc:
        if H is None:
            H = intr[0, 2] * 2
        if W is None:
            W = intr[1, 2] * 2
        x = iJoints[:, 0] / W * 2 - 1
        y = iJoints[:, 1] / H * 2 - 1
        iJoints = jnp.stack([x, y], axis=-1)
    return iJoints

def residual_with_threshold(r: jax.Array, tau: float, eps: float=1e-8) -> jax.Array:
    # r: (..., D)
    s = jnp.sqrt(r**2).sum(axis=-1, keepdims=True)
    # factor = max(0, (s - tau) / s)
    factor = (s - tau).clip(min=0.0) / (s + eps)
    return factor * r

def project_jax(oPoints, wTo, wTc, intr, ndc=False, H=None, W=None):
    """JAX version of projection function for optimization.

    Args:
        oPoints: Object points in object frame (P, 3)
        wTo: Object pose in world frame (7) - wxyz_xyz format
        wTc: Camera pose in world frame (7, ) - wxyz_xyz format
        intr: Camera intrinsics (3, 3)

    Returns:
        Projected 2D points (P, 2)
    """
    P = oPoints.shape[0]

    # Convert to SE3 objects for easier manipulation
    wTo_se3 = jaxlie.SE3(wTo)  #
    wTc_se3 = jaxlie.SE3(wTc)  #

    # Compute cTo = cTw @ wTo where cTw = wTc.inverse()
    cTw_se3 = wTc_se3.inverse()  #
    cTo_se3 = cTw_se3 @ wTo_se3  #
    # cTo_homo = cTo_se3.as_matrix()  # (4, 4)

    # Transform object points to camera frame
    # oPoints_homo = jnp.concatenate([oPoints, jnp.ones((P, 1))], axis=1)  # (P, 4)
    # oPoints_homo = oPoints_homo[None, :, :].repeat(T, axis=0)  # (T, P, 4)

    cPoints = cTo_se3 @ oPoints  # (P, 3)

    # Project to image plane
    iPoints_homo = cPoints @ intr.T
    iPoints = iPoints_homo[:, :2] / iPoints_homo[:, 2:3]  # (P, 2)

    if ndc:
        if H is None:
            H = intr[0, 2] * 2
        if W is None:
            W = intr[1, 2] * 2
        x = iPoints[:, 0] / W * 2 - 1
        y = iPoints[:, 1] / H * 2 - 1
        iPoints = jnp.stack([x, y], axis=-1)

    return iPoints


def project(wTc, intr, oPoints, wTo, wPoints=None, ndc=False, H=None, W=None):
    # B, P, _ = oPoints.shape
    B, T = wTc.shape[:2]
    wTc_tsl, wTc_6d = wTc[..., :3], wTc[..., 3:]
    wTc_mat = geom_utils.rotation_6d_to_matrix(wTc_6d)
    wTc = geom_utils.rt_to_homo(wTc_mat, wTc_tsl)

    cTw = geom_utils.inverse_rt_v2(wTc)  # (B, T, 4, 4)

    if wPoints is None:
        wTo_tsl, wTo_6d = wTo[..., :3], wTo[..., 3:]
        wTo_mat = geom_utils.rotation_6d_to_matrix(wTo_6d)
        wTo = geom_utils.rt_to_homo(wTo_mat, wTo_tsl)

        oPoints_exp = oPoints[:, None].repeat(1, T, 1, 1).reshape(B * T, -1, 3)
        wPoints = mesh_utils.apply_transform(oPoints_exp, wTo.reshape(B * T, 4, 4))

    wPoints = wPoints.reshape(B * T, -1, 3)
    P = wPoints.shape[-2]
    # cTo = cTw @ wTo
    cTw = cTw.reshape(B * T, 4, 4)

    cPoints = mesh_utils.apply_transform(wPoints, cTw).reshape(
        B, T, P, 3
    )  # (B, T, P, 3)

    intr_exp = intr.unsqueeze(1).repeat(1, T, 1, 1)  # (B, T, 3, 3)

    iPoints = cPoints @ intr_exp.transpose(-2, -1)
    iPoints = iPoints[..., :2] / iPoints[..., 2:3]  # (B, T, P, 2)

    if ndc:
        if H is None:
            H = intr[..., 0, 2] * 2
        if W is None:
            W = intr[..., 1, 2] * 2
        iPoints[..., 0] = iPoints[..., 0] / W.reshape(B, 1, 1) * 2 - 1
        iPoints[..., 1] = iPoints[..., 1] / H.reshape(B, 1, 1) * 2 - 1
    return iPoints


def test_optimization(save_dir="outputs/debug_guidance_hoi"):
    from .model import build_model
    device = "cuda:0"
    opt_file = "outputs/noisy_hand/hand_cond_out_consist_w0.1_contact10_1_bps2/opt.yaml"
    opt = OmegaConf.load(opt_file)

    save = pickle.load(open("outputs/tmp.pkl", "rb"))
    batch = save["sample"]
    batch = model_utils.to_cuda(batch, "cpu")

    pt3d_viz = Pt3dVisualizer(
        exp_name="vis_traj",
        save_dir=save_dir,
        mano_models_dir="assets/mano",
        object_mesh_dir="data/HOT3D-CLIP/object_models_eval/",
    )

    x2d = project(
        batch["wTc"],
        batch["intr"],
        batch["newPoints"],
        batch["wTo"],
        ndc=ndc,
    )  # (BS, T, P, 2)
    kP = 1000
    ind_list = []
    T = x2d.shape[1]
    b = x2d.shape[0]
    for t in range(T):
        inds = torch.randperm(x2d.shape[2])[:kP]
        ind_list.append(inds)
    ind_list = torch.stack(ind_list, dim=0).to(x2d.device)  # (T, kP)
    ind_list_exp = ind_list[None, :, :, None].repeat(b, 1, 1, 2)
    x2d = torch.gather(x2d, dim=2, index=ind_list_exp)  # (B, T, Q, 2) --? (B, T, kP, 2)

    B, T, J_3 = batch["hand_raw"].shape
    j2d = project(
        batch["wTc"],
        batch["intr"],
        None,
        None,
        wPoints=batch["hand_raw"].reshape(B, T, -1, 3),
        ndc=ndc,
    )

    pred = save["pred_raw"]
    model = build_model(opt)
    model.to(device)
    pred_dict = model.decode_dict(pred)

    obs = {
        "newPoints": batch["newPoints"],
        "wTc": se3_to_wxyz_xyz(batch["wTc"]),
        "intr": batch["intr"],
        "x2d": x2d,
        "contact": batch["contact"],
        "j3d": batch["hand_raw"].reshape(B, T, -1, 3),
        "j2d": j2d.reshape(B, T, -1, 2),
    }

    left_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="left")
    right_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="right")

    guidance_params = JaxGuidanceParams(
        use_abs_contact=True,
        use_rel_contact=True,
        contact_weight=0.1,
        use_reproj_cd=True,
        contact_th=0.5,
        max_iters=100,
        use_obj_smoothness=True,
        
        use_j3d=False,
        use_j2d=True,
        j2d_weight=1.0,

        use_contact_obs=True,
        contact_obs_weight=1.0,
    )

    # for _ in range(3):
    pred_opt, _ = do_guidance_optimization(
        pred_dict=pred_dict,
        obs=obs,
        guidance_params=guidance_params,
        left_mano_model=left_mano_model,
        right_mano_model=right_mano_model,
        guidance_mode="hoi_contact",
        phase="inner",
        verbose=True,
    )
    for k, v in pred_opt.items():
        print(k, v.shape)

    batch = model_utils.to_cuda(batch, device)
    gt_hand_meshes = model.decode_hand_mesh(
        batch["left_hand_params"][0],
        batch["right_hand_params"][0],
        hand_rep="theta",
    )
    fname = pt3d_viz.log_hoi_step(
        *gt_hand_meshes,
        batch["wTo"][0],
        batch["newMesh"],
        pref="gt",
        contact=batch["contact"][0],
    )

    opt_hand = model.decode_hand_mesh(
        pred_opt["left_hand_params"][0],
        pred_opt["right_hand_params"][0],
        hand_rep="theta",
    )
    fname = pt3d_viz.log_hoi_step(
        *opt_hand,
        pred_opt["wTo"][0],
        batch["newMesh"],
        pref="opt",
        contact=pred_opt["contact"][0],
    )

    init_hand = model.decode_hand_mesh(
        pred_dict["left_hand_params"][0],
        pred_dict["right_hand_params"][0],
        hand_rep="theta",
    )
    fname = pt3d_viz.log_hoi_step(
        *init_hand,
        pred_dict["wTo"][0],
        batch["newMesh"],
        pref="init",
        contact=pred_dict["contact"][0],
    )



if __name__ == "__main__":
    Fire(test_optimization)
