"""Optimize hand parameters to match target joints."""

from __future__ import annotations

import os

# Need to play nice with PyTorch!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pickle
from functools import partial
from pathlib import Path
from typing import cast

import jax
import jax_dataclasses as jdc
import jaxls
import numpy as onp
import torch
from fire import Fire
from jax import numpy as jnp
from jutils import hand_utils, image_utils, mesh_utils, model_utils
from omegaconf import OmegaConf

from . import build_model, fncmano_jax


class _HandParamsVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros(
        (31,)
    ),  # global_orient(3) + transl(3) + hand_pose(15) + betas(10)
    retract_fn=lambda val, delta: val + delta,  # Simple addition for free parameters
    tangent_dim=31,
):
    """Variable containing hand parameters."""


@jdc.pytree_dataclass
class HandGuidanceParams:
    joint_weight: float = 1.0
    lambda_initial: float = 0.1
    max_iters: jdc.Static[int] = 50


def do_forward_kinematics_one_hand(
    vals: jaxls.VarValues, 
    var: _HandParamsVar, 
    shaped_model: fncmano_jax.MANOShaped
):
    """Helper to compute forward kinematics from variable values.
    
    Args:
        vals: Variable values from JAXLS optimization
        var: Hand parameters variable
        shaped_model: Shaped MANO model (with betas applied)
        
    Returns:
        MANOMesh with joints and vertices
    """
    params = vals[var]
    g_orient = params[:3]
    trans = params[3:6]
    h_pose = params[6:21]

    posed = shaped_model.with_pose(
        global_orient=g_orient,
        transl=trans,
        pca=h_pose,
        add_mean=True,
    )
    mesh = posed.lbs()
    return mesh


def do_guidance_optimization_hand(
    pred_dict: dict,
    obs: dict,
    left_mano_model: fncmano_jax.MANOModel,
    right_mano_model: fncmano_jax.MANOModel,
    guidance_params: HandGuidanceParams | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize hand parameters to match target joints."""
    if guidance_params is None:
        guidance_params = HandGuidanceParams()

    # Extract parameters
    left_hand_params = pred_dict["left_hand_params"]  # (B, T, 31)
    right_hand_params = pred_dict["right_hand_params"]  # (B, T, 31)
    target_joints = obs["j3d"]  # (B, T, 42, 3)

    # Split target joints into left and right
    target_left_joints = target_joints[..., :21, :]  # (B, T, 21, 3)
    target_right_joints = target_joints[..., 21:, :]  # (B, T, 21, 3)

    # Convert to JAX
    left_hand_jax = cast(jax.Array, left_hand_params.numpy(force=True))
    right_hand_jax = cast(jax.Array, right_hand_params.numpy(force=True))
    target_left_jax = cast(jax.Array, target_left_joints.numpy(force=True))
    target_right_jax = cast(jax.Array, target_right_joints.numpy(force=True))

    # Concatenate hand params for two-hand optimization
    hand_params_concat = jnp.concatenate([left_hand_jax, right_hand_jax], axis=-1)  # (B, T, 62)

    # Optimize both hands together
    optimized_concat, _ = _optimize_vmapped_two_hand(
        left_mano_model=left_mano_model,
        right_mano_model=right_mano_model,
        hand_params_concat=hand_params_concat,
        target_left_joints=target_left_jax,
        target_right_joints=target_right_jax,
        guidance_params=guidance_params,
        verbose=verbose,
    )

    # Split back into left and right
    optimized_left = optimized_concat[..., :31]  # (B, T, 31)
    optimized_right = optimized_concat[..., 31:]  # (B, T, 31)

    # Convert back to torch
    optimized_left_torch = (
        torch.from_numpy(onp.array(optimized_left))
        .to(left_hand_params.dtype)
        .to(left_hand_params.device)
    )
    optimized_right_torch = (
        torch.from_numpy(onp.array(optimized_right))
        .to(right_hand_params.dtype)
        .to(right_hand_params.device)
    )

    return {
        "left_hand_params": optimized_left_torch,
        "right_hand_params": optimized_right_torch,
    }


@jdc.jit
def _optimize_vmapped_two_hand(
    left_mano_model: fncmano_jax.MANOModel,
    right_mano_model: fncmano_jax.MANOModel,
    hand_params_concat: jax.Array,  # (B, T, 62)
    target_left_joints: jax.Array,  # (B, T, 21, 3)
    target_right_joints: jax.Array,  # (B, T, 21, 3)
    guidance_params: HandGuidanceParams,
    verbose: jdc.Static[bool],
) -> tuple[jax.Array, dict]:
    """Vectorized optimization over batch and time for both hands."""
    return jax.vmap(
        partial(
            _optimize_two_hand,
            left_mano_model=left_mano_model,
            right_mano_model=right_mano_model,
            guidance_params=guidance_params,
            verbose=verbose,
        )
    )(
        hand_params_concat=hand_params_concat,
        target_left_joints=target_left_joints,
        target_right_joints=target_right_joints,
    )


def _optimize_two_hand(
    left_mano_model: fncmano_jax.MANOModel,
    right_mano_model: fncmano_jax.MANOModel,
    hand_params_concat: jax.Array,  # (T, 62) = (left 31 + right 31)
    target_left_joints: jax.Array,  # (T, 21, 3)
    target_right_joints: jax.Array,  # (T, 21, 3)
    guidance_params: HandGuidanceParams,
    verbose: bool,
) -> tuple[jax.Array, dict]:
    """Optimize both hands with concatenated params."""
    # Split concatenated params
    # 3+3+15+10 = 31
    left_params = hand_params_concat[..., :31]  # (T, 31)
    right_params = hand_params_concat[..., 31:]  # (T, 31)

    # Parse betas for shaping
    left_betas = left_params[..., -10:]
    right_betas = right_params[..., -10:]

    # Build forward kinematics functions
    left_shaped = left_mano_model.with_shape(jnp.mean(left_betas, axis=0))  # (10,)
    right_shaped = right_mano_model.with_shape(jnp.mean(right_betas, axis=0))  # (10,)

    # Create partially applied functions for left and right hands using reusable helper
    do_forward_kinematics_left = partial(do_forward_kinematics_one_hand, shaped_model=left_shaped)
    do_forward_kinematics_right = partial(do_forward_kinematics_one_hand, shaped_model=right_shaped)

    # Create cost factors
    factors = list[jaxls.Cost]()

    def cost_with_args(*args):
        """Decorator for appending cost factors."""

        def inner(cost_func):
            factors.append(jaxls.Cost(cost_func, args))
            return cost_func

        return inner

    timesteps = hand_params_concat.shape[0]

    # Create two separate variables for left and right hands
    var_left = _HandParamsVar(jnp.arange(timesteps))
    var_right = _HandParamsVar(jnp.arange(timesteps))

    # Left hand joint matching cost
    @cost_with_args(var_left, target_left_joints)
    def left_joint_cost(
        vals: jaxls.VarValues,
        var: _HandParamsVar,
        target: jax.Array,
    ) -> jax.Array:
        print('left_joint_cost', vals[var].shape, target.shape, vals.shape)
        """Cost for matching predicted left hand joints to target joints."""
        mesh = do_forward_kinematics_left(vals, var)
        predicted_joints = mesh.joints  # First 21 joints

        residual = predicted_joints - target  # (21, 3)
        return guidance_params.joint_weight * residual.flatten()

    # Right hand joint matching cost
    @cost_with_args(var_right, target_right_joints)
    def right_joint_cost(
        vals: jaxls.VarValues,
        var: _HandParamsVar,
        target: jax.Array,
    ) -> jax.Array:
        """Cost for matching predicted right hand joints to target joints."""
        mesh = do_forward_kinematics_right(vals, var)
        predicted_joints = mesh.joints  # First 21 joints

        residual = predicted_joints - target  # (21, 3)
        return guidance_params.joint_weight * residual.flatten()

    # Set up optimization problem
    graph = jaxls.LeastSquaresProblem(costs=factors, variables=[var_left, var_right]).analyze()

    # Solve
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make([
            var_left.with_value(left_params),
            var_right.with_value(right_params),
        ]),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(
            lambda_initial=guidance_params.lambda_initial
        ),
        termination=jaxls.TerminationConfig(max_iterations=guidance_params.max_iters),
        verbose=verbose,
    )

    out_left_params = solutions[var_left]
    out_right_params = solutions[var_right]

    assert out_left_params.shape == (timesteps, 31)
    assert out_right_params.shape == (timesteps, 31)

    # Concatenate back to (T, 62) format
    out_hand_params_concat = jnp.concatenate([out_left_params, out_right_params], axis=-1)

    return out_hand_params_concat, {}


def test_optimization(save_dir="outputs/debug_guidance"):
    """Test the hand optimization."""
    device = "cuda:0"
    opt_file = "outputs/noisy_hand/hand_cond_out_consist_w0.1_contact10_1_bps2/opt.yaml"
    opt = OmegaConf.load(opt_file)

    save = pickle.load(open("outputs/tmp.pkl", "rb"))
    batch = save["sample"]
    batch = model_utils.to_cuda(batch, "cpu")

    pred = save["pred_raw"]
    model = build_model(opt)
    model.to(device)
    pred_dict = model.decode_dict(pred)

    B, T, J_3 = batch["hand_raw"].shape

    obs = {
        "j3d": batch["hand_raw"].reshape(B, T, -1, 3),
        "contact": pred_dict["contact"],
    }

    left_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="left")
    right_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="right")

    # Optimize
    guidance_params = HandGuidanceParams(joint_weight=1.0, max_iters=50)
    pred_opt = do_guidance_optimization_hand(
        pred_dict=pred_dict,
        obs=obs,
        guidance_params=guidance_params,
        left_mano_model=left_mano_model,
        right_mano_model=right_mano_model,
        verbose=True,
    )

    # Visualize
    batch = model_utils.to_cuda(batch, device)
    # pred_opt = model_utils.to_cuda(pred_opt, device)
    pred_dict = model_utils.to_cuda(pred_dict, device)

    pred_hand_meshes = model.decode_hand_mesh(
        pred_dict["left_hand_params"][0],
        pred_dict["right_hand_params"][0],
        hand_rep="theta",
    )
    pred_hand_meshes = mesh_utils.join_scene(pred_hand_meshes)

    gt_hand_meshes = model.decode_hand_mesh(
        batch["left_hand_params"][0],
        batch["right_hand_params"][0],
        hand_rep="theta",
    )
    gt_hand_meshes = mesh_utils.join_scene(gt_hand_meshes)

    opt_hand_meshes = model.decode_hand_mesh(
        pred_opt["left_hand_params"][0],
        pred_opt["right_hand_params"][0],
        hand_rep="theta",
    )
    opt_hand_meshes = mesh_utils.join_scene(opt_hand_meshes)

    pred_hand_meshes.textures = mesh_utils.pad_texture(pred_hand_meshes, "blue")
    gt_hand_meshes.textures = mesh_utils.pad_texture(gt_hand_meshes, "red")
    opt_hand_meshes.textures = mesh_utils.pad_texture(opt_hand_meshes, "yellow")

    scene = mesh_utils.join_scene([pred_hand_meshes, gt_hand_meshes, opt_hand_meshes])
    image_list = mesh_utils.render_geom_rot_v2(scene, time_len=1)
    image_list = torch.stack(image_list, axis=0)
    time_len, T, C, H, W = image_list.shape
    image_utils.save_gif(
        image_list.reshape(time_len * T, 1, C, H, W),
        f"{save_dir}/guided_hands",
        fps=30,
        ext=".mp4",
    )


if __name__ == "__main__":
    Fire(test_optimization)
