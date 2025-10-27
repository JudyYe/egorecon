"""Optimize hand parameters to match target joints."""

from __future__ import annotations
import os

# Need to play nice with PyTorch!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from functools import partial
from typing import Literal, cast

import jax
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import torch
from jax import numpy as jnp

from . import fncmano_jax
from pathlib import Path


class _HandParamsVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros((31,)),  # global_orient(3) + transl(3) + hand_pose(15) + betas(10)
    retract_fn=lambda val, delta: val + delta,  # Simple addition for free parameters
    tangent_dim=31,
):
    """Variable containing hand parameters."""


@jdc.pytree_dataclass
class HandGuidanceParams:
    joint_weight: float = 1.0
    lambda_initial: float = 0.1
    max_iters: jdc.Static[int] = 50


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

    # Optimize both hands independently
    optimized_left, _ = _optimize_vmapped_hand(
        mano_model=left_mano_model,
        hand_params=left_hand_jax,
        target_joints=target_left_jax,
        guidance_params=guidance_params,
        verbose=verbose,
    )

    optimized_right, _ = _optimize_vmapped_hand(
        mano_model=right_mano_model,
        hand_params=right_hand_jax,
        target_joints=target_right_jax,
        guidance_params=guidance_params,
        verbose=verbose,
    )

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
def _optimize_vmapped_hand(
    mano_model: fncmano_jax.MANOModel,
    hand_params: jax.Array,
    target_joints: jax.Array,
    guidance_params: HandGuidanceParams,
    verbose: jdc.Static[bool],
) -> tuple[jax.Array, dict]:
    """Vectorized optimization over batch and time."""
    return jax.vmap(
            partial(
                _optimize_single_hand,
                mano_model=mano_model,
                guidance_params=guidance_params,
                verbose=verbose,
            )
    )(hand_params=hand_params, target_joints=target_joints)


def _optimize_single_hand(
    mano_model: fncmano_jax.MANOModel,
    hand_params: jax.Array,  # (T, 31,)
    target_joints: jax.Array,  # (T, 21, 3)
    guidance_params: HandGuidanceParams,
    verbose: bool,
) -> jax.Array:
    """Optimize hand parameters for a single frame."""
    # Parse hand parameters
    # Structure: [global_orient(3), transl(3), hand_pose(15), betas(10)]
    betas = hand_params[..., -10:]

    # Build forward kinematics function
    shaped = mano_model.with_shape(jnp.mean(betas, axis=0))  # (10,)

    def do_forward_kinematics(vals: jaxls.VarValues, var: _HandParamsVar):
        """Helper to compute forward kinematics from variable values."""
        # Extract parameters from variable
        params = vals[var]
        g_orient = params[:3]
        trans = params[3:6]
        h_pose = params[6:21]

        # Forward kinematics
        posed = shaped.with_pose(
            global_orient=g_orient,
            transl=trans,
            pca=h_pose,
            add_mean=True,
        )
        mesh = posed.lbs()
        return mesh

    # Create cost factors
    factors = list[jaxls.Cost]()

    def cost_with_args(*args):
        """Decorator for appending cost factors."""

        def inner(cost_func):
            factors.append(jaxls.Cost(cost_func, args))
            return cost_func

        return inner

    timesteps = hand_params.shape[0]
    # Joint matching cost
    @cost_with_args(_HandParamsVar(jnp.arange(timesteps)), target_joints)
    def joint_cost(
        vals: jaxls.VarValues,
        var: _HandParamsVar,
        target: jax.Array,
    ) -> jax.Array:
        """Cost for matching predicted joints to target joints."""
        mesh = do_forward_kinematics(vals, var)
        predicted_joints = mesh.joints  # First 21 joints

        # Compute residual
        residual = predicted_joints - target  # (21, 3)

        return guidance_params.joint_weight * residual.flatten()

    # Set up optimization problem
    var_hand = _HandParamsVar(jnp.arange(timesteps))
    graph = jaxls.LeastSquaresProblem(costs=factors, variables=[var_hand]).analyze()

    # Solve
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make([var_hand.with_value(hand_params)]),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=guidance_params.lambda_initial),
        termination=jaxls.TerminationConfig(max_iterations=guidance_params.max_iters),
        verbose=verbose,
    )
    out_hand_params = solutions[_HandParamsVar]
    assert out_hand_params.shape == (timesteps, 31)


    return out_hand_params, {}



# Forward kinematics helper for external use
def do_forward_kinematics_mano(
    mano_model: fncmano_jax.MANOModel,
    global_orient: jnp.ndarray,  # (..., 3)
    transl: jnp.ndarray,  # (..., 3)
    hand_pose: jnp.ndarray,  # (..., 15) - PCA components
    betas: jnp.ndarray,  # (..., 10)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform forward kinematics on MANO model."""
    shaped = mano_model.with_shape(betas)
    posed = shaped.with_pose(
        global_orient=global_orient,
        transl=transl,
        pca=hand_pose,
        add_mean=True,
    )
    mesh = posed.lbs()
    return mesh.verts, mesh.joints, mesh.faces


# Test function
from fire import Fire
import pickle
from jutils import image_utils, model_utils, hand_utils
from omegaconf import OmegaConf
from . import build_model


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
    from jutils import mesh_utils
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
