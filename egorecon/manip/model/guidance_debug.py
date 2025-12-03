"""Minimal debug version of guidance optimizer - single cost term."""
from fire import Fire
import pickle
import os

# Need to play nice with PyTorch!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from functools import partial
from typing import cast

import jax
import jax_dataclasses as jdc
import jaxls
import numpy as onp
import torch
from jax import numpy as jnp
from jutils import geom_utils

import jaxlie


def se3_to_wxyz_xyz(se3):
    """Convert SE3 (9D: tsl + rot6d) to wxyz_xyz format (7D)."""
    tsl, rot6d = torch.split(se3, [3, 6], dim=-1)
    mat = geom_utils.rotation_6d_to_matrix(rot6d)
    wxyz = geom_utils.rot_cvt.matrix_to_quaternion(mat)
    return torch.cat([wxyz, tsl], dim=-1)


def wxyz_xyz_to_se3(wxyz_xyz):
    """Convert wxyz_xyz format (7D) to SE3 (9D: tsl + rot6d)."""
    wxyz, tsl = torch.split(wxyz_xyz, [4, 3], dim=-1)
    mat = geom_utils.rot_cvt.quaternion_to_matrix(wxyz)
    return geom_utils.matrix_to_se3_v2(geom_utils.rt_to_homo(mat, tsl))


def apply_trajectory_filter(trajectory: jax.Array, filter_type: str = "none") -> jax.Array:
    """Apply trajectory filtering (e.g., smoothing, denoising).
    
    Args:
        trajectory: Full trajectory shape (T, 7) in wxyz_xyz format
        filter_type: Type of filter to apply ("none", "smooth", etc.)
    
    Returns:
        Filtered trajectory shape (T, 7)
    
    NOTE: This is used for preprocessing (Option 1). jaxls with array IDs gives
    per-timestep access (7,) in cost functions, so trajectory filtering must be
    done before optimization or through iterative optimization+filtering loops.
    """
    if filter_type == "none":
        return trajectory
    elif filter_type == "smooth":
        # Example: Simple smoothing (customize as needed)
        # You can implement more sophisticated filters here
        filtered = trajectory.copy()
        # Add your filtering logic here
        # For now, just return unchanged
        return filtered
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


@jdc.pytree_dataclass
class JaxGuidanceParams:
    """Minimal guidance parameters - only one weight."""
    wTo_weight: float = 1.0
    
    # Trajectory filtering (Option 1: preprocessing)
    filter_trajectory: bool = False
    filter_type: str = "none"  # "none", "smooth", etc.
    
    # Optimization parameters
    lambda_initial: float = 0.1
    step_quality_min: float = 1e-4
    max_iters: jdc.Static[int] = 50


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
    """Variable containing SE3 trajectory as wxyz_xyz format [qw, qx, qy, qz, x, y, z].    
    """


def do_guidance_optimization(
    pred_dict: dict,
    obs: dict,
    guidance_params: JaxGuidanceParams = None,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """Run an optimizer to optimize SE3 trajectory.

    Args:
        pred_dict: Dictionary containing predictions, e.g., {'wTo': wTo_pred}
        obs: Dictionary containing observations/targets, e.g., {'wTo': target_wTo}
        guidance_params: Guidance parameters
        verbose: Whether to print optimization progress

    Returns:
        Tuple of (optimized_trajectory_dict, debug_info)
    """
    start_time = time.time()
    if guidance_params is None:
        guidance_params = JaxGuidanceParams()

    # Extract prediction
    wTo_pred = se3_to_wxyz_xyz(pred_dict["wTo"])
    device = wTo_pred.device

    # Convert to JAX arrays
    wTo_pred_jax = cast(jax.Array, wTo_pred.numpy(force=True))

    # Extract observation
    target_wTo = se3_to_wxyz_xyz(obs["wTo"])
    target_wTo_jax = cast(jax.Array, target_wTo.numpy(force=True))

    # OPTION 1: Apply trajectory filtering as preprocessing
    if guidance_params.filter_trajectory:
        wTo_pred_jax = apply_trajectory_filter(wTo_pred_jax, filter_type=guidance_params.filter_type)
    
    # Optimize using vmapped function
    optimized_traj, debug_info = _optimize_vmapped(
        wTo=wTo_pred_jax,
        target_wTo=target_wTo_jax,
        guidance_params=guidance_params,
        verbose=verbose,
    )

    # Convert debug_info JAX arrays to Python floats
    debug_info_converted = {}
    for key, value in debug_info.items():
        if isinstance(value, jax.Array):
            value_np = onp.array(value)
            debug_info_converted[key] = value_np
        else:
            debug_info_converted[key] = value
    debug_info = debug_info_converted

    # Convert back to torch tensor
    wTo = wxyz_xyz_to_se3(torch.from_numpy(onp.array(optimized_traj["wTo"])).to(device))

    optimized_traj_torch = {
        "wTo": wTo,
    }

    
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"GPU memory - Allocated: {allocated_mb:.2f} MB, Reserved: {reserved_mb:.2f} MB, Peak: {max_allocated_mb:.2f} MB")

    return optimized_traj_torch, debug_info


@jdc.jit
def _optimize_vmapped(
    wTo: jax.Array,
    target_wTo: jax.Array,
    guidance_params: JaxGuidanceParams,
    verbose: jdc.Static[bool],
) -> tuple[dict, dict]:
    """Vmap wrapper for optimization."""
    return jax.vmap(
        partial(
            _optimize,
            guidance_params=guidance_params,
            verbose=verbose,
        )
    )(
        wTo=wTo,
        target_wTo=target_wTo,
    )


def _optimize(
    wTo: jax.Array,
    target_wTo: jax.Array,
    guidance_params: JaxGuidanceParams,
    verbose: jdc.Static[bool],
) -> tuple[dict, dict]:
    """Core optimization function - single cost term: wTo - target_wTo.
    
    For trajectory filtering, apply preprocessing before calling this function (Option 1).
    """
    timesteps = wTo.shape[0]
    assert wTo.shape == (timesteps, 7)
    assert target_wTo.shape == (timesteps, 7)

    # We'll populate a list of factors (cost terms)
    factors = list[jaxls.Cost]()

    def cost_with_args(*args):
        """Decorator for appending to the factor list."""

        def inner(cost_func):
            factors.append(jaxls.Cost(cost_func, args))
            return cost_func

        return inner

    # Create variable for the entire trajectory with array ID
    # jaxls will automatically vmap over timesteps, giving us (7,) per call
    var_traj = _SE3TrajectoryVar(jnp.arange(timesteps))

    # Single cost: wTo - target_wTo
    @cost_with_args(
        _SE3TrajectoryVar(jnp.arange(timesteps)),
        target_wTo,
    )
    def wTo_cost(
        vals: jaxls.VarValues,
        var_traj: _SE3TrajectoryVar,
        target_wTo: jax.Array,
    ) -> jax.Array:
        """Cost term: difference between wTo and target_wTo.
        
        jaxls automatically vmaps over timesteps, so we receive single pose (7,) here.
        For trajectory filtering, use preprocessing (Option 1) before optimization.
        """
        wTo = vals[var_traj]  # Shape: (7,) - single pose per timestep
        wTo_se3 = jaxlie.SE3(wTo)
        target_wTo_se3 = jaxlie.SE3(target_wTo)
        
        # Compute residual in tangent space
        residual = (wTo_se3.inverse() @ target_wTo_se3).log()
        
        # Scale by weight
        res = guidance_params.wTo_weight * residual.flatten()
        return res

    # Set up the optimization problem
    graph = jaxls.LeastSquaresProblem(
        costs=factors, variables=[var_traj]
    ).analyze()

    # Solve using Levenberg-Marquardt
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [var_traj.with_value(wTo)]
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

    # Simple debug info
    debug_info = {"wTo_cost": jnp.array(0.0)}  # Placeholder

    return {
        "wTo": optimized_traj,
    }, debug_info



def test_guidance(B: int = 1, T: int = 10, verbose: bool = True):
    """Test the guidance optimization with dummy data.
    
    Args:
        B: Batch size
        T: Number of timesteps
        verbose: Whether to print optimization progress
    """
    tmp_file = 'outputs/tmp.pkl'
    with open(tmp_file, 'rb') as f:
        data = pickle.load(f)
    target_wTo = data['sample']['wTo']
    print('target_wTo', target_wTo.shape, type(target_wTo))
    target_wTo = target_wTo[:B, :T, :]

    device = "cuda:0"
    
    # Create dummy predicted trajectory (SE3 format: tsl + rot6d = 9D)
    # Start with identity pose, add some noise
    pred_wTo = torch.zeros(B, T, 9, device=device)
    pred_wTo[..., :3] = torch.randn(B, T, 3) * 0.1  # Small translation noise
    # Identity rotation in 6D format: [1, 0, 0, 0, 1, 0]
    pred_wTo[..., 3] = 1.0
    pred_wTo[..., 6] = 1.0
    
    # Create target trajectory (slightly different from prediction)
    # For rotation, create valid rotation matrices first, then convert to 6D
    # target_wTo = pred_wTo.clone()
    # target_wTo[..., :3] += torch.randn(B, T, 3) * 0.2  # Add offset to translation
    
    # Create valid rotation matrices with small perturbations
    identity_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
    # Add small random rotation
    noise_angle = torch.randn(B, T, 3, device=device) * 0.1
    noise_rot = geom_utils.rot_cvt.axis_angle_to_matrix(noise_angle)
    target_rot_mat = torch.bmm(identity_rot.view(B * T, 3, 3), noise_rot.view(B * T, 3, 3))
    target_rot_mat = target_rot_mat.view(B, T, 3, 3)
    # Convert to 6D
    target_wTo[..., 3:9] = geom_utils.matrix_to_rotation_6d(target_rot_mat)
    
    # Create prediction dictionary
    pred_dict = {
        "wTo": pred_wTo,  # (B, T, 9) - SE3 format
    }
    
    # Create observation dictionary
    obs = {
        "wTo": target_wTo,  # (B, T, 9) - SE3 format
    }
    
    # Create guidance parameters
    guidance_params = JaxGuidanceParams(
        wTo_weight=1.0,
        lambda_initial=0.1,
        step_quality_min=1e-4,
        max_iters=10,  # Fewer iterations for quick test
    )
    
    print("Testing guidance optimization:")
    print(f"  Batch size: {B}, Timesteps: {T}")
    print(f"  Initial wTo shape: {pred_wTo.shape}")
    print(f"  Target wTo shape: {target_wTo.shape}")
    print("  Running optimization...")
    
    # Run optimization
    optimized_traj, debug_info = do_guidance_optimization(
        pred_dict=pred_dict,
        obs=obs,
        guidance_params=guidance_params,
        verbose=verbose,
    )
    
    # Print results
    print("\nOptimization complete!")
    print(f"  Optimized wTo shape: {optimized_traj['wTo'].shape}")
    print(f"  Debug info keys: {list(debug_info.keys())}")
    
    # Compute error before and after
    pred_wxyz = se3_to_wxyz_xyz(pred_wTo)  # (B, T, 7)
    target_wxyz = se3_to_wxyz_xyz(target_wTo)  # (B, T, 7)
    opt_wxyz = se3_to_wxyz_xyz(optimized_traj["wTo"])  # (B, T, 7)
    
    # Compute translation error
    pred_transl_error = torch.norm(pred_wxyz[..., 4:] - target_wxyz[..., 4:], dim=-1).mean()
    opt_transl_error = torch.norm(opt_wxyz[..., 4:] - target_wxyz[..., 4:], dim=-1).mean()
    
    print("\nTranslation error:")
    print(f"  Before optimization: {pred_transl_error.item():.6f}")
    print(f"  After optimization: {opt_transl_error.item():.6f}")
    print(f"  Improvement: {((pred_transl_error - opt_transl_error) / pred_transl_error * 100).item():.2f}%")
    
    return optimized_traj, debug_info


if __name__ == "__main__":
    Fire(test_guidance)