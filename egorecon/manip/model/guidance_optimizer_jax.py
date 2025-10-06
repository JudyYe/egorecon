"""Optimize SE3 trajectories using Levenberg-Marquardt."""

from __future__ import annotations

import os

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

from ...utils.rotation_utils import rotation_6d_to_matrix_numpy


def do_guidance_optimization(
    traj: Tensor,
    obs: dict,
    guidance_mode: GuidanceMode,
    phase: Literal["inner", "post"],
    verbose: bool,
) -> tuple[Tensor, dict]:
    """Run an optimizer to optimize SE3 trajectory.
    
    Args:
        traj: SE3 trajectory to optimize (B, T, 9) - [x,y,z,rx,ry,rz,sx,sy,sz]
        obs: Dictionary containing observations/targets, e.g., {'x': gt_trajectory}
        guidance_mode: Guidance mode
        phase: Optimization phase
        verbose: Whether to print optimization progress
        
    Returns:
        Tuple of (optimized_trajectory, debug_info)
    """
    guidance_params = JaxGuidanceParams.defaults(guidance_mode, phase)
    
    # Extract trajectory and observations
    x_traj = traj  # Shape: (B, T, 9) - [x,y,z,rx,ry,rz,sx,sy,sz] - PARAMETERS TO OPTIMIZE
    gt_traj = obs['x']  # Shape: (B, T, 9) - [x,y,z,rx,ry,rz,sx,sy,sz] - OBSERVATIONS/TARGETS
    batch_size, timesteps, _ = x_traj.shape
    
    start_time = time.time()
    
    # Convert to JAX arrays and optimize
    x_traj_jax = cast(jax.Array, x_traj.numpy(force=True))
    gt_traj_jax = cast(jax.Array, gt_traj.numpy(force=True))
    
    # Optimize using vmapped function
    optimized_traj, debug_info = _optimize_vmapped(
        x_traj=x_traj_jax,
        gt_traj=gt_traj_jax,
        guidance_params=guidance_params,
        verbose=verbose,
    )
    
    # Convert back to torch tensor
    optimized_traj_torch = torch.from_numpy(onp.array(optimized_traj)).to(x_traj.dtype).to(x_traj.device)
    
    print(f"SE3 trajectory optimization finished in {time.time() - start_time}sec")
    return optimized_traj_torch, debug_info


class _SE3TrajectoryVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # [x,y,z,rx,ry,rz,sx,sy,sz]
    retract_fn=lambda val, delta: val + delta,  # Simple additive retraction for now
    tangent_dim=9,  # 9D parameter space
):
    """Variable containing SE3 trajectory parameters [x,y,z,rx,ry,rz,sx,sy,sz]."""


@jdc.jit
def _optimize_vmapped(
    x_traj: jax.Array,
    gt_traj: jax.Array,
    guidance_params: JaxGuidanceParams,
    verbose: jdc.Static[bool],
) -> tuple[jax.Array, dict]:
    """Vectorized optimization over batch dimension using JAX vmap.
    
    Benefits of _optimize_vmapped:
    1. **Parallel Processing**: Optimizes multiple trajectories in parallel using JAX's vmap
    2. **Compilation Efficiency**: JIT compilation is more effective with vectorized operations
    3. **Memory Efficiency**: Better memory access patterns for batch processing
    4. **GPU Utilization**: Maximizes GPU parallelization for batch optimization
    5. **Code Reusability**: Single optimization function can handle both single and batch inputs
    6. **Performance**: Significantly faster than sequential optimization loops
    
    Args:
        x_traj: SE3 trajectories to optimize (B, T, 9) - PARAMETERS
        gt_traj: Ground truth SE3 trajectories (B, T, 9) - OBSERVATIONS  
        guidance_params: Guidance parameters
        verbose: Whether to print optimization progress
        
    Returns:
        Tuple of (optimized_trajectories, debug_info)
    """
    return jax.vmap(
        partial(
            _optimize,
            guidance_params=guidance_params,
            verbose=verbose,
        )
    )(x_traj=x_traj, gt_traj=gt_traj)


# Modes for guidance
GuidanceMode = Literal[
    "simple_l2",      # Simple L2 distance to target
    "with_smoothness", # L2 + smoothness
    "reproj_corrsp",          # Reprojection loss with points in correspondence
]


@jdc.pytree_dataclass
class JaxGuidanceParams:
    # L2 cost weights
    l2_weight: float = 1.0
    
    # Smoothness weights
    smoothness_weight: float = 0.1
    use_smoothness: jdc.Static[bool] = False
    
    # Optimization parameters
    lambda_initial: float = 0.1  # Increased for better convergence
    max_iters: jdc.Static[int] = 50  # Increased for better convergence
    
    @staticmethod
    def defaults(
        mode: GuidanceMode,
        phase: Literal["inner", "post"],
    ) -> JaxGuidanceParams:
        if mode == "simple_l2":
            return JaxGuidanceParams(
                use_smoothness=False,
                max_iters=5 if phase == "inner" else 20,  # Much faster for paper performance
            )
        elif mode == "with_smoothness":
            return JaxGuidanceParams(
                use_smoothness=True,
                smoothness_weight=0.1,
                max_iters=3 if phase == "inner" else 5,  # Much faster for paper performance
            )
        else:
            assert_never(mode)


def _optimize(
    x_traj: jax.Array,
    gt_traj: jax.Array,
    guidance_params: JaxGuidanceParams,
    verbose: bool,
) -> jax.Array:
    """Apply constraints using Levenberg-Marquardt optimizer. Returns updated SE3 trajectory.
    
    Args:
        x_traj: SE3 trajectory to optimize (T, 9) - [x,y,z,rx,ry,rz,sx,sy,sz] - PARAMETERS
        gt_traj: Ground truth SE3 trajectory (T, 9) - [x,y,z,rx,ry,rz,sx,sy,sz] - OBSERVATIONS
        guidance_params: Guidance parameters
        verbose: Whether to print optimization progress
        
    Returns:
        Optimized SE3 trajectory (T, 9)
    """
    timesteps = x_traj.shape[0]
    assert x_traj.shape == (timesteps, 9)
    assert gt_traj.shape == (timesteps, 9)
    
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
    
    # L2 distance cost - minimize difference from ground truth observations
    @cost_with_args(var_traj, gt_traj)
    def l2_distance_cost(
        vals: jaxls.VarValues,
        var_traj: _SE3TrajectoryVar,
        gt_traj: jax.Array,
    ) -> jax.Array:
        """L2 distance cost between optimized trajectory and ground truth observations."""
        current_traj = vals[var_traj]  # Direct access - much faster!
        # Compute squared L2 distance for proper convergence
        diff = current_traj - gt_traj
        return guidance_params.l2_weight * (diff * diff).flatten()
    
    # Temporal smoothness cost (optional) - more efficient vectorized version
    if guidance_params.use_smoothness:
        @cost_with_args(var_traj)
        def smoothness_cost(
            vals: jaxls.VarValues,
            *args,
        ) -> jax.Array:
            """Temporal smoothness cost between consecutive frames - vectorized."""
            current_traj = vals[var_traj]
            # Compute differences between consecutive frames
            diff = current_traj[1:] - current_traj[:-1]
            return guidance_params.smoothness_weight * diff.flatten()
    
    # Set up the optimization problem
    graph = jaxls.LeastSquaresProblem(
        costs=factors, variables=[var_traj]
    ).analyze()
    
    # Solve using Levenberg-Marquardt
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make([
            var_traj.with_value(x_traj)
        ]),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(
            lambda_initial=guidance_params.lambda_initial
        ),
        termination=jaxls.TerminationConfig(max_iterations=guidance_params.max_iters),
        verbose=verbose,
    )
    
    # Extract optimized trajectory
    optimized_traj = solutions[var_traj]
    debug_info = {"final_cost": None}  # Could compute final cost if needed
    
    return optimized_traj, debug_info


def project(oPoints, wTo, wTc, intr):
    B, P, _ = oPoints.shape
    T = wTo.shape[1]
    wTo_tsl, wTo_6d = wTo[..., :3], wTo[..., 3:]
    wTo_mat = geom_utils.rotation_6d_to_matrix(wTo_6d)
    
    wTo = geom_utils.rt_to_homo(wTo_mat, wTo_tsl)
    cTw = geom_utils.inverse_rt_v2(wTc)  # (B, T, 4, 4)
    cTo = cTw @ wTo
    cTo = cTo.reshape(B*T, 4, 4)

    oPoints_exp = oPoints[:, None].repeat(1, T, 1, 1).reshape(B*T, P, 3)
    cPoints = mesh_utils.apply_transform(oPoints_exp, cTo).reshape(B, T, P, 3)  # (B, T, P, 3)

    intr_exp = intr.unsqueeze(1).repeat(1, T, 1, 1)  # (B, T, 3, 3)

    iPoints = cPoints @ intr_exp.transpose(-2, -1)
    iPoints = iPoints[..., :2] / iPoints[..., 2:3]  # (B, T, P, 2)
    return iPoints


def test_optimization():
    """Test function to demonstrate the optimizer."""
    import pickle

    # Load test data
    fname = 'outputs/tmp.pkl'
 
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    
    print("Testing SE3 trajectory optimization...")
    
    # Set up observations and trajectory to optimize
    x2d = project(data['batch']['newPoints'], data['gt'], data['batch']['wTc'], data['batch']['intr'])

    obs = {
        'x': data['gt'],    # (BS, T, 3+6) translation + 6D rotation - OBSERVATIONS/TARGETS
        'oPoints': data['batch']['newPoints'],  # (BS, P, 3)
        'wTc': data['batch']['wTc'],  # (BS, T, 3+6)
        'intr': data['batch']['intr'],  # (BS, 3, 3)
        'x2d': x2d,
    }
    guidance_mode = "simple_l2"

    guidance_verbose = True
    traj = data['x']  # (BS, T, 3+6) translation + 6D rotation - PARAMETERS TO OPTIMIZE
    

    # Visualize results with proper RGB colors
    traj_list = [
        traj[0],        # Original trajectory
        obs['x'][0],    # Ground truth
    ]
    color_list = [
        [255, 100, 100],  # Light red for original
        [100, 100, 255],  # Light blue for ground truth  
    ]
    
    # vis(traj_list, color_list, "outputs/debug_guidance/vis.rrd")

    # Run optimization
    x_0_pred, _ = do_guidance_optimization(
        traj=traj,
        obs=obs,
        guidance_mode=guidance_mode,
        phase="inner",
        verbose=guidance_verbose,
    )
    
    print(f"Original trajectory shape: {traj.shape}")
    print(f"Ground truth shape: {obs['x'].shape}")
    print(f"Optimized trajectory shape: {x_0_pred.shape}")
    print("Optimization completed successfully!")

    # Visualize results with proper RGB colors
    traj_list = [
        obs['x'][0],    # Ground truth
        traj[0],        # Original trajectory
        x_0_pred[0],    # Optimized trajectory
    ]
    color_list = [ 'red', 'yellow', 'blue' ]
    
    # vis(traj_list, color_list, "outputs/debug_guidance/vis_optimized.rrd")

    print((traj[0] - obs['x'][0]).norm())
    print((x_0_pred[0] - obs['x'][0]).norm())


    pt3d_viz = Pt3dVisualizer(
        exp_name="vis_traj",
        save_dir="outputs/debug_guidance",
        mano_models_dir="assets/mano",
        object_mesh_dir="data/HOT3D-CLIP/object_models_eval/",
    )

    pt3d_viz.log_training_step(
        None, 
        None, 
        # traj_list,
        # color_list,
        [obs['x'][0]],
        ['red'],
        uid='000033',
        pref='debug'
    )

    viz = RerunVisualizer(
        exp_name="vis_traj",
        save_dir="outputs/debug_guidance",
        mano_models_dir="assets/mano",
        object_mesh_dir="data/HOT3D-CLIP/object_models_eval/",
    )
    
    newPoints_meshes = plot_utils.pc_to_cubic_meshes(data['batch']['newPoints'])
    viz.log_dynamic_step(
        None, 
        None, 
        traj_list,
        ['gt', 'original', 'optimized'],
        uid=newPoints_meshes,
        pref='debug'
    )


if __name__ == "__main__":
    test_optimization()