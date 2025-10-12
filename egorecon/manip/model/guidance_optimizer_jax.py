"""Optimize SE3 trajectories using Levenberg-Marquardt."""

from __future__ import annotations

from fire import Fire
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
import jaxlie


def do_guidance_optimization(
    traj: Tensor,
    obs: dict,
    guidance_mode: GuidanceMode,
    phase: Literal["inner", "post"],
    verbose: bool,
) -> tuple[Tensor, dict]:
    """Run an optimizer to optimize SE3 trajectory.

    Args:
        traj: SE3 trajectory to optimize (B, T, 7) - wxyz_xyz format [qw, qx, qy, qz, x, y, z]
        obs: Dictionary containing observations/targets, e.g., {'x': gt_trajectory}
        guidance_mode: Guidance mode
        phase: Optimization phase
        verbose: Whether to print optimization progress

    Returns:
        Tuple of (optimized_trajectory, debug_info)
    """
    guidance_params = JaxGuidanceParams.defaults(guidance_mode, phase)

    # Extract trajectory and observations
    x_traj = traj  # Shape: (B, T, 7) - wxyz_xyz format [qw, qx, qy, qz, x, y, z] - PARAMETERS TO OPTIMIZE
    gt_traj = obs[
        "x"
    ]  # Shape: (B, T, 7) - wxyz_xyz format [qw, qx, qy, qz, x, y, z] - OBSERVATIONS/TARGETS
    batch_size, timesteps, _ = x_traj.shape

    start_time = time.time()

    # Convert to JAX arrays and optimize
    x_traj_jax = cast(jax.Array, x_traj.numpy(force=True))
    gt_traj_jax = cast(jax.Array, gt_traj.numpy(force=True))

    # Extract additional parameters for reprojection if available
    oPoints_jax = None
    wTc_jax = None
    intr_jax = None
    x2d_jax = None

    if "oPoints" in obs:
        oPoints_jax = cast(jax.Array, obs["oPoints"].numpy(force=True))
    if "wTc" in obs:
        wTc_jax = cast(jax.Array, obs["wTc"].numpy(force=True))
    if "intr" in obs:
        intr_jax = cast(jax.Array, obs["intr"].numpy(force=True))
    if "x2d" in obs:
        x2d_jax = cast(jax.Array, obs["x2d"].numpy(force=True))

    print(
        "before vmap",
        x_traj_jax.shape,
        gt_traj_jax.shape,
        oPoints_jax.shape,
        wTc_jax.shape,
        intr_jax.shape,
        x2d_jax.shape,
    )
    # Optimize using vmapped function
    optimized_traj, debug_info = _optimize_vmapped(
        x_traj=x_traj_jax,
        gt_traj=gt_traj_jax,
        guidance_params=guidance_params,
        verbose=verbose,
        oPoints=oPoints_jax,
        wTc=wTc_jax,
        intr=intr_jax,
        x2d=x2d_jax,
    )

    # Convert back to torch tensor
    optimized_traj_torch = (
        torch.from_numpy(onp.array(optimized_traj)).to(x_traj.dtype).to(x_traj.device)
    )

    print(f"SE3 trajectory optimization finished in {time.time() - start_time}sec")
    return optimized_traj_torch, debug_info


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


@jdc.jit
def _optimize_vmapped(
    x_traj: jax.Array,
    gt_traj: jax.Array,
    guidance_params: JaxGuidanceParams,
    verbose: jdc.Static[bool],
    oPoints: jax.Array = None,
    wTc: jax.Array = None,
    intr: jax.Array = None,
    x2d: jax.Array = None,
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
            # oPoints=oPoints,
            # wTc=wTc,
            # intr=intr,
            # x2d=x2d,
        )
    )(x_traj=x_traj, gt_traj=gt_traj, oPoints=oPoints, wTc=wTc, intr=intr, x2d=x2d)


# Modes for guidance
GuidanceMode = Literal[
    "simple_l2",  # Simple L2 distance to target
    "with_smoothness",  # L2 + smoothness
    "reproj_corrsp",  # Reprojection loss with points in correspondence
]


@jdc.pytree_dataclass
class JaxGuidanceParams:
    # L2 cost weights
    l2_weight: float = 1.0
    use_l2: jdc.Static[bool] = True

    # Smoothness weights
    smoothness_weight: float = 0.1
    use_smoothness: jdc.Static[bool] = False

    # Reprojection cost weights
    reproj_weight: float = 1.0
    use_reproj: jdc.Static[bool] = False

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
                max_iters=5
                if phase == "inner"
                else 20,  # Much faster for paper performance
            )
        elif mode == "with_smoothness":
            return JaxGuidanceParams(
                use_smoothness=True,
                smoothness_weight=0.1,
                max_iters=3
                if phase == "inner"
                else 5,  # Much faster for paper performance
            )
        elif mode == "reproj_corrsp":
            return JaxGuidanceParams(
                use_l2=False,
                use_reproj=True,
                reproj_weight=1.0,
                max_iters=50
            )
        else:
            assert_never(mode)


def _optimize(
    x_traj: jax.Array,
    gt_traj: jax.Array,
    guidance_params: JaxGuidanceParams,
    verbose: bool,
    oPoints: jax.Array = None,
    wTc: jax.Array = None,
    intr: jax.Array = None,
    x2d: jax.Array = None,
) -> jax.Array:
    """Apply constraints using Levenberg-Marquardt optimizer. Returns updated SE3 trajectory.

    Args:
        x_traj: SE3 trajectory to optimize (T, 7) - wxyz_xyz format [qw, qx, qy, qz, x, y, z] - PARAMETERS
        gt_traj: Ground truth SE3 trajectory (T, 7) - wxyz_xyz format [qw, qx, qy, qz, x, y, z] - OBSERVATIONS
        guidance_params: Guidance parameters
        verbose: Whether to print optimization progress

    Returns:
        Optimized SE3 trajectory (T, 7)
    """
    timesteps = x_traj.shape[0]
    assert x_traj.shape == (timesteps, 7)
    assert gt_traj.shape == (timesteps, 7)
    assert intr.shape == (3, 3)
    assert oPoints.shape == (5000, 3)
    assert x2d.shape == (120, 5000, 2)
    assert wTc.shape == (120, 7)
    print("oPoints", oPoints.shape, wTc.shape, intr.shape, x2d.shape)

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
    if guidance_params.use_l2:

        @cost_with_args(var_traj, gt_traj)
        def l2_distance_cost(
            vals: jaxls.VarValues,
            var_traj: _SE3TrajectoryVar,
            gt_traj: jax.Array,
        ) -> jax.Array:
            """L2 distance cost between optimized trajectory and ground truth observations."""
            current_traj = vals[var_traj]  # (T, 7) - wxyz_xyz format

            # Convert to SE3 and compute distance in tangent space
            current_se3 = jaxlie.SE3(current_traj)
            gt_se3 = jaxlie.SE3(gt_traj)

            # Compute relative transformation and its log (tangent space distance)
            relative_se3 = gt_se3.inverse() @ current_se3
            tangent_distance = relative_se3.log()  # (T, 6)

            return (
                guidance_params.l2_weight
                * (tangent_distance * tangent_distance).flatten()
            )

    # Temporal smoothness cost (optional) - more efficient vectorized version
    if guidance_params.use_smoothness:

        @cost_with_args(var_traj)
        def smoothness_cost(
            vals: jaxls.VarValues,
            *args,
        ) -> jax.Array:
            """Temporal smoothness cost between consecutive frames - vectorized."""
            current_traj = vals[var_traj]  # (T, 7) - wxyz_xyz format

            # Convert to SE3 and compute relative transformations
            current_se3 = jaxlie.SE3(current_traj)

            # Compute relative transformations between consecutive frames
            relative_se3 = current_se3[1:].inverse() @ current_se3[:-1]
            tangent_distance = relative_se3.log()  # (T-1, 6)

            return guidance_params.smoothness_weight * tangent_distance.flatten()

    # Reprojection cost (optional) - project object points and compare with 2D observations
    if guidance_params.use_reproj:
        assert (
            oPoints is not None
            and wTc is not None
            and intr is not None
            and x2d is not None
        )
        print("before @", x2d.shape, wTc.shape, oPoints.shape, intr.shape)

        @cost_with_args(
            _SE3TrajectoryVar(jnp.arange(timesteps)),
            x2d,
            wTc,
            oPoints[None],
            intr[None],
        )
        def reprojection_cost(
            vals: jaxls.VarValues,
            var_traj: _SE3TrajectoryVar,
            x2d: jax.Array,
            wTc: jax.Array,
            oPoints: jax.Array,
            intr: jax.Array,
        ) -> jax.Array:
            """Reprojection cost: project object points and compare with 2D observations."""
            current_traj = vals[var_traj]  # (7, ) - wxyz_xyz format

            # inside (7,) (5000, 2) (7,) (5000, 3) (3, 3)
            # Project object points using current trajectory
            projected_2d = project_jax(oPoints, current_traj, wTc, intr, ndc=True)  # (T, P, 2)

            # Compute L2 distance between projected and observed 2D points
            diff = projected_2d - x2d  # (P, 2)
            return guidance_params.reproj_weight * (diff * diff).flatten()

    # Set up the optimization problem
    graph = jaxls.LeastSquaresProblem(costs=factors, variables=[var_traj]).analyze()

    # Solve using Levenberg-Marquardt
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make([var_traj.with_value(x_traj)]),
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


def project(oPoints, wTo, wTc, intr, ndc=False, H=None, W=None):
    B, P, _ = oPoints.shape
    T = wTo.shape[1]

    wTo_tsl, wTo_6d = wTo[..., :3], wTo[..., 3:]
    wTo_mat = geom_utils.rotation_6d_to_matrix(wTo_6d)
    wTo = geom_utils.rt_to_homo(wTo_mat, wTo_tsl)

    wTc_tsl, wTc_6d = wTc[..., :3], wTc[..., 3:]
    wTc_mat = geom_utils.rotation_6d_to_matrix(wTc_6d)
    wTc = geom_utils.rt_to_homo(wTc_mat, wTc_tsl)

    cTw = geom_utils.inverse_rt_v2(wTc)  # (B, T, 4, 4)
    cTo = cTw @ wTo
    cTo = cTo.reshape(B * T, 4, 4)

    oPoints_exp = oPoints[:, None].repeat(1, T, 1, 1).reshape(B * T, P, 3)
    cPoints = mesh_utils.apply_transform(oPoints_exp, cTo).reshape(
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
        print(intr.shape, H.shape, iPoints.shape)
        iPoints[..., 0] = iPoints[..., 0] / W.reshape(B, 1, 1) * 2 - 1
        iPoints[..., 1] = iPoints[..., 1] / H.reshape(B, 1, 1) * 2 - 1
    return iPoints


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
        # iPoints[:, 0] = iPoints[:, 0] / W * 2 - 1
        # iPoints[:, 1] = iPoints[:, 1] / H * 2 - 1
        
    return iPoints


def test_optimization(mode="reproj_corrsp"):
    """Test function to demonstrate the optimizer."""
    import pickle

    # Load test data
    fname = "outputs/tmp.pkl"

    with open(fname, "rb") as f:
        data = pickle.load(f)

    print("Testing SE3 trajectory optimization...")

    # Set up observations and trajectory to optimize
    wTc = geom_utils.se3_to_matrix_v2(data["batch"]["wTc"])
    intr = data["batch"]["intr"]
    print("wTc", "intr", wTc.shape, intr.shape)

    x2d = project(
        data["batch"]["newPoints"],
        data["gt"],
        data["batch"]["wTc"],
        data["batch"]["intr"],
        ndc=True,
    )


    obs = {
        "x": se3_to_wxyz_xyz(
            data["gt"]
        ),  # (BS, T, 7) wxyz_xyz format - OBSERVATIONS/TARGETS
        "oPoints": data["batch"]["newPoints"],  # (BS, P, 3)
        "wTc": se3_to_wxyz_xyz(data["batch"]["wTc"]),  # (BS, T, 7) wxyz_xyz format
        "intr": data["batch"]["intr"],  # (BS, 3, 3)
        "x2d": x2d,
    }
    guidance_mode = mode  # "reproj_corrsp"  # Test reprojection mode

    # # Convert to JAX arrays for testing
    # oPoints_jax = cast(jax.Array, data["batch"]["newPoints"][0].numpy(force=True))  # (P, 3)
    # wTc_jax = cast(jax.Array, obs["wTc"][0].numpy(force=True))  # (T, 7)
    # wTo_jax = cast(jax.Array, obs["x"][0].numpy(force=True))  # (T, 7)
    # intr_jax = cast(jax.Array, intr[0].numpy(force=True))  # (3, 3)
    
    # print('oPoints', oPoints_jax.shape, wTc_jax.shape, wTo_jax.shape, intr_jax.shape)
    
    # # Use vmap to project across all timesteps
    # iPoints = jax.vmap(
    #     partial(project_jax, oPoints=oPoints_jax, intr=intr_jax)
    # )(wTo=wTo_jax, wTc=wTc_jax)

    # print("iPoints", iPoints.shape, type(iPoints))

    # print(x2d[0], iPoints)

    guidance_verbose = True
    traj = data["x"]  # (BS, T, 7) wxyz_xyz format - PARAMETERS TO OPTIMIZE

    # Visualize results with proper RGB colors
    traj_list = [
        traj[0],  # Original trajectory
        obs["x"][0],  # Ground truth
    ]

    # vis(traj_list, color_list, "outputs/debug_guidance/vis.rrd")

    # Run optimization
    x_0_pred, _ = do_guidance_optimization(
        traj=se3_to_wxyz_xyz(traj),
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
        obs["x"][0],  # Ground truth
        traj[0],  # Original trajectory
        x_0_pred[0],  # Optimized trajectory
    ]
    # color_list = [ 'red', 'yellow', 'blue' ]  # Not used in this version

    # vis(traj_list, color_list, "outputs/debug_guidance/vis_optimized.rrd")

    # Compute SE3 distance using tangent space
    traj_se3 = jaxlie.SE3(traj[0])
    gt_se3 = jaxlie.SE3(obs["x"][0])
    pred_se3 = jaxlie.SE3(x_0_pred[0])

    # original_distance = (gt_se3.inverse() @ traj_se3).log()
    # optimized_distance = (gt_se3.inverse() @ pred_se3).log()

    # print(f"Original trajectory distance: {jnp.linalg.norm(original_distance)}")
    # print(f"Optimized trajectory distance: {jnp.linalg.norm(optimized_distance)}")

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
        [obs["x"][0]],
        ["red"],
        uid="000033",
        pref="debug",
    )


    B, T, J_3 = data["batch"]["hand_raw"].shape
    J = J_3 // 3  // 2
    hands = data["batch"]["hand_raw"]
    left_hands, right_hands = torch.split(hands, J_3 // 2, dim=-1)
    left_hands = left_hands.reshape(B, T, J, 3)
    right_hands = right_hands.reshape(B, T, J, 3)

    left_hand_mesh = plot_utils.pc_to_cubic_meshes(left_hands[0], eps=0.02)
    left_hand_mesh.textures = mesh_utils.pad_texture(left_hand_mesh, 'white')
    right_hand_mesh = plot_utils.pc_to_cubic_meshes(right_hands[0], eps=0.02)
    right_hand_mesh.textures = mesh_utils.pad_texture(right_hand_mesh, 'blue')

        

    viz = RerunVisualizer(
        exp_name="vis_traj",
        save_dir="outputs/debug_guidance",
        mano_models_dir="assets/mano",
        object_mesh_dir="data/HOT3D-CLIP/object_models_eval/",
    )

    newPoints_meshes = plot_utils.pc_to_cubic_meshes(data["batch"]["newPoints"])
    viz.log_dynamic_step(
        left_hand_mesh,
        right_hand_mesh,
        traj_list,
        ["gt", "original", "optimized"],
        uid=newPoints_meshes,
        pref="debug",
        wTc_list=wTc[0],
        focal=intr[0],
    )


def se3_to_wxyz_xyz(se3):
    tsl, rot6d = torch.split(se3, [3, 6], dim=-1)
    mat = geom_utils.rotation_6d_to_matrix(rot6d)
    wxyz = geom_utils.rot_cvt.matrix_to_quaternion(mat)

    return torch.cat([wxyz, tsl], dim=-1)


if __name__ == "__main__":
    Fire(test_optimization)
