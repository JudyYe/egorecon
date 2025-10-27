"""MANO model, implemented in JAX.

Following the same pattern as SmplhModel in egoallo/fncsmpl_jax.py
"""
from __future__ import annotations
import pickle

from pathlib import Path
from typing import Sequence

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from einops import einsum
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int




@jdc.pytree_dataclass
class MANOModel:
    """The MANO human hand model."""

    side: Int[Array, ""]
    faces: Int[Array, "faces 3"]
    """Vertex indices for mesh faces."""
    J_regressor: Float[Array, "joints+1 verts"]
    """Linear map from vertex to joint positions.
    1 root + 15 hand joints."""
    parent_indices: Int[Array, "joints"]
    """Defines kinematic tree."""
    weights: Float[Array, "verts joints+1"]
    """LBS weights."""
    posedirs: Float[Array, "verts 3 joints*9"]
    """Pose blend shape bases."""
    v_template: Float[Array, "verts 3"]
    """Canonical mesh verts."""
    shapedirs: Float[Array, "verts 3 n_betas"]
    """Shape bases."""
    hands_components: Float[Array, "n_pca_components 45"] | None = None
    """PCA components for hand pose compression."""
    hands_mean: Float[Array, "45"] | None = None
    """Mean hand pose in rotation vector space."""

    @staticmethod
    def load(mano_dir: Path, side: str='right') -> MANOModel:
        # mano_params: dict[str, onp.ndarray] = onp.load(npz_path, allow_pickle=True)
        npz_path = mano_dir / f"MANO_{side.upper()}.pkl"
        with open(npz_path, 'rb') as mano_file:
            mano_params = pickle.load(mano_file, encoding='latin1')
        for k, v in mano_params.items():
            if not isinstance(v, onp.ndarray):
                if k == 'J_regressor':
                    # scipy.sparse._csc.csc_matrix -> onp.ndarray
                    mano_params[k] = onp.array(v.toarray())
                elif k == 'shapedirs':  # shapedirs <class 'chumpy.reordering.Select'> -> onp.ndarray
                    mano_params[k] = onp.array(v)
                else:
                    mano_params[k] = onp.array(v)

        mano_params = {k: _normalize_dtype(v) for k, v in mano_params.items()}
        kintree_table = mano_params["kintree_table"][0] 
        kintree_table[0] = -1
        
        # Load PCA components if available
        hands_components = None
        hands_mean = None
        if "hands_components" in mano_params:
            hands_components = jnp.array(mano_params["hands_components"])
        if "hands_mean" in mano_params:
            hands_mean = jnp.array(mano_params["hands_mean"])
        side= 1 if side == "right" else 0
        side = jnp.array([side])
        return MANOModel(
            side=side,
            faces=jnp.array(mano_params["f"]),
            J_regressor=jnp.array(mano_params["J_regressor"]),
            # parent_indices=jnp.array(mano_params["kintree_table"][0][1:] - 1),
            parent_indices=jnp.array(kintree_table),
            weights=jnp.array(mano_params["weights"]),
            posedirs=jnp.array(mano_params["posedirs"]),
            v_template=jnp.array(mano_params["v_template"]),
            shapedirs=jnp.array(mano_params["shapedirs"]),
            hands_components=hands_components,
            hands_mean=hands_mean,
        )

    def with_shape(
        self, betas: Float[Array | onp.ndarray, "... n_betas"]
    ) -> MANOShaped:
        """Compute a new body model, with betas applied."""
        num_betas = betas.shape[-1]
        assert num_betas <= 16
        verts_with_shape = self.v_template + einsum(
            self.shapedirs[:, :, :num_betas],
            betas,
            "verts xyz beta, ... beta -> ... verts xyz",
        )
        root_and_joints_pred = einsum(
            self.J_regressor,
            verts_with_shape,
            "joints verts, ... verts xyz -> ... joints xyz",
        )  # th_j
        
        # Compute relative joint positions
        rel_joints = root_and_joints_pred.copy()
        root = root_and_joints_pred[..., 0:1, :]
        rest = root_and_joints_pred[..., 1:, :] - root_and_joints_pred[..., self.parent_indices[1:], :]
        rel_joints = jnp.concatenate([root, rest], axis=-2)
        
        return MANOShaped(
            body_model=self,
            verts_zero=verts_with_shape,
            joints_zero=root_and_joints_pred,
            t_parent_joint=rel_joints,
        )


@jdc.pytree_dataclass
class MANOShaped:
    """The MANO body model with a body shape applied."""

    body_model: MANOModel
    verts_zero: Float[Array, "verts 3"]
    """Vertices of shaped body _relative to the root joint_ at the zero
    configuration."""
    joints_zero: Float[Array, "joints 3"]
    """Joints of shaped body _relative to the root joint_ at the zero
    configuration."""
    t_parent_joint: Float[Array, "joints 3"]
    """Position of each shaped body joint relative to its parent."""

    def with_pose(
        self,
        global_orient: Float[Array | onp.ndarray, "... 3"],
        transl: Float[Array | onp.ndarray, "... 3"],
        local_quats: Float[Array | onp.ndarray, "... 15 4"] = None,
        pca: Float[Array | onp.ndarray, "... 15"] = None,
        add_mean: bool = True,
    ) -> MANOShapedAndPosed:
        """Pose the MANO model.

        Args:
            global_orient: Root rotation in axis-angle format
            transl: Translation in world space
            local_quats: 15 hand joint quaternions (optional)
            pca: PCA coefficients for hand pose (optional). If provided, will be converted to quaternions.
            add_mean: Whether to add the mean hand pose when converting from PCA

        Returns:
            MANOShapedAndPosed with poses applied
        """
        # Convert PCA to quaternions if provided
        if pca is not None:
            if local_quats is not None:
                raise ValueError("Cannot specify both local_quats and pca")
            
            # Ensure pca is a JAX array
            pca = jnp.asarray(pca)
            
            # Check if PCA components are available
            if self.body_model.hands_components is None or self.body_model.hands_mean is None:
                raise ValueError("PCA components not loaded in MANO model. Use local_quats instead.")
            
            # Convert PCA coefficients to rotation vectors
            # pca shape: (..., 15)
            # hands_components shape: (15, 45)
            # hands_mean shape: (45,)
            
            # Project PCA coefficients to full rotation vector space
            # R = mean + pca @ components (following SMPLX convention)
            # einsum('...pc,pc->...p', pca, components) where p=15 (pca components), c=45 (rotation vector dims)
            D = pca.shape[-1]
            rotation_vecs = einsum(
                pca,
                self.body_model.hands_components[:D],
                "... n_components, n_components rotvec -> ... rotvec",
            )  # (..., 45)
            
            # Add mean if requested
            if add_mean:
                rotation_vecs = rotation_vecs + self.body_model.hands_mean
            
            # Reshape to (..., 15, 3) - 15 joints, 3D rotation vectors
            rotation_vecs = rotation_vecs.reshape(*pca.shape[:-1], 15, 3)
            
            # Convert rotation vectors to quaternions
            local_quats = jaxlie.SO3.exp(rotation_vecs).wxyz  # (..., 15, 4)
        elif local_quats is None:
            raise ValueError("Must provide either local_quats or pca")
        
        return MANOShapedAndPosed(
            shaped_model=self,
            global_orient=global_orient,
            transl=transl,
            local_quats=local_quats,
        )


@jdc.pytree_dataclass
class MANOShapedAndPosed:
    """The MANO model with shape and pose applied."""

    shaped_model: MANOShaped
    """Underlying shaped body model."""

    global_orient: Float[Array, "... 3"]
    """Root rotation in axis-angle format."""

    transl: Float[Array, "... 3"]
    """Translation vector."""

    local_quats: Float[Array, "... 15 4"]
    """Local joint orientations (15 hand joints)."""

    def lbs(self, mano_order=False) -> MANOMesh:
        """Apply linear blend skinning and return vertices and faces.

        Returns:
            MANOMesh with vertices and faces
        """
        # Convert global_orient to quaternion
        global_orient_quat = jaxlie.SO3.exp(self.global_orient).wxyz

        # Concatenate root orientation with hand joints
        all_quats = jnp.concatenate(
            [global_orient_quat[..., None, :], self.local_quats], axis=-2
        )  # (..., 16, 4)

        # Forward kinematics
        parent_indices = self.shaped_model.body_model.parent_indices
        num_joints = parent_indices.shape[-1]
        assert num_joints == 16

        # Get relative transforms
        Ts_parent_child = jnp.concatenate(
            [all_quats, self.shaped_model.t_parent_joint], axis=-1  
        )  # (..., 16, 7)

        # Compute joint transforms
        identity = jaxlie.SE3.identity().wxyz_xyz
        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                parent_indices[i] == -1,
                # T_world_root,
                # zeros
                identity,
                Ts_world_joint[..., parent_indices[i], :],
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent) @ jaxlie.SE3(Ts_parent_child[..., i, :])
                ).wxyz_xyz
            )

        # Root + 15 hand joints
        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=num_joints,
            body_fun=compute_joint,
            init_val=jnp.zeros_like(Ts_parent_child),
        )

        # Linear blend SKINNING 
        # Apply pose blend shapes
        hand_quats = self.local_quats  # (..., 15, 4)
        verts_with_blend = self.shaped_model.verts_zero + einsum(
            self.shaped_model.body_model.posedirs,
            (jaxlie.SO3(hand_quats).as_matrix() - jnp.eye(3)).flatten(),
            "verts j joints_times_9, ... joints_times_9 -> ... verts j",
        )

        # Get transforms for root and joints
        T_joints = jaxlie.SE3(Ts_world_joint).as_matrix()[..., :3, :]  # (..., 16, 3, 4)
        T_all = T_joints

        # Build joint positions for blend deformation
        joint_positions = self.shaped_model.joints_zero[None, :, :]

        # Blend vertices
        verts_offset = verts_with_blend[..., None, :] - joint_positions
        verts_offset_homo = jnp.pad(
            verts_offset, ((0, 0), (0, 0), (0, 1)), constant_values=1.0
        )

        # Transform vertices
        verts_transformed = einsum(
            T_all,
            self.shaped_model.body_model.weights,
            verts_offset_homo,
            "... joints_p1 i j, verts joints_p1, ... verts joints_p1 j -> ... verts i",
        )

        # Apply translation
        verts_final = verts_transformed + self.transl[..., None, :]
        joints_final = jnp.concatenate([Ts_world_joint[..., 4:7]], axis=0) + self.transl[..., None, :]

        # Add fingertip vertices as joints
        # Use jnp.where to handle the conditional without tracing issues
        side = self.shaped_model.body_model.side
        tips_right = verts_final[..., [745, 317, 444, 556, 673], :]
        tips_left = verts_final[..., [745, 317, 445, 556, 673], :]
        # side is an array, convert to boolean for jnp.where
        is_right = side == 1  # This creates a boolean array
        # jnp.where expects same shape arrays, but we have different shaped outputs
        # So we need to handle this differently by using the index directly
        tips = jnp.where(
            jnp.broadcast_to(is_right, tips_right.shape), 
            tips_right, 
            tips_left
        )

        joints_final = jnp.concatenate([joints_final, tips], axis=-2)
        # Reorder joints to match MANO convention
        if mano_order:
            idx = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            joints_final = joints_final[..., idx, :]

        return MANOMesh(
            posed_model=self,
            verts=verts_final,
            joints=joints_final,
            faces=self.shaped_model.body_model.faces,
        )


@jdc.pytree_dataclass
class MANOMesh:
    posed_model: MANOShapedAndPosed

    verts: Float[Array, "verts 3"]
    """Vertices for mesh."""

    joints: Float[Array, "joints 3"]
    """Joint positions."""

    faces: Int[Array, "faces 3"]
    """Faces for mesh."""


def broadcasting_cat(arrays: Sequence[jax.Array | onp.ndarray], axis: int) -> jax.Array:
    """Like jnp.concatenate, but broadcasts leading axes."""
    assert len(arrays) > 0
    output_dims = max(map(lambda t: len(t.shape), arrays))
    arrays = [
        t.reshape((1,) * (output_dims - len(t.shape)) + t.shape) for t in arrays
    ]
    max_sizes = [max(t.shape[i] for t in arrays) for i in range(output_dims)]
    expanded_arrays = [
        jnp.broadcast_to(
            array,
            tuple(
                array.shape[i] if i == axis % len(array.shape) else max_size
                for i, max_size in enumerate(max_sizes)
            ),
        )
        for array in arrays
    ]
    return jnp.concatenate(expanded_arrays, axis=axis)


def _normalize_dtype(v: onp.ndarray) -> onp.ndarray:
    """Normalize datatypes; all arrays should be either int32 or float32."""
    if "int" in str(v.dtype):
        return v.astype(onp.int32)
    elif "float" in str(v.dtype):
        return v.astype(onp.float32)
    else:
        return v
