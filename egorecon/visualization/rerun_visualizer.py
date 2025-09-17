#!/usr/bin/env python3
"""Concise Rerun visualizer based on proven working code"""

import os
import pickle
import numpy as np
import rerun as rr
import torch
import shutil
from pathlib import Path


class RerunVisualizer:
    """Concise Rerun visualizer for hand-to-object training"""

    def __init__(
        self,
        exp_name,
        save_dir,
        enable_visualization=True,
        mano_models_dir="data/mano_models",
        object_mesh_dir="data/object_meshes",
        use_hand_articulations=False,
    ):
        """Initialize visualizer"""
        self.exp_name = exp_name
        self.save_dir = Path(save_dir)
        self.enable_visualization = enable_visualization
        self.mano_models_dir = Path(mano_models_dir)
        self.object_mesh_dir = Path(object_mesh_dir)
        self.use_hand_articulations = use_hand_articulations

        self.rerun_dir = None
        self.left_mano = None
        self.right_mano = None
        self.hand_articulations = None
        self.object_meshes = {}  # Cache loaded meshes

        if self.enable_visualization:
            self._setup()

    def _setup(self):
        """Setup Rerun and load models"""
        try:
            # Initialize Rerun
            rr.init(f"HandToObject_{self.exp_name}")
            self.rerun_dir = self.save_dir / "rerun_visualizations"
            self.rerun_dir.mkdir(parents=True, exist_ok=True)

            # Start recording immediately (like the working version)
            main_output_file = self.rerun_dir / "training_session.rrd"
            rr.save(str(main_output_file))

            # Load MANO models
            self._load_mano_models()

            # Load hand articulations if requested
            if self.use_hand_articulations:
                self._load_hand_articulations()

            print(
                f"✓ RerunVisualizer ready: MANO={'✓' if self.left_mano else '✗'}, "
                f"Articulations={'✓' if self.hand_articulations else '✗'}"
            )

        except Exception as e:
            print(f"⚠️ Visualization setup failed: {e}")
            self.enable_visualization = False

    def _load_mano_models(self):
        """Load MANO models using proven working method"""
        try:
            import smplx

            # Check files exist
            left_src = self.mano_models_dir / "MANO_LEFT.pkl"
            right_src = self.mano_models_dir / "MANO_RIGHT.pkl"

            if not left_src.exists() or not right_src.exists():
                print(f"⚠️ MANO files not found in {self.mano_models_dir}")
                return

            # Create temp directory (proven working method)
            mano_temp_dir = "/tmp/mano_models"
            os.makedirs(mano_temp_dir, exist_ok=True)

            # Copy files
            shutil.copy2(str(left_src), os.path.join(mano_temp_dir, "MANO_LEFT.pkl"))
            shutil.copy2(str(right_src), os.path.join(mano_temp_dir, "MANO_RIGHT.pkl"))

            # Load models
            self.left_mano = smplx.MANO(
                model_path=os.path.join(mano_temp_dir, "MANO_LEFT.pkl"),
                is_rhand=False,
                use_pca=False,
                flat_hand_mean=True,
            )
            self.right_mano = smplx.MANO(
                model_path=os.path.join(mano_temp_dir, "MANO_RIGHT.pkl"),
                is_rhand=True,
                use_pca=False,
                flat_hand_mean=True,
            )

        except ImportError:
            print("⚠️ smplx not available")
        except Exception as e:
            print(f"⚠️ MANO loading failed: {e}")

    def _load_hand_articulations(self):
        """Load hand articulations data"""
        if not self.hand_articulations_path.exists():
            print(f"⚠️ Hand articulations not found: {self.hand_articulations_path}")
            return

        try:
            with open(self.hand_articulations_path, "rb") as f:
                self.hand_articulations = pickle.load(f)
            print(
                f"✓ Loaded hand articulations from {len(self.hand_articulations)} sequences"
            )
        except Exception as e:
            print(f"⚠️ Hand articulations loading failed: {e}")

    def load_object_mesh(self, object_id):
        """Load object mesh using proven working method"""
        if object_id in self.object_meshes:
            return self.object_meshes[object_id]

        try:
            import trimesh

            # Look for GLB files (proven to work)
            mesh_path = self.object_mesh_dir / f"{object_id}.glb"
            if not mesh_path.exists():
                return None

            # Load mesh
            mesh = trimesh.load(mesh_path)

            # Handle Scene objects (GLB files load as scenes)
            if hasattr(mesh, "geometry") and mesh.geometry:
                # Get the first geometry from the scene
                geometry = list(mesh.geometry.values())[0]
                mesh_data = {
                    "vertices": geometry.vertices,
                    "faces": geometry.faces,
                    "path": mesh_path,
                }
            else:
                # Direct mesh object
                mesh_data = {
                    "vertices": mesh.vertices,
                    "faces": mesh.faces,
                    "path": mesh_path,
                }

            self.object_meshes[object_id] = mesh_data
            return mesh_data

        except ImportError:
            print("⚠️ trimesh not available for GLB loading")
            return None
        except Exception as e:
            print(f"⚠️ Mesh loading failed for {object_id}: {e}")
            return None

    def _apply_coordinate_transform(self, position):
        """Apply coordinate transformation (from proven working code)"""
        transform_matrix = np.array(
            [
                [1, 0, 0],  # X stays the same
                [0, 1, 0],  # Y becomes -Z
                [0, 0, 1],  # Z becomes Y
            ]
        )
        return transform_matrix @ position

    def log_training_step(
        self,
        step,
        left_hand,
        right_hand,
        object_gt,
        object_pred=None,
        object_noisy=None,
        seq_len=None,
        is_moving=None,
        mean_velocity=None,
        pref='training/',
    ):
        """Log training step with trajectories and MANO hands"""
        if not self.enable_visualization:
            return

        # try:
        # Handle both integer and string step values
        if isinstance(step, str):
            # Extract numeric part from strings like "best_1000" or "eval_5"
            import re

            numeric_part = re.findall(r"\d+", step)
            step_num = int(numeric_part[0]) if numeric_part else 0
        else:
            step_num = step

        # rr.set_time("training_step", sequence=step_num)
        rr.set_time_sequence("training_step", step_num)

        # Extract positions
        if left_hand.shape[-1] == 21 * 3:
            left_hand = left_hand.reshape(1, -1, 21, 3)
            right_hand = right_hand.reshape(1, -1, 21, 3)

        left_pos = left_hand[0, :, 0, :3].cpu().numpy()  # [T, 3]
        right_pos = right_hand[0, :, 0, :3].cpu().numpy()
        object_pos_gt = object_gt[0, :, :3].cpu().numpy()

        # Apply valid length
        valid_len = seq_len.item() if seq_len is not None else len(left_pos)
        valid_len = min(valid_len, len(left_pos), len(right_pos), len(object_pos_gt))

        left_pos = left_pos[:valid_len]
        right_pos = right_pos[:valid_len]
        object_pos_gt = object_pos_gt[:valid_len]

        # Log trajectories
        rr.log(
            f"{pref}/left_hand_traj",
            rr.LineStrips3D([left_pos], colors=[[0, 255, 0]], radii=[0.01]),
        )
        rr.log(
            f"{pref}/right_hand_traj",
            rr.LineStrips3D([right_pos], colors=[[0, 0, 255]], radii=[0.01]),
        )
        rr.log(
            f"{pref}/object_gt_traj",
            rr.LineStrips3D([object_pos_gt], colors=[[255, 0, 0]], radii=[0.015]),
        )

        # Log hand pose as point cloud [B=0, T=0/-1, J=21, 3] the the start and last frame
        for t in [0, -1]:
            rr.log(
                f"{pref}/left_hand_pose/frame_{t}",
                rr.Points3D(
                    left_hand[0, t, :, :3].cpu().numpy(),
                    colors=[[0, 255, 0]],
                    radii=[0.01],
                ),
            )
            rr.log(
                f"{pref}/right_hand_pose/frame_{t}",
                rr.Points3D(
                    right_hand[0, t, :, :3].cpu().numpy(),
                    colors=[[0, 0, 255]],
                    radii=[0.01],
                ),
            )

        # Log prediction if available
        if object_pred is not None:
            object_pos_pred = object_pred[0, :valid_len, :3].cpu().numpy()
            rr.log(
                f"{pref}/object_pred_traj",
                rr.LineStrips3D(
                    [object_pos_pred], colors=[[255, 165, 0]], radii=[0.015]
                ),
            )
        if object_noisy is not None:
            print('object_noisy shape: ', object_noisy.shape, object_gt.shape)
            object_pos_noisy = object_noisy[0, :valid_len, :3].cpu().numpy()
            rr.log(
                f"{pref}/object_noisy_traj",
                rr.LineStrips3D(
                    [object_pos_noisy], colors=[[255, 0, 255]], radii=[0.015]
                ),
            )

        # Log MANO hands at current position
        if valid_len > 0 and self.left_mano is not None:
            self._log_mano_hands(left_pos[-1], right_pos[-1], pref)

        # Log metadata
        metadata = f"Step: {step}"
        if seq_len is not None:
            metadata += f", Seq: {seq_len.item()}"
        if is_moving is not None:
            metadata += f", Moving: {is_moving}"
        if mean_velocity is not None:
            metadata += f", Vel: {mean_velocity:.4f}"
        rr.log(f"{pref}/info", rr.TextDocument(metadata))

        # Note: We're recording continuously to training_session.rrd, no need for periodic saves
        # except Exception as e:
        #     print(f"⚠️ Training step logging failed: {e}")

    def _log_mano_hands(self, left_pos, right_pos, pref):
        """Log MANO hand meshes at positions"""
        try:
            # Left hand
            with torch.no_grad():
                left_output = self.left_mano()
                left_vertices = left_output.vertices[0].numpy() + left_pos
                rr.log(
                    f"{pref}/left_hand_mesh",
                    rr.Mesh3D(
                        vertex_positions=left_vertices,
                        triangle_indices=self.left_mano.faces,
                        vertex_colors=[0.3, 0.8, 0.3],
                    ),
                )

            # Right hand
            with torch.no_grad():
                right_output = self.right_mano()
                right_vertices = right_output.vertices[0].numpy() + right_pos
                rr.log(
                    f"{pref}/right_hand_mesh",
                    rr.Mesh3D(
                        vertex_positions=right_vertices,
                        triangle_indices=self.right_mano.faces,
                        vertex_colors=[0.3, 0.3, 0.8],
                    ),
                )

        except Exception as e:
            print(f"⚠️ MANO hand logging failed: {e}")

    def log_best_model_prediction(
        self,
        step,
        left_hand,
        right_hand,
        object_gt,
        diffusion_model,
        device,
        seq_len=None,
        is_moving=None,
        mean_velocity=None,
    ):
        """Generate and log best model prediction"""
        if not self.enable_visualization:
            return

        # Log with prediction
        self.log_training_step(
            f"best_{step}",
            left_hand,
            right_hand,
            object_gt,
            object_pred,
            seq_len,
            is_moving,
            mean_velocity,
        )

    def log_final_trajectory(self, dataset, sampled_trajectory):
        """Log final trajectory comparison with object mesh"""
        if not self.enable_visualization:
            return

        try:
            rr.set_time_sequence("final", 0)

            # Get trajectory data
            left_traj = dataset.left_hand_full[:, :3].numpy()
            right_traj = dataset.right_hand_full[:, :3].numpy()
            object_gt_traj = dataset.object_motion_full[:, :3].numpy()
            object_pred_traj = sampled_trajectory[:, :3].numpy()

            # Log trajectories
            rr.log(
                "final/left_hand",
                rr.LineStrips3D([left_traj], colors=[[0, 255, 0]], radii=[0.008]),
            )
            rr.log(
                "final/right_hand",
                rr.LineStrips3D([right_traj], colors=[[0, 0, 255]], radii=[0.008]),
            )
            rr.log(
                "final/object_gt",
                rr.LineStrips3D([object_gt_traj], colors=[[255, 0, 0]], radii=[0.012]),
            )
            rr.log(
                "final/object_pred",
                rr.LineStrips3D(
                    [object_pred_traj], colors=[[255, 165, 0]], radii=[0.012]
                ),
            )

            # Load and visualize object mesh if available
            object_id = getattr(dataset, "target_object_id", None)
            if object_id:
                mesh = self.load_object_mesh(object_id)
                if mesh:
                    # Show mesh at key positions
                    step_size = max(
                        1, len(object_pred_traj) // 20
                    )  # Show ~20 positions
                    for i in range(0, len(object_pred_traj), step_size):
                        # GT mesh
                        gt_vertices = mesh["vertices"] + object_gt_traj[i]
                        rr.log(
                            f"final/object_mesh_gt/frame_{i}",
                            rr.Mesh3D(
                                vertex_positions=gt_vertices,
                                triangle_indices=mesh["faces"],
                                vertex_colors=[0.8, 0.2, 0.2],
                            ),
                        )

                        # Predicted mesh
                        pred_vertices = mesh["vertices"] + object_pred_traj[i]
                        rr.log(
                            f"final/object_mesh_pred/frame_{i}",
                            rr.Mesh3D(
                                vertex_positions=pred_vertices,
                                triangle_indices=mesh["faces"],
                                vertex_colors=[0.8, 0.5, 0.1],
                            ),
                        )

            print(f"✓ Final visualization logged to continuous recording")

        except Exception as e:
            print(f"⚠️ Final trajectory logging failed: {e}")

    def setup_for_overfit_training(self, dataset, object_id=None):
        """Setup for overfit training"""
        if not self.enable_visualization:
            return

        # Pre-load object mesh if provided
        if object_id:
            mesh = self.load_object_mesh(object_id)
            if mesh:
                print(f"✓ Pre-loaded object mesh for {object_id}")

    def get_summary(self):
        """Get concise summary"""
        if not self.enable_visualization:
            return "Visualization: Disabled"

        features = []
        features.append("Hand trajectories")
        features.append("Object trajectories")
        if self.left_mano:
            features.append("MANO hand meshes")
        if self.hand_articulations:
            features.append("Hand articulations")
        if self.object_meshes:
            features.append(f"Object meshes ({len(self.object_meshes)})")

        summary = f"🎯 RerunVisualizer: {', '.join(features)}"
        if self.rerun_dir:
            summary += f"\n📁 Files: {self.rerun_dir}"
            summary += (
                f"\n🎆 View: rerun {self.rerun_dir}/training_session.rrd --web-viewer"
            )

        return summary
