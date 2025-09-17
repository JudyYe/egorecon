import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from tqdm import tqdm

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.manip.data.hand_to_object_dataset import HandToObjectDataset
from src.manip.model.transformer_hand_to_object_diffusion_model import \
    CondGaussianDiffusion
from src.visualization.rerun_visualizer import RerunVisualizer


def sample_full_trajectory_inpaint(
    diffusion_model, full_dataset, device, window_size=120, overlap=60
):
    """
    Sample full trajectory using inpainting approach with sliding windows.
    Works with any dataset split (train/val/all).
    """
    if not full_dataset.has_full_trajectory_data():
        print("Warning: Dataset doesn't have full trajectory data.")
        return None

    print(
        f"\nSampling full trajectory using inpainting (window={window_size}, overlap={overlap})..."
    )
    full_length = full_dataset.full_length
    left_hand_full = full_dataset.left_hand_full.to(device)
    right_hand_full = full_dataset.right_hand_full.to(device)
    hand_poses = torch.cat([left_hand_full, right_hand_full], dim=-1).unsqueeze(0)

    step_size = window_size - overlap
    num_windows = (full_length - overlap + step_size - 1) // step_size

    sampled_trajectory = torch.zeros_like(full_dataset.object_motion_full).to(device)

    diffusion_model.eval()
    with torch.no_grad():
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, full_length)
            actual_window_size = end_idx - start_idx

            hand_window = hand_poses[:, start_idx:end_idx, :]

            if actual_window_size < window_size:
                pad_len = window_size - actual_window_size
                hand_window = torch.cat(
                    [
                        hand_window,
                        torch.zeros(1, pad_len, hand_window.shape[2]).to(device),
                    ],
                    dim=1,
                )

            # Create initial object trajectory (zeros as starting point)
            object_motion_init = torch.zeros(1, window_size, full_dataset.pose_dim).to(
                device
            )
            sampled_window = diffusion_model.sample(object_motion_init, hand_window)
            sampled_trajectory[start_idx:end_idx] = sampled_window[
                0, :actual_window_size
            ]

    print(f"Full trajectory sampling completed: {sampled_trajectory.shape}")
    return sampled_trajectory.cpu()


def sample_full_trajectory_sliding_window(
    diffusion_model, full_dataset, device, window_size=120, overlap=60
):
    """
    Sample the full trajectory using sliding windows with overlap.
    Works with any dataset split (train/val/all).
    """
    if not full_dataset.has_full_trajectory_data():
        print("Warning: Dataset doesn't have full trajectory data.")
        return None

    print(
        f"\nSampling full trajectory using sliding windows (window={window_size}, overlap={overlap})..."
    )

    full_length = full_dataset.full_length
    left_hand_full = full_dataset.left_hand_full.to(device)
    right_hand_full = full_dataset.right_hand_full.to(device)

    # Initialize output trajectory
    sampled_trajectory = torch.zeros_like(full_dataset.object_motion_full).to(device)
    weight_map = torch.zeros(full_length).to(device)

    step_size = window_size - overlap
    num_windows = (full_length - overlap + step_size - 1) // step_size

    diffusion_model.eval()
    with torch.no_grad():
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, full_length)
            actual_window_size = end_idx - start_idx

            print(
                f"Processing window {i + 1}/{num_windows}: frames {start_idx}-{end_idx}"
            )

            # Extract window
            left_hand_window = left_hand_full[start_idx:end_idx].unsqueeze(0)
            right_hand_window = right_hand_full[start_idx:end_idx].unsqueeze(0)

            # Pad if necessary
            if actual_window_size < window_size:
                pad_len = window_size - actual_window_size
                left_hand_window = torch.cat(
                    [
                        left_hand_window,
                        torch.zeros(1, pad_len, full_dataset.pose_dim).to(device),
                    ],
                    dim=1,
                )
                right_hand_window = torch.cat(
                    [
                        right_hand_window,
                        torch.zeros(1, pad_len, full_dataset.pose_dim).to(device),
                    ],
                    dim=1,
                )

            # Prepare input
            hand_poses = torch.cat([left_hand_window, right_hand_window], dim=-1)
            object_motion_init = torch.zeros(1, window_size, full_dataset.pose_dim).to(
                device
            )

            # Generate padding mask
            seq_len_tensor = torch.tensor([actual_window_size + 1]).to(device)
            tmp_mask = torch.arange(window_size + 1, device=device).expand(
                1, window_size + 1
            ) < seq_len_tensor[:, None].repeat(1, window_size + 1)
            padding_mask = tmp_mask[:, None, :]

            # Sample window
            sampled_window = diffusion_model.sample(
                object_motion_init, hand_poses, padding_mask=padding_mask
            )
            sampled_window = sampled_window[0, :actual_window_size]

            # Add to output with overlap handling
            sampled_trajectory[start_idx:end_idx] += sampled_window
            weight_map[start_idx:end_idx] += 1.0

    # Average overlapping regions
    sampled_trajectory = sampled_trajectory / weight_map.unsqueeze(-1)

    print(f"Full trajectory sampling completed: {sampled_trajectory.shape}")
    return sampled_trajectory.cpu()


def evaluate_model(
    diffusion_model,
    val_dataset,
    device,
    num_eval_windows=10,
    visualizer: RerunVisualizer = None,
    global_step=0,
):
    """
    Evaluate the model by sampling from multiple windows and computing trajectory accuracy.
    Returns dictionary with position, rotation, and combined errors.
    """
    diffusion_model.eval()

    total_position_errors = []
    total_max_errors = []
    total_rotation_errors = []
    total_combined_errors = []

    with torch.no_grad():
        # Evaluate on a subset of windows
        eval_windows = min(num_eval_windows, len(val_dataset.windows))

        for i in range(eval_windows):
            # Get a window using __getitem__ method
            sample = val_dataset[i]

            # Extract data from sample
            cond = (
                sample["condition"].unsqueeze(0).to(device)
            )  # [1, T, 2*D] - left + right hand
            object_motion_gt = (
                sample["target_raw"].unsqueeze(0).to(device)
            )  # [1, T, D] - unnormalized ground truth
            seq_len = torch.tensor([cond.shape[1]]).to(
                device
            )  # [1] - actual sequence length

            # Prepare input
            object_motion_init = torch.randn_like(object_motion_gt).to(device)

            # Generate padding mask
            actual_seq_len = seq_len + 1
            tmp_mask = torch.arange(val_dataset.window_size + 1, device=device).expand(
                1, val_dataset.window_size + 1
            ) < actual_seq_len[:, None].repeat(1, val_dataset.window_size + 1)
            padding_mask = tmp_mask[:, None, :]

            # Sample from model
            # import pdb; pdb.set_trace()
            sampled_motion, _ = diffusion_model.sample_raw(
                object_motion_init,
                cond,
                padding_mask=padding_mask,
            )
            # sampled_motion = diffusion_model.sample(object_motion_init, hand_poses, padding_mask=padding_mask, )

            # Compute position errors (only for valid sequence length)
            valid_len = seq_len.item()

            # Extract positions and rotations
            if val_dataset.use_velocity:
                # For 12D: translation (0:3), velocity (3:6), rotation (6:12)
                gt_positions = object_motion_gt[0, :valid_len, 0:3]  # [T, 3]
                pred_positions = sampled_motion[0, :valid_len, 0:3]  # [T, 3]
                gt_rotations = object_motion_gt[0, :valid_len, 6:12]  # [T, 6]
                pred_rotations = sampled_motion[0, :valid_len, 6:12]  # [T, 6]
            else:
                # For 9D: translation (0:3), rotation (3:9)
                gt_positions = object_motion_gt[0, :valid_len, 0:3]  # [T, 3]
                pred_positions = sampled_motion[0, :valid_len, 0:3]  # [T, 3]
                gt_rotations = object_motion_gt[0, :valid_len, 3:9]  # [T, 6]
                pred_rotations = sampled_motion[0, :valid_len, 3:9]  # [T, 6]

            # Compute position errors (L2 distance per frame)
            position_errors = torch.norm(pred_positions - gt_positions, dim=1)  # [T]

            # Compute rotation errors (L2 distance in 6D rotation space)
            rotation_errors = torch.norm(pred_rotations - gt_rotations, dim=1)  # [T]

            # Combined error (weighted sum or separate tracking)
            combined_errors = (
                position_errors + rotation_errors
            )  # Weight the same for now

            # Store metrics
            mean_pos_error = position_errors.mean().item()
            max_pos_error = position_errors.max().item()
            mean_rot_error = rotation_errors.mean().item()
            mean_combined_error = combined_errors.mean().item()

            total_position_errors.append(mean_pos_error)
            total_max_errors.append(max_pos_error)
            total_rotation_errors.append(mean_rot_error)
            total_combined_errors.append(mean_combined_error)

            # visualize
            hand_raw = sample["hand_raw"].unsqueeze(0).to(device)
            left_hand_raw, right_hand_raw = torch.split(hand_raw, 21 * 3, dim=-1)
            object_motion_raw = sample["target_raw"].unsqueeze(0).to(device)
            object_pred_raw = sampled_motion
            is_moving = sample["is_motion"]
            mean_velocity = sample["mean_velocity"]

            if visualizer:
                print(
                    "hand",
                    left_hand_raw.shape,
                    right_hand_raw.shape,
                    object_motion_raw.shape,
                    object_pred_raw.shape,
                )
                visualizer.log_training_step(
                    global_step,
                    left_hand_raw,
                    right_hand_raw,
                    object_motion_raw,
                    object_noisy=sample["traj_noisy_raw"].unsqueeze(0).to(device),
                    object_pred=object_pred_raw,
                    seq_len=seq_len,
                    is_moving=is_moving,
                    mean_velocity=mean_velocity,
                    pref="val/",
                )

    # Compute overall metrics
    overall_mean_pos_error = sum(total_position_errors) / len(total_position_errors)
    overall_max_pos_error = max(total_max_errors)
    overall_mean_rot_error = sum(total_rotation_errors) / len(total_rotation_errors)
    overall_mean_combined_error = sum(total_combined_errors) / len(
        total_combined_errors
    )

    diffusion_model.train()

    return {
        "mean_position_error": overall_mean_pos_error,
        "max_position_error": overall_max_pos_error,
        "mean_rotation_error": overall_mean_rot_error,
        "mean_combined_error": overall_mean_combined_error,
    }


def train_overfit(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Load dataset

    train_dataset = HandToObjectDataset(
        opt.data_path,
        window_size=opt.window_size,
        use_velocity=opt.use_velocity,
        single_demo=opt.demo_id,
        single_object=opt.target_object_id,
        sampling_strategy="random",
        split="mini",
        split_seed=42,  # Ensure reproducible splits
        noise_scheme="syn",
        noise_std_obj_rot=opt.noise_std_obj_rot,
        noise_std_obj_trans=opt.noise_std_obj_trans,
        noise_std_mano_global_rot=opt.noise_std_mano_global_rot,
        noise_std_mano_body_rot=opt.noise_std_mano_body_rot,
        noise_std_mano_trans=opt.noise_std_mano_trans,
        noise_std_mano_betas=opt.noise_std_mano_betas,
    )

    val_dataset = HandToObjectDataset(
        opt.data_path,
        window_size=opt.window_size,
        use_velocity=opt.use_velocity,
        single_demo=opt.demo_id,
        single_object=opt.target_object_id,
        sampling_strategy="random",
        split="mini",
        split_seed=42,  # Use same seed for consistent splits
        noise_scheme="real",
    )

    # Create combined dataset for final evaluation
    full_dataset = HandToObjectDataset(
        opt.data_path,
        window_size=opt.window_size,
        use_velocity=opt.use_velocity,
        single_demo=opt.demo_id,
        single_object=opt.target_object_id,
        sampling_strategy="random",
        split="mini",  # Use all data
        split_seed=42,
    )
    print(
        f"  Successfully loaded dataset from demo {train_dataset.demo_id}, object {train_dataset.target_object_id}"
    )

    # Define model - use window size for model architecture
    repr_dim = train_dataset.pose_dim  # Output dimension (3D translation + 6D rotation)
    cond_dim = 2 * 21 * 3 + 9  # Input dimension (2 hands × pose_dim each)

    diffusion_model = CondGaussianDiffusion(
        opt,
        d_feats=repr_dim,
        condition_dim=cond_dim,
        d_model=opt.d_model,
        n_head=opt.n_head,
        n_dec_layers=opt.n_dec_layers,
        d_k=opt.d_k,
        d_v=opt.d_v,
        max_timesteps=opt.window_size + 1,
        out_dim=repr_dim,
        timesteps=1000,
        loss_type="l1",
        objective="pred_x0",
    )
    diffusion_model.set_metadata(full_dataset.stats)

    diffusion_model.to(device)

    # Initialize optimizer and scaler
    optimizer = Adam(diffusion_model.parameters(), lr=opt.learning_rate)
    scaler = GradScaler(enabled=True)

    # Initialize wandb
    if opt.use_wandb:
        print(opt, opt.wandb_pj_name, opt.entity, opt.exp_name, opt.save_dir)
        wandb.init(
            config=opt,
            project=opt.wandb_pj_name,
            # entity=opt.entity,
            name=opt.exp_name,
            dir=opt.save_dir,
        )

    # Track best model based on evaluation metrics
    best_eval_error = float("inf")
    best_model_state = None
    best_step = 0

    # Training loop
    print("Starting training loop...")
    print(f"Total windows available: {len(train_dataset.windows)}")
    print(f"Each epoch will see all {len(train_dataset.windows)} windows once")
    print(f"Using {opt.sampling_mode} sampling mode")

    # Initialize Rerun visualization if requested
    visualizer = None
    if opt.use_rerun:
        visualizer = RerunVisualizer(
            exp_name=opt.exp_name,
            save_dir=opt.save_dir,
            enable_visualization=True,
            mano_models_dir=getattr(opt, "mano_models_dir", "data/mano_models"),
            object_mesh_dir=getattr(opt, "object_mesh_dir", "data/object_meshes"),
            use_hand_articulations=getattr(opt, "use_hand_articulations", False),
            hand_articulations_path=getattr(
                opt, "hand_articulations_path", "data/hand_articulations.pkl"
            ),
        )
        # Setup for overfit training
        visualizer.setup_for_overfit_training(
            full_dataset, object_id=full_dataset.target_object_id
        )
    elif opt.use_pt3d:
        visualizer = Pt3dVisualizer(
            exp_name=opt.exp_name,
            save_dir=opt.save_dir,
        )
        # visualizer.setup_template(full_dataset.target_object_id)

    else:
        print("Rerun visualization disabled")

    # Handle backward compatibility and determine training parameters
    train_dataset_size = len(train_dataset)
    steps_per_epoch = train_dataset_size

    # Use epochs (preferred method)
    total_epochs = opt.num_epochs
    total_steps = total_epochs * steps_per_epoch

    print(f"Training setup for overfitting:")
    print(f"  Dataset size: {train_dataset_size} windows")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Total steps: {total_steps}")

    for epoch in tqdm(range(total_epochs)):
        # Create new permutation at the start of each epoch
        epoch_indices = np.random.permutation(train_dataset_size)

        for step_in_epoch in range(steps_per_epoch):
            # Calculate global step for logging and checkpoints
            global_step = epoch * steps_per_epoch + step_in_epoch

            optimizer.zero_grad()

            # Get sample for this step
            sample_idx = epoch_indices[step_in_epoch]
            sample = train_dataset[sample_idx]

            cond = sample["condition"].unsqueeze(0).to(device)  # [1, T, 2*D]
            object_motion = (
                sample["target"].unsqueeze(0).to(device)
            )  # [1, T, D] - normalized target
            seq_len = torch.tensor([cond.shape[1]]).to(device)  # [1] - sequence length

            ######### add occlusion mask for traj repr, with some schedules
            mask_prob = 0.5
            max_infill_ratio = 0.1
            prob = random.uniform(0, 1)
            batch_size, clip_len, _ = cond.shape
            if prob > 1 - mask_prob:
                traj_feat_dim = sample["traj_noisy_raw"].shape[-1]
                start = torch.FloatTensor(batch_size).uniform_(0, clip_len - 1).long()
                mask_len = (
                    clip_len
                    * torch.FloatTensor(batch_size).uniform_(0, 1)
                    * max_infill_ratio
                ).long()
                end = start + mask_len
                end[end > clip_len] = clip_len
                mask_traj = torch.ones(batch_size, clip_len).to(device)  # [bs, t]
                for bs in range(batch_size):
                    mask_traj[bs, start[bs] : end[bs]] = 0
                mask_traj_exp = mask_traj.unsqueeze(-1).repeat(
                    1, 1, traj_feat_dim
                )  # [bs, t, 4]
                cond[:, :, -traj_feat_dim:] = (
                    cond[:, :, -traj_feat_dim:] * mask_traj_exp
                )
            else:
                mask_traj = torch.ones(batch_size, clip_len).to(device)

            # Extract data from sample and move to device

            # Additional sample info for visualization
            is_moving = sample["is_motion"]
            mean_velocity = sample["mean_velocity"]

            # Generate padding mask
            actual_seq_len = seq_len + 1
            tmp_mask = torch.arange(opt.window_size + 1, device=device).expand(
                1, opt.window_size + 1
            ) < actual_seq_len[:, None].repeat(1, opt.window_size + 1)
            padding_mask = tmp_mask[:, None, :]

            with autocast(enabled=True):
                loss = diffusion_model(object_motion, cond, padding_mask=padding_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Evaluate model periodically
            if global_step % 1000 == 0:
                print(f"\n  Evaluating model at step {global_step}...")
                eval_results = evaluate_model(
                    diffusion_model,
                    val_dataset,
                    device,
                    num_eval_windows=5,
                    visualizer=visualizer,
                    global_step=global_step,
                )

                print(f"  Evaluation Results:")
                print(
                    f"    Mean position error: {eval_results['mean_position_error']:.4f}m"
                )
                print(
                    f"    Max position error: {eval_results['max_position_error']:.4f}m"
                )
                print(
                    f"    Mean rotation error: {eval_results['mean_rotation_error']:.4f}"
                )
                print(
                    f"    Mean combined error: {eval_results['mean_combined_error']:.4f}"
                )

                # Track best model based on evaluation metrics (use combined error)
                combined_error = eval_results["mean_combined_error"]
                if combined_error < best_eval_error:
                    best_eval_error = combined_error
                    best_step = global_step
                    # Save best model state

                if opt.use_wandb:
                    wandb.log(
                        {
                            "eval/mean_position_error": eval_results[
                                "mean_position_error"
                            ],
                            "eval/max_position_error": eval_results[
                                "max_position_error"
                            ],
                            "eval/mean_rotation_error": eval_results[
                                "mean_rotation_error"
                            ],
                            "eval/mean_combined_error": eval_results[
                                "mean_combined_error"
                            ],
                            "eval/best_combined_error": best_eval_error,
                            "eval/best_step": best_step,
                        },
                        step=global_step,
                    )

                # Save checkpoint
                checkpoint = {
                    "step": global_step,
                    "model": diffusion_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "dataset_info": {
                        "demo_id": train_dataset.demo_id,
                        "object_id": train_dataset.target_object_id,
                        "window_size": opt.window_size,
                        "full_length": train_dataset.full_length,
                        "num_windows": len(train_dataset.windows),
                    },
                }
                torch.save(checkpoint, os.path.join(wdir, f"model-{global_step}.pt"))

            if global_step % 100 == 0:
                epoch_progress = f"Epoch {epoch + 1}/{total_epochs}, Step {step_in_epoch + 1}/{steps_per_epoch}"
                print(
                    f"Step {global_step} ({epoch_progress}), Loss: {loss.item():.6f}, Best eval error: {best_eval_error:.4f}m (step {best_step})"
                )

                if opt.use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/best_eval_error": best_eval_error,
                            "train/best_step": best_step,
                            "train/current_epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            # Visualize training frame (with configurable frequency)
            if global_step % opt.visualization_frequency == 0 and visualizer:
                hand_poses_raw = (
                    sample["hand_raw"].unsqueeze(0).to(device)
                )  # [1, T, 2*D] - left + right hand
                object_motion_raw = (
                    sample["target_raw"].unsqueeze(0).to(device)
                )  # [1, T, D] - unnormalized ground truth
                left_hand_raw, right_hand_raw = torch.split(
                    hand_poses_raw, 21 * 3, dim=-1
                )

                object_pred_raw, _ = diffusion_model.sample_raw(
                    torch.randn_like(object_motion), cond, padding_mask=padding_mask
                )
                visualizer.log_training_step(
                    global_step,
                    left_hand_raw,
                    right_hand_raw,
                    object_motion_raw,
                    object_noisy=sample["traj_noisy_raw"].unsqueeze(0).to(device),
                    object_pred=object_pred_raw,
                    seq_len=seq_len,
                    is_moving=is_moving,
                    mean_velocity=mean_velocity,
                )

    print("Training completed!")
    print(f"Total epochs completed: {total_epochs}")
    print(f"Best model: step {best_step} with eval error {best_eval_error:.4f}m")

    # Test sampling - generate full trajectory using sliding windows
    print("\nTesting sampling on full trajectory...")
    sampled_motion_full = sample_full_trajectory_inpaint(
        diffusion_model,
        full_dataset,
        device,
        window_size=opt.window_size,
        overlap=opt.window_size // 2,
    )

    if sampled_motion_full is not None:
        # Save results - full trajectory
        save_path = os.path.join(opt.save_dir, "sampled_motion.npy")
        np.save(save_path, sampled_motion_full.unsqueeze(0).numpy())
        print(f"Saved full sampled motion to {save_path}")

        # Save input hand poses and ground truth - full trajectory
        hand_poses_full = torch.cat(
            [full_dataset.left_hand_full, full_dataset.right_hand_full], dim=-1
        )
        hand_poses_path = os.path.join(opt.save_dir, "input_hand_poses.npy")
        np.save(hand_poses_path, hand_poses_full.unsqueeze(0).numpy())
        print(f"Saved full input hand poses to {hand_poses_path}")

        gt_object_path = os.path.join(opt.save_dir, "ground_truth_object.npy")
        np.save(gt_object_path, full_dataset.object_motion_full.unsqueeze(0).numpy())
        print(f"Saved full ground truth object motion to {gt_object_path}")

        # Visualize final full trajectory
        if visualizer:
            visualizer.log_final_trajectory(full_dataset, sampled_motion_full)
    else:
        print(
            "Skipping full trajectory sampling - not available for this dataset configuration"
        )

    # Save training summary
    summary_path = os.path.join(opt.save_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n")
        f.write(f"Total epochs: {total_epochs}\n")
        f.write(f"Total steps: {total_steps}\n")
        f.write(f"Dataset size: {train_dataset_size} windows\n")
        f.write(f"Best step: {best_step}\n")
        f.write(f"Best eval error: {best_eval_error:.4f}m\n")
        f.write(f"Final loss: {loss.item():.6f}\n")
        f.write(
            f"Final mean position error: {final_results['mean_position_error']:.4f}m\n"
        )
        f.write(
            f"Final max position error: {final_results['max_position_error']:.4f}m\n"
        )
        f.write(
            f"Final mean rotation error: {final_results['mean_rotation_error']:.4f}\n"
        )
        f.write(
            f"Final mean combined error: {final_results['mean_combined_error']:.4f}\n"
        )
        f.write(f"Demo ID: {train_dataset.demo_id}\n")
        f.write(f"Object ID: {train_dataset.target_object_id}\n")
        f.write(f"Window size: {opt.window_size}\n")
        f.write(f"Use velocity: {opt.use_velocity}\n")
        f.write(f"Data dimension: {train_dataset.pose_dim}D\n")
        f.write(f"Rerun visualization: {'Enabled' if opt.use_rerun else 'Disabled'}\n")
        if opt.use_rerun:
            f.write(f"Enhanced scene visualization: Enabled\n")
            f.write(f"MANO hand models: Enabled\n")
            f.write(
                f"Hand articulations: {'Enabled' if opt.use_hand_articulations else 'Disabled'}\n"
            )
    print(f"Saved training summary to {summary_path}")

    # Rerun visualization summary
    if visualizer:
        print(visualizer.get_summary())

    if opt.use_wandb:
        wandb.log(
            {
                "final/total_epochs": total_epochs,
                "final/total_steps": total_steps,
                "final/dataset_size": train_dataset_size,
                "final/best_eval_error": best_eval_error,
                "final/best_step": best_step,
                "final/final_loss": loss.item(),
                "final/final_mean_error": final_results["mean_combined_error"],
                "final/final_max_error": final_results["max_position_error"],
                "final/final_mean_rotation_error": final_results["mean_rotation_error"],
                "final/final_mean_position_error": final_results["mean_position_error"],
            }
        )
        wandb.finish()


def parse_opt():
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="/move/u/yufeiy2/data/HOT3D/pred_pose/mini_P0001_624f2ba9.npz",
        help="Path to processed data pickle file",
    )
    parser.add_argument(
        "--demo_id",
        type=str,
        default=None,
        help="Specific demo ID to use (if None, use first available)",
    )
    parser.add_argument(
        "--target_object_id",
        type=str,
        default=None,
        help="Specific object ID to track (if None, use first available)",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="random",
        choices=["random", "sequential"],
        help="Window sampling mode: random (better performance) or sequential",
    )
    parser.add_argument(
        "--use_velocity",
        action="store_true",
        default=False,
        help="Use 12D data with velocity (default: False)",
    )

    # Model parameters
    parser.add_argument(
        "--window_size",
        type=int,
        default=120,
        help="Training window size and model max sequence length",
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Transformer model dimension"
    )
    parser.add_argument(
        "--n_dec_layers", type=int, default=6, help="Number of decoder layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--d_k", type=int, default=64, help="Key dimension")
    parser.add_argument("--d_v", type=int, default=64, help="Value dimension")

    # Training parameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (complete passes through dataset)",
    )
    parser.add_argument(
        "--guidance_weight", type=float, default=0.0, help="Guidance weight"
    )
    parser.add_argument(
        "--guidance_mode", action="store_true", help="Enable guidance mode"
    )

    # Logging parameters
    parser.add_argument(
        "--save_dir",
        type=str,
        default="runs/overfit_rerun",
        help="Directory to save results",
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="egorecon", help="Wandb project name"
    )
    parser.add_argument(
        "--entity", type=str, default="egorecon", help="Wandb entity name"
    )
    parser.add_argument(
        "--exp_name", type=str, default="overfit_with_rerun", help="Experiment name"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb for logging"
    )
    parser.add_argument(
        "--use_rerun", action="store_true", help="Use Rerun for real-time visualization"
    )
    parser.add_argument(
        "--use_hand_articulations",
        action="store_true",
        help="Use hand articulations in visualization (requires articulation data)",
    )

    # Visualization parameters
    parser.add_argument(
        "--mano_models_dir",
        type=str,
        default="data/mano_models",
        help="Directory containing MANO model files",
    )
    parser.add_argument(
        "--object_mesh_dir",
        type=str,
        default="data/object_meshes",
        help="Directory containing object mesh files",
    )
    parser.add_argument(
        "--hand_articulations_path",
        type=str,
        default="data/hand_articulations.pkl",
        help="Path to hand articulations pickle file",
    )
    parser.add_argument(
        "--generation_data_path",
        type=str,
        default="data/generation.pkl",
        help="Path to generation data pickle file",
    )
    parser.add_argument(
        "--visualization_frequency",
        type=int,
        default=1000,
        help="How often to save visualization files (steps)",
    )
    parser.add_argument(
        "--enhanced_scene_frames",
        type=int,
        default=500,
        help="Number of frames for enhanced scene visualization",
    )

    parser.add_argument("--noise_std_obj_rot", type=float, default=2)
    parser.add_argument("--noise_std_obj_trans", type=float, default=0.1)
    parser.add_argument("--noise_std_mano_global_rot", type=float, default=2)
    parser.add_argument("--noise_std_mano_body_rot", type=float, default=2)
    parser.add_argument("--noise_std_mano_trans", type=float, default=0.1)
    parser.add_argument("--noise_std_mano_betas", type=float, default=0.2)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_overfit(opt, device)
