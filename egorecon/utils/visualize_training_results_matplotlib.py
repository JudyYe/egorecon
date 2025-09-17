#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm

from .quaternion import quat_from_6v

def load_numpy_data(file_path):
    """Load numpy data from file."""
    return np.load(file_path)

def pose_9d_to_translation_and_rotation(pose_9d):
    """
    Convert 9D pose to translation and rotation matrix.
    
    Args:
        pose_9d: numpy array of shape [9] -> [tx, ty, tz, r1, r2, r3, r4, r5, r6]
    
    Returns:
        translation, rotation_matrix
    """
    translation = pose_9d[:3]  # [3]
    rotation_6d = pose_9d[3:]  # [6]
    
    # Convert 6D rotation back to quaternion
    rotation_6d_tensor = torch.tensor(rotation_6d, dtype=torch.float32).unsqueeze(0)  # [1, 6]
    quaternion_tensor = quat_from_6v(rotation_6d_tensor)  # [1, 4]
    quaternion = quaternion_tensor.squeeze(0).numpy()  # [4]
    
    # Convert quaternion to rotation matrix
    x, y, z, w = quaternion
    rotation_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    
    return translation, rotation_matrix

def visualize_training_results_matplotlib(results_dir, max_frames=None, save_plots=True):
    """
    Visualize training results using matplotlib.
    """
    results_path = Path(results_dir)
    
    # Load the saved results
    sampled_motion_path = results_path / "sampled_motion.npy"
    ground_truth_path = results_path / "ground_truth_object.npy"
    hand_poses_path = results_path / "input_hand_poses.npy"
    
    if not all(p.exists() for p in [sampled_motion_path, ground_truth_path, hand_poses_path]):
        missing = [p.name for p in [sampled_motion_path, ground_truth_path, hand_poses_path] if not p.exists()]
        print(f"Missing files: {missing}")
        return
    
    print("Loading training results...")
    sampled_motion = load_numpy_data(sampled_motion_path).squeeze(0)  # [T, 9]
    ground_truth_object = load_numpy_data(ground_truth_path).squeeze(0)  # [T, 9]
    input_hand_poses = load_numpy_data(hand_poses_path).squeeze(0)  # [T, 18]
    
    # Split hand poses
    left_hand_poses = input_hand_poses[:, :9]  # [T, 9]
    right_hand_poses = input_hand_poses[:, 9:]  # [T, 9]
    
    seq_length = sampled_motion.shape[0]
    if max_frames is not None and seq_length > max_frames:
        seq_length = max_frames
        sampled_motion = sampled_motion[:max_frames]
        ground_truth_object = ground_truth_object[:max_frames]
        left_hand_poses = left_hand_poses[:max_frames]
        right_hand_poses = right_hand_poses[:max_frames]
    
    print(f"Processing {seq_length} frames...")
    
    # Extract trajectories
    sampled_translations = []
    ground_truth_translations = []
    left_hand_translations = []
    right_hand_translations = []
    
    for i in tqdm(range(seq_length)):
        sampled_trans, _ = pose_9d_to_translation_and_rotation(sampled_motion[i])
        gt_trans, _ = pose_9d_to_translation_and_rotation(ground_truth_object[i])
        left_trans, _ = pose_9d_to_translation_and_rotation(left_hand_poses[i])
        right_trans, _ = pose_9d_to_translation_and_rotation(right_hand_poses[i])
        
        sampled_translations.append(sampled_trans)
        ground_truth_translations.append(gt_trans)
        left_hand_translations.append(left_trans)
        right_hand_translations.append(right_trans)
    
    # Convert to numpy arrays
    sampled_translations = np.array(sampled_translations)
    ground_truth_translations = np.array(ground_truth_translations)
    left_hand_translations = np.array(left_hand_translations)
    right_hand_translations = np.array(right_hand_translations)
    
    # Calculate position errors
    position_errors = np.linalg.norm(ground_truth_translations - sampled_translations, axis=1)
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    
    print(f"Mean position error: {mean_error:.4f} meters")
    print(f"Max position error: {max_error:.4f} meters")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(left_hand_translations[:, 0], left_hand_translations[:, 1], left_hand_translations[:, 2], 
             'r-', label='Left Hand', linewidth=2)
    ax1.plot(right_hand_translations[:, 0], right_hand_translations[:, 1], right_hand_translations[:, 2], 
             'b-', label='Right Hand', linewidth=2)
    ax1.plot(ground_truth_translations[:, 0], ground_truth_translations[:, 1], ground_truth_translations[:, 2], 
             'g-', label='Ground Truth Object', linewidth=2)
    ax1.plot(sampled_translations[:, 0], sampled_translations[:, 1], sampled_translations[:, 2], 
             'orange', linestyle='--', label='Sampled Object', linewidth=2)
    ax1.set_title('3D Trajectories')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Position Error Over Time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(position_errors, 'purple', linewidth=2)
    ax2.set_title(f'Position Error Over Time\nMean: {mean_error:.4f}m, Max: {max_error:.4f}m')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Error (m)')
    ax2.grid(True)
    
    # 3. X-Y trajectory view
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(left_hand_translations[:, 0], left_hand_translations[:, 1], 'r-', label='Left Hand', linewidth=2)
    ax3.plot(right_hand_translations[:, 0], right_hand_translations[:, 1], 'b-', label='Right Hand', linewidth=2)
    ax3.plot(ground_truth_translations[:, 0], ground_truth_translations[:, 1], 'g-', label='GT Object', linewidth=2)
    ax3.plot(sampled_translations[:, 0], sampled_translations[:, 1], 'orange', linestyle='--', label='Sampled Object', linewidth=2)
    ax3.set_title('Top View (X-Y)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # 4. Position components over time
    ax4 = fig.add_subplot(2, 3, 4)
    frames = np.arange(seq_length)
    ax4.plot(frames, ground_truth_translations[:, 0], 'g-', label='GT X', alpha=0.7)
    ax4.plot(frames, sampled_translations[:, 0], 'orange', linestyle='--', label='Sampled X', alpha=0.7)
    ax4.plot(frames, ground_truth_translations[:, 1], 'g-', label='GT Y', alpha=0.7)
    ax4.plot(frames, sampled_translations[:, 1], 'orange', linestyle='--', label='Sampled Y', alpha=0.7)
    ax4.plot(frames, ground_truth_translations[:, 2], 'g-', label='GT Z', alpha=0.7)
    ax4.plot(frames, sampled_translations[:, 2], 'orange', linestyle='--', label='Sampled Z', alpha=0.7)
    ax4.set_title('Position Components Over Time')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Position (m)')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Distance from origin over time
    ax5 = fig.add_subplot(2, 3, 5)
    gt_distances = np.linalg.norm(ground_truth_translations, axis=1)
    sampled_distances = np.linalg.norm(sampled_translations, axis=1)
    left_distances = np.linalg.norm(left_hand_translations, axis=1)
    right_distances = np.linalg.norm(right_hand_translations, axis=1)
    
    ax5.plot(frames, gt_distances, 'g-', label='GT Object', linewidth=2)
    ax5.plot(frames, sampled_distances, 'orange', linestyle='--', label='Sampled Object', linewidth=2)
    ax5.plot(frames, left_distances, 'r-', label='Left Hand', alpha=0.7)
    ax5.plot(frames, right_distances, 'b-', label='Right Hand', alpha=0.7)
    ax5.set_title('Distance from Origin')
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Distance (m)')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Error statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(position_errors, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax6.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}m')
    ax6.axvline(max_error, color='orange', linestyle='--', linewidth=2, label=f'Max: {max_error:.4f}m')
    ax6.set_title('Position Error Distribution')
    ax6.set_xlabel('Error (m)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        output_path = results_path / "training_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    print("\nTrajectory Legend:")
    print("  🔴 Red line: Left hand trajectory")
    print("  🔵 Blue line: Right hand trajectory")
    print("  🟢 Green line: Ground truth object trajectory")
    print("  🌸 Orange dashed line: Sampled object trajectory")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize training results using matplotlib."
    )
    parser.add_argument(
        "--results_dir", "-r",
        type=str,
        default="runs/overfit",
        help="Directory containing training results"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to visualize (if None, use all)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save plots to file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.results_dir).exists():
        print(f"Error: Results directory {args.results_dir} not found!")
        return
    
    visualize_training_results_matplotlib(args.results_dir, max_frames=args.max_frames, save_plots=args.save)

if __name__ == "__main__":
    main() 