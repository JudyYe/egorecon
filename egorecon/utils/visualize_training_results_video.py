#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import cv2

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

def create_training_video(results_dir, max_frames=None, fps=10, trail_length=20):
    """
    Create an animated video showing the training results over time.
    
    Args:
        results_dir: Directory containing training results
        max_frames: Maximum number of frames to visualize
        fps: Frames per second for the output video
        trail_length: Number of previous positions to show as trails
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
    sampled_motion = load_numpy_data(sampled_motion_path)  # [T, 9]
    ground_truth_object = load_numpy_data(ground_truth_path)  # [T, 9]
    input_hand_poses = load_numpy_data(hand_poses_path)  # [T, 18]
    
    # Split hand poses
    left_hand_poses = input_hand_poses[:, :9]  # [T, 9]
    right_hand_poses = input_hand_poses[:, 9:]  # [T, 9]
    
    original_seq_length = sampled_motion.shape[0]
    print(f"Original sequence length: {original_seq_length} frames")
    
    # Handle very long sequences - automatically downsample if needed
    if max_frames is None:
        if original_seq_length > 1000:
            print(f"⚠️  Sequence is very long ({original_seq_length} frames). Auto-limiting to 500 frames for performance.")
            print("   Use --max_frames to override this limit if needed.")
            max_frames = 500
        else:
            max_frames = original_seq_length
    elif max_frames == -1:
        # -1 means use all frames, no auto-limiting
        print(f"Using all {original_seq_length} frames as requested (--max_frames -1)")
        max_frames = original_seq_length
    
    # Downsample if needed
    if original_seq_length > max_frames:
        print(f"Downsampling from {original_seq_length} to {max_frames} frames...")
        indices = np.linspace(0, original_seq_length - 1, max_frames, dtype=int)
        sampled_motion = sampled_motion[indices]
        ground_truth_object = ground_truth_object[indices]
        left_hand_poses = left_hand_poses[indices]
        right_hand_poses = right_hand_poses[indices]
        seq_length = max_frames
    else:
        seq_length = original_seq_length
    
    print(f"Processing {seq_length} frames (estimated video duration: {seq_length/fps:.1f}s)...")
    
    # Extract trajectories
    sampled_translations = []
    ground_truth_translations = []
    left_hand_translations = []
    right_hand_translations = []
    
    print("Converting poses to translations...")
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
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots: 3D view and 2D plots
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_error = fig.add_subplot(2, 2, 2)
    ax_xy = fig.add_subplot(2, 2, 3)
    ax_z = fig.add_subplot(2, 2, 4)
    
    # Set up 3D plot
    all_positions = np.concatenate([sampled_translations, ground_truth_translations, 
                                   left_hand_translations, right_hand_translations])
    
    # Set axis limits with some padding
    margin = 0.1
    x_min, x_max = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
    y_min, y_max = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
    z_min, z_max = all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin
    
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(z_min, z_max)
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectories (Real-time)')
    
    # Initialize empty plots
    left_trail, = ax_3d.plot([], [], [], 'r-', alpha=0.6, linewidth=1, label='Left Hand Trail')
    right_trail, = ax_3d.plot([], [], [], 'b-', alpha=0.6, linewidth=1, label='Right Hand Trail')
    gt_trail, = ax_3d.plot([], [], [], 'g-', alpha=0.6, linewidth=1, label='GT Object Trail')
    sampled_trail, = ax_3d.plot([], [], [], 'orange', linestyle='--', alpha=0.6, linewidth=1, label='Sampled Trail')
    
    # Current position markers
    left_point, = ax_3d.plot([], [], [], 'ro', markersize=8, label='Left Hand')
    right_point, = ax_3d.plot([], [], [], 'bo', markersize=8, label='Right Hand')
    gt_point, = ax_3d.plot([], [], [], 'go', markersize=8, label='GT Object')
    sampled_point, = ax_3d.plot([], [], [], 'o', color='orange', markersize=8, label='Sampled Object')
    
    ax_3d.legend(loc='upper right')
    
    # Set up 2D plots
    ax_error.set_xlim(0, seq_length)
    ax_error.set_ylim(0, max(position_errors) * 1.1)
    ax_error.set_xlabel('Frame')
    ax_error.set_ylabel('Position Error (m)')
    ax_error.set_title('Position Error Over Time')
    ax_error.grid(True)
    error_line, = ax_error.plot([], [], 'purple', linewidth=2)
    error_point, = ax_error.plot([], [], 'ro', markersize=6)
    
    # X-Y view
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top View (X-Y)')
    ax_xy.grid(True)
    ax_xy.axis('equal')
    
    xy_left_trail, = ax_xy.plot([], [], 'r-', alpha=0.6, linewidth=1)
    xy_right_trail, = ax_xy.plot([], [], 'b-', alpha=0.6, linewidth=1)
    xy_gt_trail, = ax_xy.plot([], [], 'g-', alpha=0.6, linewidth=1)
    xy_sampled_trail, = ax_xy.plot([], [], 'orange', linestyle='--', alpha=0.6, linewidth=1)
    
    xy_left_point, = ax_xy.plot([], [], 'ro', markersize=6)
    xy_right_point, = ax_xy.plot([], [], 'bo', markersize=6)
    xy_gt_point, = ax_xy.plot([], [], 'go', markersize=6)
    xy_sampled_point, = ax_xy.plot([], [], 'o', color='orange', markersize=6)
    
    # Z over time
    ax_z.set_xlim(0, seq_length)
    ax_z.set_ylim(z_min, z_max)
    ax_z.set_xlabel('Frame')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.set_title('Height (Z) Over Time')
    ax_z.grid(True)
    
    z_gt_line, = ax_z.plot([], [], 'g-', linewidth=2, label='GT Object')
    z_sampled_line, = ax_z.plot([], [], 'orange', linestyle='--', linewidth=2, label='Sampled Object')
    z_gt_point, = ax_z.plot([], [], 'go', markersize=6)
    z_sampled_point, = ax_z.plot([], [], 'o', color='orange', markersize=6)
    ax_z.legend()
    
    # Add text for current frame and error
    frame_text = fig.suptitle('', fontsize=14)
    
    def animate(frame):
        # Calculate trail indices
        start_idx = max(0, frame - trail_length)
        end_idx = frame + 1
        
        # Update 3D trails
        if end_idx > start_idx:
            left_trail.set_data_3d(left_hand_translations[start_idx:end_idx, 0],
                                  left_hand_translations[start_idx:end_idx, 1],
                                  left_hand_translations[start_idx:end_idx, 2])
            right_trail.set_data_3d(right_hand_translations[start_idx:end_idx, 0],
                                   right_hand_translations[start_idx:end_idx, 1],
                                   right_hand_translations[start_idx:end_idx, 2])
            gt_trail.set_data_3d(ground_truth_translations[start_idx:end_idx, 0],
                                ground_truth_translations[start_idx:end_idx, 1],
                                ground_truth_translations[start_idx:end_idx, 2])
            sampled_trail.set_data_3d(sampled_translations[start_idx:end_idx, 0],
                                     sampled_translations[start_idx:end_idx, 1],
                                     sampled_translations[start_idx:end_idx, 2])
        
        # Update current positions
        left_point.set_data_3d([left_hand_translations[frame, 0]], 
                              [left_hand_translations[frame, 1]], 
                              [left_hand_translations[frame, 2]])
        right_point.set_data_3d([right_hand_translations[frame, 0]], 
                               [right_hand_translations[frame, 1]], 
                               [right_hand_translations[frame, 2]])
        gt_point.set_data_3d([ground_truth_translations[frame, 0]], 
                            [ground_truth_translations[frame, 1]], 
                            [ground_truth_translations[frame, 2]])
        sampled_point.set_data_3d([sampled_translations[frame, 0]], 
                                 [sampled_translations[frame, 1]], 
                                 [sampled_translations[frame, 2]])
        
        # Update error plot
        error_line.set_data(range(frame + 1), position_errors[:frame + 1])
        error_point.set_data([frame], [position_errors[frame]])
        
        # Update X-Y view
        if end_idx > start_idx:
            xy_left_trail.set_data(left_hand_translations[start_idx:end_idx, 0],
                                  left_hand_translations[start_idx:end_idx, 1])
            xy_right_trail.set_data(right_hand_translations[start_idx:end_idx, 0],
                                   right_hand_translations[start_idx:end_idx, 1])
            xy_gt_trail.set_data(ground_truth_translations[start_idx:end_idx, 0],
                                ground_truth_translations[start_idx:end_idx, 1])
            xy_sampled_trail.set_data(sampled_translations[start_idx:end_idx, 0],
                                     sampled_translations[start_idx:end_idx, 1])
        
        xy_left_point.set_data([left_hand_translations[frame, 0]], [left_hand_translations[frame, 1]])
        xy_right_point.set_data([right_hand_translations[frame, 0]], [right_hand_translations[frame, 1]])
        xy_gt_point.set_data([ground_truth_translations[frame, 0]], [ground_truth_translations[frame, 1]])
        xy_sampled_point.set_data([sampled_translations[frame, 0]], [sampled_translations[frame, 1]])
        
        # Update Z plot
        z_gt_line.set_data(range(frame + 1), ground_truth_translations[:frame + 1, 2])
        z_sampled_line.set_data(range(frame + 1), sampled_translations[:frame + 1, 2])
        z_gt_point.set_data([frame], [ground_truth_translations[frame, 2]])
        z_sampled_point.set_data([frame], [sampled_translations[frame, 2]])
        
        # Update title with current info
        current_error = position_errors[frame]
        frame_text.set_text(f'Frame {frame}/{seq_length-1} | Current Error: {current_error:.4f}m | Mean Error: {mean_error:.4f}m')
        
        return (left_trail, right_trail, gt_trail, sampled_trail,
                left_point, right_point, gt_point, sampled_point,
                error_line, error_point,
                xy_left_trail, xy_right_trail, xy_gt_trail, xy_sampled_trail,
                xy_left_point, xy_right_point, xy_gt_point, xy_sampled_point,
                z_gt_line, z_sampled_line, z_gt_point, z_sampled_point,
                frame_text)
    
    # Create animation with progress callback
    print(f"Creating animation with {seq_length} frames at {fps} FPS...")
    print("This may take a few minutes for long sequences...")
    
    # Use blit=True for better performance when possible
    use_blit = seq_length < 200  # Only use blit for shorter sequences
    
    anim = animation.FuncAnimation(fig, animate, frames=seq_length, 
                                 interval=1000//fps, blit=use_blit, repeat=True)
    
    # Save animation with progress updates
    output_path = results_path / "training_animation.mp4"
    print(f"Saving video to: {output_path}")
    print("⏳ Rendering video... This may take several minutes for long sequences.")
    
    # Use different writers based on availability
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Training Visualizer'), bitrate=1800)
        
        # Save with progress updates
        print("🎬 Starting video encoding...")
        anim.save(output_path, writer=writer, progress_callback=lambda i, n: print(f"🎬 Encoding frame {i+1}/{n}", end='\r') if i % 50 == 0 else None)
        print(f"\n✅ Video saved successfully to: {output_path}")
    except Exception as save_error:
        print(f"❌ Error with ffmpeg: {save_error}")
        try:
            # Fallback to pillow
            print("Trying alternative format (GIF)...")
            anim.save(output_path.with_suffix('.gif'), writer='pillow', fps=fps)
            print(f"✅ GIF saved successfully to: {output_path.with_suffix('.gif')}")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
            print("Showing interactive plot instead...")
            plt.show()
            return
    
    print("\n📊 Video Summary:")
    print(f"  📏 Original sequence length: {original_seq_length} frames")
    if original_seq_length != seq_length:
        print(f"  📏 Downsampled to: {seq_length} frames")
    print(f"  🎯 Mean position error: {mean_error:.4f} meters")
    print(f"  📈 Max position error: {max_error:.4f} meters")
    print(f"  🎬 Video duration: {seq_length/fps:.1f} seconds")
    print("\n🎨 Color Legend:")
    print("  🔴 Red: Left hand trajectory")
    print("  🔵 Blue: Right hand trajectory")
    print("  🟢 Green: Ground truth object trajectory")
    print("  🟠 Orange dashed: Sampled object trajectory")

def main():
    parser = argparse.ArgumentParser(
        description="Create animated video of training results.",
        epilog="Note: Very long sequences (>1000 frames) will be automatically downsampled to 500 frames for performance unless --max_frames is specified. Use --max_frames -1 to force using all frames."
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
        help="Maximum number of frames to visualize. For sequences >1000 frames, defaults to 500 for performance. Set to -1 to use all frames (no auto-limiting)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output video (default: 10)"
    )
    parser.add_argument(
        "--trail_length",
        type=int,
        default=20,
        help="Number of previous positions to show as trails (default: 20)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.results_dir).exists():
        print(f"Error: Results directory {args.results_dir} not found!")
        return
    
    create_training_video(args.results_dir, 
                         max_frames=args.max_frames, 
                         fps=args.fps, 
                         trail_length=args.trail_length)

if __name__ == "__main__":
    main() 