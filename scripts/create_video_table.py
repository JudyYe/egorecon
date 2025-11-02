import os
import argparse
import re
from glob import glob
from jutils import web_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Create HTML table displaying video results')
    parser.add_argument('--dir', type=str, default="outputs/oracle_cond/bps2_False/eval_hoi_contact_ddim_long_lambda1/log/", help='Directory containing video files')
    parser.add_argument('--width', type=int, default=400, help='Video width in pixels')
    parser.add_argument('--height', type=int, default=None, help='Video height in pixels (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file path (default: {dir}/index.html)')
    parser.add_argument('--inplace', action='store_true', help='Use videos in place without copying')
    return parser.parse_args()


def extract_index_from_filename(filename):
    """Extract the index from filenames like test_guided_0000_gt_0000000.mp4"""
    # Match pattern: test_guided_{index}_gt or test_guided_{index}_pred or test_post_{index} or test_sample_0_{index}
    patterns = [
        r'test_guided_(\d+)_gt',
        r'test_guided_(\d+)_pred',
        r'test_post_(\d+)_',
        r'test_sample_0_(\d+)_',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return None


def find_videos_by_pattern(dir_path, index):
    """Find videos for a specific index matching the required patterns"""
    videos = {
        'sample': None,  # test_sample_0_{index}_0000000.mp4
        'pred': None,    # test_guided_{index}_pred_0000000.mp4
        'post': None,    # test_post_{index}_0000000.mp4
        'gt': None,      # test_guided_{index}_gt_0000000.mp4
    }
    
    # Pattern for sample
    sample_pattern = os.path.join(dir_path, f'test_sample_0_{index:04d}_*.mp4')
    sample_files = glob(sample_pattern)
    if sample_files:
        videos['sample'] = sorted(sample_files)[0]
    
    # Pattern for pred
    pred_pattern = os.path.join(dir_path, f'test_guided_{index:04d}_pred_*.mp4')
    pred_files = glob(pred_pattern)
    if pred_files:
        videos['pred'] = sorted(pred_files)[0]
    
    # Pattern for post
    post_pattern = os.path.join(dir_path, f'test_post_{index:04d}_*.mp4')
    post_files = glob(post_pattern)
    if post_files:
        videos['post'] = sorted(post_files)[0]
    
    # Pattern for gt
    gt_pattern = os.path.join(dir_path, f'test_guided_{index:04d}_gt_*.mp4')
    gt_files = glob(gt_pattern)
    if gt_files:
        videos['gt'] = sorted(gt_files)[0]
    
    return videos


def collect_all_indices(dir_path):
    """Collect all unique indices from video files in the directory"""
    all_files = glob(os.path.join(dir_path, 'test_*.mp4'))
    indices = set()
    
    for filename in all_files:
        basename = os.path.basename(filename)
        idx = extract_index_from_filename(basename)
        if idx is not None:
            indices.add(idx)
    
    return sorted(indices)


def create_video_table(args):
    """Main function to create video table HTML"""
    dir_path = args.dir
    
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")
    
    # Collect all indices
    indices = collect_all_indices(dir_path)
    
    if not indices:
        print(f"No matching video files found in {dir_path}")
        return
    
    print(f"Found {len(indices)} indices: {indices}")
    
    # Build cell_list: each row is [sample, pred, post, gt]
    cell_list = []
    for idx in indices:
        videos = find_videos_by_pattern(dir_path, idx)
        row = [
            videos['sample'],  # Column 0: test_sample_0_
            videos['pred'],    # Column 1: test_guided_{index}_pred_
            videos['post'],    # Column 2: test_post_{index}_
            videos['gt'],      # Column 3: test_guided_{index}_gt_
        ]
        cell_list.append(row)
        print(f"Index {idx}: sample={videos['sample'] is not None}, "
              f"pred={videos['pred'] is not None}, "
              f"post={videos['post'] is not None}, "
              f"gt={videos['gt'] is not None}")
    
    # Determine output path
    if args.output:
        html_root = args.output
    else:
        html_root = os.path.join(dir_path, 'vis.html')
    
    # Create HTML table
    web_utils.run(
        html_root=html_root,
        cell_list=cell_list,
        width=args.width,
        height=args.height,
        inplace=args.inplace,
    )
    print(f"\nHTML table created at: {html_root}")


if __name__ == '__main__':
    args = parse_args()
    create_video_table(args)
