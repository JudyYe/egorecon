import os
import argparse
import re
from glob import glob
from jutils import web_utils

# Shared pattern-title configuration
# Format: (regex_pattern, glob_pattern_template, title)
# glob_pattern_template uses {index:04d} for formatting
PATTERN_TITLE_CONFIG = [
    (r'(\d+)_input', '{index:04d}_input*.mp4', 'input'),
    # (r'(\d+)_0_sample_', '{index:04d}_0_sample_*.mp4', 'sample'),
    # (r'(\d+)_guided_', '{index:04d}_guided_*.mp4', 'guided'),
    (r'(\d+)_post_', '{index:04d}_post_*.mp4', 'post'),
    (r'(\d+)_gt_', '{index:04d}_gt_*.mp4', 'gt'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Create HTML table displaying video results')
    parser.add_argument('--dir', type=str, default="outputs/oracle_cond/bps2_False/eval_hoi_contact_ddim_long_lambda1/log/", help='Directory containing video files')
    parser.add_argument('--width', type=int, default=None, help='Video width in pixels')
    parser.add_argument('--height', type=int, default=320, help='Video height in pixels (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file path (default: {dir}/index.html)')
    parser.add_argument('--inplace', action='store_true', help='Use videos in place without copying')
    return parser.parse_args()


def extract_index_from_filename(filename):
    """Extract the index and title from filenames matching predefined patterns"""
    for regex_pattern, glob_pattern, title in PATTERN_TITLE_CONFIG:
        match = re.search(regex_pattern, filename)
        if match:
            return int(match.group(1)), title
    return None, None


def find_videos_by_pattern(dir_path, index):
    """Find videos for a specific index matching the required patterns"""
    videos = {}
    
    for regex_pattern, glob_pattern_template, title in PATTERN_TITLE_CONFIG:
        # Format the glob pattern with the index
        glob_pattern = glob_pattern_template.format(index=index)
        full_pattern = os.path.join(dir_path, glob_pattern)
        matched_files = glob(full_pattern)
        
        if matched_files:
            videos[title] = sorted(matched_files)[0]
        else:
            videos[title] = None
    
    return videos


def collect_all_indices(dir_path):
    """Collect all unique indices from video files in the directory"""
    all_files = glob(os.path.join(dir_path, '*.mp4'))
    indices = set()
    
    for filename in all_files:
        basename = os.path.basename(filename)
        idx, title = extract_index_from_filename(basename)
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
    
    # Build cell_list: each row contains videos in order defined by PATTERN_TITLE_CONFIG
    titles = [title for _, _, title in PATTERN_TITLE_CONFIG]
    cell_list = []
    for idx in indices:
        videos = find_videos_by_pattern(dir_path, idx)
        row = [videos[title] for title in titles]
        cell_list.append(row)
        
        # Print status for each video type
        status_parts = [f"{title}={videos[title] is not None}" for title in titles]
        print(f"Index {idx}: {', '.join(status_parts)}")
    
    # Determine output path
    if args.output:
        html_root = args.output
    else:
        html_root = os.path.join(dir_path, 'vis.html')
    
    # Create HTML table
    web_utils.run(
        html_root=html_root,
        cell_list=cell_list,
        # width=args.width,
        height=args.height,
        inplace=True,
    )
    print(f"\nHTML table created at: {html_root}")


if __name__ == '__main__':
    args = parse_args()
    create_video_table(args)
