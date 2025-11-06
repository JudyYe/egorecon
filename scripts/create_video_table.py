import os
import argparse
import re
from glob import glob
from jutils import web_utils

# Shared pattern-title configuration
# Format: (regex_pattern, glob_pattern_template, title)
# glob_pattern_template uses {seq_objid} for formatting (e.g., 003078_000017)

OURS_PATTERN_TITLE_CONFIG = [
    (r'(\d{6}_\d{6})_input', '{seq_objid}_input*.mp4', 'input'),
    # (r'(\d{6}_\d{6})_0_sample_', '{seq_objid}_0_sample_*.mp4', 'sample'),
    # (r'(\d{6}_\d{6})_guided_', '{seq_objid}_guided_*.mp4', 'guided'),
    (r'(\d{6}_\d{6})_post_', '{seq_objid}_post_*.mp4', 'post'),
    (r'(\d{6}_\d{6})_gt_', '{seq_objid}_gt_*.mp4', 'gt'),
]

# BASELIN PATTERN
# outputs/baselines/fp_hawor/eval_test50obj_fp_full_more/log/
# 001874_000007_fp_0000000.mp4
# 001874_000007_gt_0000000.mp4
BASELINE_PATTERN_TITLE_CONFIG = [
    (r'(\d{6}_\d{6})_fp_', '{seq_objid}_fp_*.mp4', 'fp'),
    (r'(\d{6}_\d{6})_gt_', '{seq_objid}_gt_*.mp4', 'gt'),
]

PATTERN_TITLE_CONFIG = BASELINE_PATTERN_TITLE_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description='Create HTML table displaying video results')
    parser.add_argument('--dir', type=str, default="outputs/oracle_cond/bps2_False/eval_hoi_contact_ddim_long_lambda1/log/", help='Directory containing video files')
    parser.add_argument('--width', type=int, default=None, help='Video width in pixels')
    parser.add_argument('--height', type=int, default=320, help='Video height in pixels (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file path (default: {dir}/index.html)')
    parser.add_argument('--inplace', action='store_true', help='Use videos in place without copying')
    return parser.parse_args()


def extract_seq_objid_from_filename(filename):
    """Extract the seq_objid and title from filenames matching predefined patterns"""
    for regex_pattern, glob_pattern, title in PATTERN_TITLE_CONFIG:
        match = re.search(regex_pattern, filename)
        if match:
            return match.group(1), title  # Return seq_objid as string (e.g., "003078_000017")
    return None, None


def find_videos_by_pattern(dir_path, seq_objid):
    """Find videos for a specific seq_objid matching the required patterns"""
    videos = {}
    
    for regex_pattern, glob_pattern_template, title in PATTERN_TITLE_CONFIG:
        # Format the glob pattern with the seq_objid
        glob_pattern = glob_pattern_template.format(seq_objid=seq_objid)
        full_pattern = os.path.join(dir_path, glob_pattern)
        matched_files = glob(full_pattern)
        
        if matched_files:
            videos[title] = sorted(matched_files)[0]
        else:
            videos[title] = None
    
    return videos


def collect_all_seq_objids(dir_path):
    """Collect all unique seq_objids from video files in the directory"""
    all_files = glob(os.path.join(dir_path, '*.mp4'))
    seq_objids = set()
    
    for filename in all_files:
        basename = os.path.basename(filename)
        seq_objid, title = extract_seq_objid_from_filename(basename)
        if seq_objid is not None:
            seq_objids.add(seq_objid)
    
    return sorted(seq_objids)


def create_video_table(args):
    """Main function to create video table HTML"""
    dir_path = args.dir
    
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")
    
    # Collect all seq_objids
    seq_objids = collect_all_seq_objids(dir_path)
    
    if not seq_objids:
        print(f"No matching video files found in {dir_path}")
        return
    
    print(f"Found {len(seq_objids)} seq_objids: {seq_objids[:5]}..." if len(seq_objids) > 5 else f"Found {len(seq_objids)} seq_objids: {seq_objids}")
    
    # Build cell_list: each row contains videos in order defined by PATTERN_TITLE_CONFIG
    titles = [title for _, _, title in PATTERN_TITLE_CONFIG]
    cell_list = []
    for seq_objid in seq_objids:
        videos = find_videos_by_pattern(dir_path, seq_objid)
        row = [videos[title] for title in titles]
        cell_list.append(row)
        
        # Print status for each video type
        status_parts = [f"{title}={videos[title] is not None}" for title in titles]
        print(f"seq_objid {seq_objid}: {', '.join(status_parts)}")
    
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
