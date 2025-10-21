#!/usr/bin/env python3
"""
Create an HTML page to display test videos with ground truth in first column and predictions in second column.
"""

import os
import glob
from pathlib import Path

def find_video_files(base_dir="/move/u/yufeiy2/egorecon"):
    """Find all test_post_*_gt_0000000.mp4 and test_post_*_pred_0000000.mp4 files."""
    
    # Search for GT and pred video files
    gt_pattern = os.path.join(base_dir, "**", "test_post_*_gt_0000000.mp4")
    pred_pattern = os.path.join(base_dir, "**", "test_post_*_pred_0000000.mp4")
    
    gt_files = glob.glob(gt_pattern, recursive=True)
    pred_files = glob.glob(pred_pattern, recursive=True)
    
    print(f"Found {len(gt_files)} GT files and {len(pred_files)} pred files")
    
    # Create pairs by matching the test_post_* part
    pairs = []
    for gt_file in gt_files:
        # Extract the test_post_* part from the GT file
        gt_name = os.path.basename(gt_file)
        test_id = gt_name.replace("_gt_0000000.mp4", "")
        
        # Find corresponding pred file
        pred_file = None
        for pred in pred_files:
            if test_id in os.path.basename(pred):
                pred_file = pred
                break
        
        if pred_file:
            pairs.append({
                'test_id': test_id,
                'gt_file': gt_file,
                'pred_file': pred_file,
                'gt_relative': os.path.relpath(gt_file, base_dir),
                'pred_relative': os.path.relpath(pred_file, base_dir)
            })
    
    return pairs

def create_html_page(video_pairs, output_file="video_comparison.html"):
    """Create an HTML page displaying the video pairs."""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Video Comparison - GT vs Predictions</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .video-pair {{
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .video-container {{
            text-align: center;
            margin-bottom: 15px;
        }}
        .video-container h3 {{
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .video-container video {{
            width: 100%;
            max-width: 500px;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .test-id {{
            text-align: center;
            font-weight: bold;
            font-size: 18px;
            color: #34495e;
            margin-bottom: 15px;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }}
        .gt-label {{
            color: #27ae60;
            font-weight: bold;
        }}
        .pred-label {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #3498db;
            color: white;
            border-radius: 8px;
        }}
        .file-path {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Video Comparison</h1>
        <p>Ground Truth vs Predictions</p>
    </div>
    
    <div class="stats">
        <h2>Total Video Pairs: {len(video_pairs)}</h2>
    </div>
    
    <div class="video-grid">
"""
    
    for i, pair in enumerate(video_pairs):
        html_content += f"""
        <div class="video-pair">
            <div class="test-id">Test: {pair['test_id']}</div>
            
            <div class="video-container">
                <h3 class="gt-label">Ground Truth</h3>
                <video controls>
                    <source src="{pair['gt_relative']}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="file-path">{pair['gt_relative']}</div>
            </div>
            
            <div class="video-container">
                <h3 class="pred-label">Prediction</h3>
                <video controls>
                    <source src="{pair['pred_relative']}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="file-path">{pair['pred_relative']}</div>
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <script>
        // Auto-play videos when they come into view
        const videos = document.querySelectorAll('video');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.play();
                } else {
                    entry.target.pause();
                }
            });
        });
        
        videos.forEach(video => {
            observer.observe(video);
        });
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML page created: {output_file}")

def main():
    """Main function to create the video comparison HTML page."""
    
    base_dir = "/move/u/yufeiy2/egorecon"
    output_file = "/move/u/yufeiy2/egorecon/video_comparison.html"
    
    print("Searching for video files...")
    video_pairs = find_video_files(base_dir)
    
    if not video_pairs:
        print("No video pairs found!")
        return
    
    print(f"Found {len(video_pairs)} video pairs:")
    for pair in video_pairs:
        print(f"  - {pair['test_id']}")
    
    print("Creating HTML page...")
    create_html_page(video_pairs, output_file)
    
    print(f"✅ HTML page created successfully!")
    print(f"Open the file in your browser: {output_file}")

if __name__ == "__main__":
    main()
