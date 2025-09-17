#!/usr/bin/env python3
"""
Script to check nominal frame rates from VRS files
"""
import os
import sys

# Add hot3d to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../hot3d/hot3d'))

try:
    from dataset_api import Hot3dDataProvider
    from data_loaders.loader_object_library import load_object_library
    from data_loaders.mano_layer import loadManoHandModel
    from projectaria_tools.core.stream_id import StreamId
except ImportError as e:
    print(f"Error importing HOT3D modules: {e}")
    sys.exit(1)

def check_nominal_rates(sequence_folder, object_library_folder):
    """Check nominal frame rates for different streams"""
    print(f"Checking nominal rates for: {sequence_folder}")
    print("=" * 60)
    
    # Load object library
    object_library = load_object_library(object_library_folderpath=object_library_folder)
    
    # Initialize data provider
    data_provider = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        mano_hand_model=None,
        fail_on_missing_data=False,
    )
    
    # Get device data provider
    device_provider = data_provider.device_data_provider
    
    # Check nominal rates for different streams
    print("Nominal frame rates for different streams:")
    print("-" * 40)
    
    # Common stream IDs
    stream_ids = [
        StreamId("214-1"),    # RGB camera (Aria)
        StreamId("1201-1"),  # SLAM-LEFT (both Aria and Quest)
        StreamId("1201-2"),  # SLAM-RIGHT (both Aria and Quest)
        StreamId("1202-1"),  # SLAM-LEFT (alternative)
        StreamId("1202-2"),  # SLAM-RIGHT (alternative)
    ]
    
    for stream_id in stream_ids:
        try:
            nominal_rate = device_provider.get_nominal_rate_hz(stream_id)
            print(f"Stream {stream_id}: {nominal_rate:.2f} Hz")
        except Exception as e:
            print(f"Stream {stream_id}: Not available ({e})")
    
    # Get all available stream IDs
    print("\nAll available streams:")
    print("-" * 40)
    try:
        all_stream_ids = device_provider.get_stream_ids()
        for stream_id in all_stream_ids:
            try:
                nominal_rate = device_provider.get_nominal_rate_hz(stream_id)
                print(f"Stream {stream_id}: {nominal_rate:.2f} Hz")
            except Exception as e:
                print(f"Stream {stream_id}: Error getting rate ({e})")
    except Exception as e:
        print(f"Error getting stream IDs: {e}")
    
    # Check device type
    print(f"\nDevice type: {data_provider.get_device_type()}")
    
    # Get sequence timestamps and analyze
    print("\nTimestamp analysis:")
    print("-" * 40)
    timestamps = device_provider.get_sequence_timestamps()
    print(f"Total timestamps: {len(timestamps)}")
    
    if len(timestamps) > 1:
        # Calculate time differences
        time_diffs = []
        for i in range(1, min(10, len(timestamps))):  # First 10 differences
            diff_ns = timestamps[i] - timestamps[i-1]
            time_diffs.append(diff_ns)
        
        # Convert to seconds and calculate FPS
        time_diffs_s = [diff / 1e9 for diff in time_diffs]
        fps_values = [1.0 / diff_s for diff_s in time_diffs_s]
        
        print(f"First 10 time differences (ns): {time_diffs}")
        print(f"First 10 time differences (s): {[f'{x:.6f}' for x in time_diffs_s]}")
        print(f"First 10 FPS values: {[f'{x:.3f}' for x in fps_values]}")
        print(f"Average FPS from first 10 frames: {sum(fps_values) / len(fps_values):.3f} Hz")

if __name__ == "__main__":
    # Check if we have a sequence to analyze
    dataset_dir = os.path.join(os.path.dirname(__file__), "../../hot3d/hot3d/dataset")
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found!")
        print("Please download HOT3D sequences first.")
        sys.exit(1)
    
    # Find first available sequence
    sequence_folders = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if (os.path.isdir(item_path) and 
            item != 'assets' and 
            not item.startswith('.')):
            sequence_folders.append(item_path)
    
    if not sequence_folders:
        print("No sequence folders found in dataset directory!")
        sys.exit(1)
    
    # Check first sequence
    first_sequence = sequence_folders[0]
    object_library = os.path.join(dataset_dir, "assets")
    
    check_nominal_rates(first_sequence, object_library)
