#!/bin/bash

# Loop through id from 1 to 33
for id in {1..33}; do
    echo "Submitting job for segment id=$id"
    
    # id=${id} && python -m move_utils.slurm_wrapper --sl_name render${id} --slurm --sl_time_hr 10 --sl_ngpu 1 \
    id=${id} && bash mayday/blender_launcher.sh 15 \
    python -m mayday.blender_teaser build_teaser_video \
      --pred_dir outputs/ready/ours/teaser_hoi_better_segment/segment${id}/post \
      --bundle_root outputs/blender_results/teaser_segment/segment${id} \
      --image_folder video
    
    echo "Submitted job for segment id=$id"
done

echo "All jobs submitted!"

