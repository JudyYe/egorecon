preprocessing stage:
- get observation
  - 2D joints
  - get 3D template
- gsplat
- retrain diffusion with better contact loss, longer epoch



more guidance:

change condition: 
- change object template
- change hand traj 


guidance:
guide by different hand contact
guide by key-frame object pose? 


GOAL: 
[] optimize memory usage of guidance optimization: under 40G? 
[] soft contact for trainig to




python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  datasets.save_cache=false datasets.use_cache=false  \
  test_folder=app/mem_debug_\${guide.hint}  eval=mem_debug guide.num_target_points=100 test_num=1 \





python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  datasets.save_cache=false datasets.use_cache=false  \
  test_folder=app/\${guide.hint}_soft\${guide.use_contact_soft}_self\${guide.use_contact_self}  eval=hand_guide \
  guide.use_contact_soft=true guide.use_contact_self=true 



-

python -m mayday.blender_merge --mode 3way_teaser 




id=4 && bash mayday/blender_launcher.sh 10 \
python -m mayday.blender_teaser build_teaser_video   \
  --pred_dir outputs/ready/ours/teaser_hoi_better_segment/segment${id}/post \
  --bundle_root outputs/blender_results/teaser_segment/segment${id}     \
  --image_folder video




render sample 
python -m mayday.blender_wrapper --mode=render_sample

python -m move_utils.slurm_wrapper --sl_name ours --slurm --sl_time_hr 10 --sl_ngpu 1     \
bash mayday/blender_launcher.sh 15 \
python -m mayday.blender_wrapper     \
  --image_folder video  \
  --method_list '["ours"]'     \
  --render_video   \
  --render_cam_h 600 --skip 


python -m move_utils.slurm_wrapper --sl_name ours_no_obj --slurm --sl_time_hr 10 --sl_ngpu 1     \
bash mayday/blender_launcher.sh 10 \
python -m mayday.blender_wrapper     \
  --image_folder video_no_obj --no_obj  \
  --method_list '["ours"]'     \
  --render_video   \
  --render_cam_h 600 --skip


python -m move_utils.slurm_wrapper --sl_name gt_no_obj --slurm --sl_time_hr 10 --sl_ngpu 1     \
bash mayday/blender_launcher.sh 10 \
python -m mayday.blender_wrapper     \
  --image_folder video_no_obj --no_obj  \
  --method_list '["gt"]'     \
  --render_video   \
  --render_cam_h 600 --skip

python -m move_utils.slurm_wrapper --sl_name fp_simple_no_obj --slurm --sl_time_hr 10 --sl_ngpu 1     \
bash mayday/blender_launcher.sh 10 \
python -m mayday.blender_wrapper     \
  --image_folder video_no_obj --no_obj  \
  --method_list '["fp_simple"]'     \
  --render_video   \
  --render_cam_h 600 --skip

python -m move_utils.slurm_wrapper --sl_name fp_full_no_obj --slurm --sl_time_hr 10 --sl_ngpu 1     \
bash mayday/blender_launcher.sh 10 \
python -m mayday.blender_wrapper     \
  --image_folder video_no_obj --no_obj  \
  --method_list '["fp_full"]'     \
  --render_video   \
  --render_cam_h 600 --skip


no hand 

python -m move_utils.slurm_wrapper --sl_name fp --slurm --sl_time_hr 10 --sl_ngpu 1     \
bash mayday/blender_launcher.sh 5 \
python -m mayday.blender_wrapper     \
  --image_folder video_no_hand   --no_hand  \
  --method_list '["fp"]'     \
  --render_video   \
  --render_cam_h 600 --skip 

  
python -m move_utils.slurm_wrapper --sl_name gt --slurm --sl_time_hr 10 --sl_ngpu 1     \
python -m mayday.blender_wrapper     \
  --image_folder video_no_hand   --no_hand  \
  --method_list '["gt"]'     \
  --render_video   \
  --render_cam_h 600

python -m move_utils.slurm_wrapper --sl_name fp_simple --slurm --sl_time_hr 10 --sl_ngpu 1     \
python -m mayday.blender_wrapper     \
  --image_folder video_no_hand   --no_hand  \
  --method_list '["fp_simple"]'     \
  --render_video   \
  --render_cam_h 600 

python -m move_utils.slurm_wrapper --sl_name fp_full --slurm --sl_time_hr 10 --sl_ngpu 1     \
python -m mayday.blender_wrapper     \
  --image_folder video_no_hand   --no_hand  \
  --method_list '["fp_full"]'     \
  --render_video   \
  --render_cam_h 600 


python -m move_utils.slurm_wrapper --sl_name ours --slurm --sl_time_hr 10 --sl_ngpu 1     \
python -m mayday.blender_wrapper     \
  --image_folder video_no_hand   --no_hand  \
  --method_list '["ours"]'     \
  --render_video   \
  --render_cam_h 600  --skip


-

render comparison video from allocentric camera

python -m mayday.blender_wrapper   \
  --image_folder video_new_no_hand   \
  --method_list '["gt", "fp", "fp_simple", "fp_full", "ours"]'   \
  --render_video_alloc_joint --no_hand  \
  --divide 0.  \
  --render_width 1440 \
  --render_height 1080  \
  --dynamic_floor  \
  --skip &


python -m mayday.blender_wrapper   \
  --image_folder video_new   \
  --method_list '["gt", "fp", "fp_simple", "fp_full", "ours"]'   \
  --render_video_alloc_joint --no_hand  \
  --divide 0.  \
  --render_width 1440 \
  --render_height 1080  \
  --dynamic_floor  \
  --skip &


python -m move_utils.slurm_wrapper --slurm --sl_time_hr 5 --sl_ngpu 1    --sl_name our \
python -m mayday.blender_wrapper   \
  --image_folder video_new   \
  --method_list '["gt", "fp_simple", "fp_full", "ours"]'   \
  --render_video_alloc_joint   \
  --divide 0.5  \
  --render_width 1920 \
  --render_height 800  \
  --dynamic_floor  \
  --skip &


python -m move_utils.slurm_wrapper --slurm --sl_time_hr 5 --sl_ngpu 1    --sl_name our \
python -m mayday.blender_wrapper     \
  --image_folder video     \
  --method_list '["ours"]'     \
  --render_video   \
  --render_cam_h 600



python -m mayday.blender_wrapper     \
  --image_folder video     \
  --method_list '["gt"]'     \
  --render_video   \
  --render_cam_h 600


python -m move_utils.slurm_wrapper --slurm --sl_time_hr 5 --sl_ngpu 1    --sl_name fp_simple \
python -m mayday.blender_wrapper     \
  --image_folder video     \
  --method_list '["fp_simple"]'     \
  --render_video   \
  --render_cam_h 600

python -m move_utils.slurm_wrapper --slurm --sl_time_hr 5 --sl_ngpu 1    --sl_name fp_full \
python -m mayday.blender_wrapper     \
  --image_folder video     \
  --method_list '["fp_full"]'     \
  --render_video   \
  --render_cam_h 600 --skip 


python -m move_utils.slurm_wrapper --slurm --sl_time_hr 5 --sl_ngpu 1    --sl_name fp_full \
python -m mayday.blender_wrapper     \
  --image_folder video     \
  --method_list '["ours"]'     \
  --render_video   \
  --render_cam_h 600 --skip 


python -m mayday.blender_wrapper   \
  --image_folder video   \
  --method_list '["gt", "fp_simple", "fp_full", "ours"]'   \
  --render_video 



python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=teaser dyn_only=true \
  datasets.save_cache=false datasets.use_cache=false  \
  test_folder=teaser_\${guide.hint} eval=ddim_long_better  \  









  

python -m preprocess.vlm_ablation \
--num-samples 10 --force-gt --variants vanilla one_of_k one_of_k_visual full \
--output-dir outputs/vlm_ablation \
--gt-dir outputs/vlm_ablation/gt

python preprocess/vlm_ablation.py \
 --num-samples 2 \
 --variants vanilla one_of_k one_of_k_visual full


-
python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt eval=ddim_long \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  test_folder=rot_\${guide.hint}_vlm\${vlm}_skip\${contact_every}_\${post.rel_contact_weight}  \
  contact_every=10  vlm=true  post.rel_contact_weight=10 \
  +engine=move engine.exclude=move1+move2+humanoid1 engine.slurm_job_name=\${test_folder}



python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt eval=ddim_long_better \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  test_folder=rot_\${guide.hint}_vlm\${vlm}_skip\${contact_every}_\${inner.use_static}_\${inner.static_weight}  \
  contact_every=10  vlm=true inner.use_static=true inner.static_weight=0.1,0.01,1 \
  +engine=move engine.exclude=move1+move2+move3+humanoid1 engine.slurm_job_name=\${test_folder}
















-
python preprocess/call_vlm_for_contact.py \
  --images_dir "data/HOT3D-CLIP/extract_images-rot90/clip-001911/" \
  --objects "milk bottle" \
  --prompt "For each listed object, determine whether the LEFT and RIGHT human hands are physically touching the object. 'Touching' means direct physical contact with the object, including grasping, holding, or pressing. If in contact, output 1; otherwise, output 0. Return only the contact labels according to the schema." \
  --model "gpt-4o" \
  --output contact_by_object.jsonl


[] relative contact: something is wrong for printing????? 
[] why guided last step != post first step? 


python -m mayday.blender_teaser build_teaser_video   \
  --bundle_root outputs/blender_results/teaser/video1/     \
  --seq_objs 001907_000025,001908_000011,001909_000018       \
  --output_dir outputs/blender_results/teaser/video1/demo  \
  --image_folder video



teaser video
python -m mayday.blender_teaser build_teaser_video   --bundle_root outputs/blender_results/teaser/video1/     --seq_objs 001907_000025,001908_000011,001909_000018       --output_dir outputs/blender_results/teaser/video1/demo  --image_folder video

python -m mayday.blender_teaser   \
  --bundle_root outputs/blender_results/teaser/video1/   \
  --seq_objs 001907_000025,001908_000011,001909_000018   \
  --output_dir outputs/blender_results/teaser/video1   \
  --target_frames 0+30+50+149,0+10+20+149,0+20+30+40+50+60+149   \
  --spacing 0   --vis_obj_trail   --render_hand_per_method 1,0,0

generation 
python -m mayday.blender_wrapper \
  --image_folder sample \
  --method_list '["ours-gen-0", "ours-gen-1", "ours-gen-2", "ours-gen-3", "ours-gen-4"]' \
  --vis_contact \
  --allocentric_step 30 \
  --target_frames [] \
  --seq_obj 001896_000001 


compare 

python -m mayday.blender_wrapper \
  --image_folder image \
  --method_list '["gt", "fp_simple", "fp_full", "ours"]' \
  --allocentric_step 50 \
  --seq_obj 001896_000001 
-


python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=segment1 dyn_only=true \
  datasets.save_cache=false datasets.use_cache=false  \
  test_folder=teaser_\${guide.hint}_segment/\${testdata.testsplit} eval=ddim_long_better  \





# Option 1: Using bash brace expansion for range (segment1 to segment32)
python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=segment1,segment2,segment3,segment4,segment5,segment6,segment7,segment8,segment9,segment10,segment11,segment12,segment13,segment14,segment15,segment16,segment17,segment18,segment19,segment20,segment21,segment22,segment23,segment24,segment25,segment26,segment27,segment28,segment29,segment30,segment31,segment32 dyn_only=true \
  datasets.save_cache=false datasets.use_cache=false  \
  test_folder=teaser_\${guide.hint}_segment/\${testdata.testsplit} eval=ddim_long_better skip=true \
  +engine=move engine.exclude=move1+move2+humanoid1 engine.slurm_job_name=\${test_folder} engine.cpus_per_task=5 \

# Option 2: Using comma-separated explicit list (original)
# python -m egorecon.training.test_hoi  -m  \
#   expname=ready/ours \
#   ckpt_index=model-20.pt \
#   testdata=hotclip_train testdata.testsplit=segment2,segment3,segment4 dyn_only=true \
#   datasets.save_cache=false datasets.use_cache=false  \
#   test_folder=teaser_\${guide.hint}_\${testdata.testsplit} eval=ddim_long_better skip=true \
#   +engine=move engine.exclude=move1+move2+humanoid1 engine.slurm_job_name=\${test_folder} \

# Option 3: Using Python range in a sweep config (see below for config file example)

-

python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=teaser dyn_only=true \
  datasets.save_cache=false datasets.use_cache=false  \
  test_folder=teaser   \  


python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  datasets.save_cache=false datasets.use_cache=false datasets.slam=true \
  test_folder=slam   \



python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  test_folder=generation sample_num=5  \
  inner_guidance=false post_guidance=false \



test hand 
GT mask 
[] change contact 

bowl: 194930206998778
spoon: 225397651484143


python -m mayday.blender_cvt \
  --mode convert \
  --pkl_dir outputs/debug_blender/post/ \
  --output_dir outputs/debug_blender/post_meshes/ \
  

python -m egorecon.training.test_hoi  -m    expname=baselines/fp_hawor   dir=outputs/ready/ours/opt.yaml   \
  testdata=hotclip_train testdata.testsplit=test50obj     dyn_only=true   \
  fp_index
  test_folder=eval_test50obj_\${guide.hint} guide.hint=fp_simple   test_num=50    test_mode=fp \


python -m egorecon.training.test_hoi  -m    expname=baselines/fp_hawor_\${fp_index}   dir=outputs/ready/ours/opt.yaml   \
  testdata=hotclip_train testdata.testsplit=test50obj     dyn_only=true   \
  fp_index=foundation_pose_metric3d \
  test_folder=eval_\${fp_index}_test50obj_\${guide.hint} guide.hint=fp_full   test_num=50    test_mode=fp \




/move/u/yufeiy2/Package/blender-4.3.2-linux-x64/blender \
  --background --python \
  mayday/blender_vis.py -- --mode render_assets \
  --asset-dir outputs/debug_blender/post_meshes/001874_000007 \
  --output-dir outputs/debug_blender/001874_000007 \
  --target-frame 50 --allocentric-step 50 --hand-color blue1,blue2 \
  --object-color pink --render-camera 


-
python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  test_folder=vlm_test_\${guide.hint}_vlm\${vlm}_skip\${contact_every}  \
  contact_every=10  vlm=true guide.hint=no_contact,no_contact_static \


python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  test_folder=vlm_test_\${guide.hint}_vlm\${vlm}_skip\${contact_every}  \
  contact_every=10  vlm=true guide.hint=only_post inner_guidance=false \


python -m egorecon.training.test_hoi  -m  \
  expname=ready/ours \
  ckpt_index=model-20.pt \
  testdata=hotclip_train testdata.testsplit=test50obj dyn_only=true \
  test_folder=vlm_test_\${guide.hint}_vlm\${vlm}_skip\${contact_every} guide.hint=hoi_contact \
  contact_every=10  vlm=true,false \

-


test fp 
python -m egorecon.training.test_hoi  -m    expname=baselines/fp_hawor   dir=outputs/ready/ours/opt.yaml   \
  testdata=hotclip_train testdata.testsplit=test50obj     dyn_only=true   \
  test_folder=eval_test50obj_\${guide.hint} guide.hint=fp_simple   test_num=50    test_mode=fp \


python -m egorecon.training.test_hoi  -m    expname=baselines/fp_hawor   dir=outputs/ready/ours/opt.yaml   \
  testdata=hotclip_train testdata.testsplit=test50obj     dyn_only=true   \
  test_folder=eval_test50obj_\${guide.hint} guide.hint=fp_full   test_num=50    test_mode=fp \



python -m egorecon.training.test_hoi  -m  \
  expname=baselines/fp_hawor \
  dir=outputs/ready/ours/opt.yaml \
  testdata=hotclip_train \
  dyn_only=true \
  test_folder=eval_\${guide.hint} guide.hint=fp_full \
  test_num=50  \
  test_mode=fp 
  



python -m egorecon.training.test_hoi  -m  \
  expname=baselines/fp_hawor \
  dir=outputs/ready/ours/opt.yaml \
  testdata=hotclip_train \
  dyn_only=true \
  test_folder=eval_\${guide.hint} guide.hint=fp_simple \
  test_num=50  \
  test_mode=fp 
  


-

python -m eval.eval_joints   \
  --mode pose6d --skip_not_there True --split test50 \
  --pred_file outputs/first_frame_hoi/firstFalse_dynTrue_static100_contact_100_smoothness10/eval_hoi_contact_ddim_long_vis/post_objects.pkl



python -m egorecon.training.trainer_hoi  -m    \
  expname=first_frame_hoi/first\${condition.first_wTo}_dyn\${dyn_only}_static\${loss.w_static}_contact_\${loss.w_contact}_smoothness\${loss.w_smoothness} \
  experiment=noisy_hand   condition.first_wTo=false  \
  traindata=hotclip_train   \
  dyn_only=true \
  condition.bps=2   \
  general.wandb=true   \
  train.warmup=0 loss.w_consistency=1 loss.w_smoothness=10 loss.w_contact=100 loss.w_static=100 \
  general.vis_every=5000 general.eval_every=\${general.vis_every} general.save_and_sample_every=\${general.vis_every} \
  ckpt=outputs/first_frame/dynTrue/weights/model-20.pt \
  +engine=move




python -m egorecon.training.trainer_hoi  -m    \
  expname=first_frame_hoi/dyn\${dyn_only}_static\${loss.w_static}_contact\${loss.w_contact}_smoothness\${loss.w_smoothness} \
  experiment=noisy_hand   condition.first_wTo=true  \
  traindata=hotclip_train   \
  dyn_only=true \
  condition.bps=2   \
  general.wandb=true   \
  train.warmup=0 loss.w_consistency=1 loss.w_smoothness=10 loss.w_contact=100 loss.w_static=100 \
  general.vis_every=5000 general.eval_every=\${general.vis_every} general.save_and_sample_every=\${general.vis_every} \
  ckpt=outputs/first_frame/dynTrue/weights/model-20.pt \
  +engine=move



python -m egorecon.training.test_hoi  -m  \
  expname=first_frame_hoi/dynTrue_static100_contact100_smoothness10 \
  ckpt_index=model-20.pt \
  testdata=hotclip_train \
  dyn_only=true \
  test_folder=eval_\${guide.hint}_\${sample}_vis guide.hint=hoi_contact \
  test_num=50  \



python -m egorecon.training.test_hoi  -m  \
  expname=first_frame/dynFalse \
  ckpt_index=model-20.pt \
  testdata=hotclip_train \
  dyn_only=true \
  test_folder=eval_\${guide.hint}_\${sample}_vis guide.hint=hoi_contact \
  test_num=50  \



python -m egorecon.training.test_hoi  -m  \
  expname=first_frame/dynFalse \
  ckpt_index=model-20.pt \
  testdata=hotclip_train \
  dyn_only=true \
  test_folder=eval_\${guide.hint}_\${sample} guide.hint=hoi_contact \
  test_num=50  \




python -m egorecon.training.trainer_hoi  -m    \
  expname=first_frame/dyn\${dyn_only}_firstframe\${condition.first_wTo} \
  experiment=noisy_hand   \
  traindata=hotclip_train   \
  dyn_only=false,true \
  condition.bps=2   \
  general.wandb=true   \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10 loss.w_static=10 \
  condition.first_wTo=false \
  general.vis_every=5000 general.eval_every=\${general.vis_every} general.save_and_sample_every=\${general.vis_every} \
  ckpt=outputs/oracle_cond/bps2_False/weights/model-9.pt \
  +engine=move
  


python -m egorecon.training.trainer_hoi  -m    \
  expname=dev/tmp \
  experiment=noisy_hand   \
  traindata=hotclip_train   \
  dyn_only=false \
  condition.bps=2   \
  general.wandb=true   \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10 loss.w_static=10 \
  oracle_cond=true \
  condition.first_wTo=true \
  start_mask_step=.. 





python -m egorecon.training.test_hoi  -m  \
  expname=oracle_cond/bps2_False \
  ckpt_index=model-5.pt \
  guide.hint=hoi_contact \
  test_folder=eval_\${guide.hint}_\${sample}_lambda1 \
  testdata=hotclip_mini test_num=5 \




bug:
static loss
paddding mask ( validation)
can bps added to the front? 

python -m preprocess.est_noise \
  --est_dataset_name dataset_contact_patched_hawor_v2_camGT_gt

python -m preprocess.hot3dclip_extract \
  --shelf_name hawor_v2_camGT_gt \
  --orig_name dataset_contact \


python -m egorecon.training.trainer_hoi  -m    \
  expname=oracle_cond/bps\${condition.bps}_\${dyn_only} \
  experiment=noisy_hand   \
  traindata=hotclip_train   \
  dyn_only=false \
  condition.bps=2   \
  general.wandb=true   \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10 loss.w_static=10 \
  oracle_cond=true \


python -m egorecon.training.trainer_hoi  -m    \
  expname=oracle_cond/bps\${condition.bps}_\${dyn_only} \
  experiment=noisy_hand   \
  traindata=hotclip_train   \
  dyn_only=true,false \
  condition.bps=2   \
  general.wandb=true   \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10 loss.w_static=10 \
  oracle_cond=true \
  +engine=move
  



python -m egorecon.training.trainer_hoi  -m    \
  expname=noisy_hand_weight/hand_\${hand}_consist_w\${loss.w_consistency}_contact\${loss.w_contact}_\${loss.w_static} \
  experiment=noisy_hand   \
  traindata=hotclip_train   \
  condition.bps=2,1   \
  general.wandb=true   \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10,1 loss.w_static=10,1 \
  +engine=move engine.timeout_min=2880



python -m egorecon.training.trainer_hoi  -m    \
  expname=dev/tmp \
  experiment=noisy_hand   \
  traindata=hotclip_mini   \
  condition.bps=1   \
  loss.w_static=1 train.warmup=10 \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10 loss.w_static=1 \



python -m preprocess.est_noise

python -m egorecon.training.test_hoi  -m  \
  expname=noisy_hand/hand_cond_out_consist_w0.1_contact10_1_bps2 \
  ckpt_index=model-6.pt \
  guide.hint=hoi_contact \
  test_folder=eval_\${guide.hint}_\${sample}_test \
  testdata=hotclip_train test_num=5 \
  sample=ddim_long testdata=hotclip_mini datasets.window=150 datasets.use_cache=false datasets.save_cache=false \


-
python -m egorecon.training.trainer_hoi  -m    \
  expname=noisy_hand/hand_\${hand}_consist_w\${loss.w_consistency}_contact\${loss.w_contact}_\${loss.w_smoothness}_bps\${condition.bps}  \
  experiment=noisy_hand   \
  traindata=hotclip_train   \
  condition.bps=1,2   \
  general.wandb=true   \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10 loss.w_static=1 \
  +engine=move engine.timeout_min=2880


python -m egorecon.training.trainer_hoi  -m    \
  expname=noisy_hand/hand_\${hand}_consist_w\${loss.w_consistency}_contact\${loss.w_contact}_\${loss.w_smoothness}_bps\${condition.bps}  \
  experiment=noisy_hand   \
  traindata=hotclip_train   \
  condition.bps=1,2   \
  general.wandb=true   \
  +engine=move engine.timeout_min=2880



# one
python -m egorecon.training.trainer_hoi  -m    \
  expname=noisy_hand/hand_\${hand}_consist_w\${loss.w_consistency}_contact\${loss.w_contact}_\${loss.w_smoothness}  \
  experiment=obj_only   \
  traindata=hotclip_train   \
  condition.bps=2   \
  general.wandb=true   \
  loss.w_consistency=0.1 loss.w_smoothness=1 loss.w_contact=10 loss.w_static=1 \
  +engine=move engine.timeout_min=2880

python -m egorecon.training.trainer_hoi  -m    \
  expname=noisy_hand/hand_\${hand}_consist_w\${loss.w_consistency}_contact\${loss.w_contact}_\${loss.w_smoothness}  \
  experiment=obj_only   \
  traindata=hotclip_train   \
  condition.bps=2   \
  general.wandb=true   \
  +engine=move engine.timeout_min=2880

# stronger regu



python -m egorecon.training.trainer_hoi  -m    \
  expname=noisy_hand_overfit/beijing_noisy_hand_one_go_w\${loss.w_consistency}  \
  experiment=noisy_hand   \
  traindata=hotclip_mini   \
  condition.bps=2   \
  general.eval_every=1000 general.vis_every=1000 general.train_num_steps=10000   general.save_and_sample_every=\${general.vis_every} \
  general.wandb=true   \
  loss.w_consistency=0.1 \
  +engine=move

python -m egorecon.training.trainer_hoi  -m    \
  expname=fix_bps_overfit/bps\${condition.bps}_w\${loss.w_contact}_\${loss.w_rel_contact}_\${loss.w_smoothness}   \
  experiment=obj_only   \
  dyn_only=true   output.contact=true   \
  hand_rep=joint   \
  traindata=hotclip_mini   \
  condition.bps=2   \
  loss.w_contact=1 \
  loss.w_rel_contact=0.1,0,0.01 \
  loss.w_smoothness=0.1 \
  general.eval_every=2000 general.vis_every=2000 general.train_num_steps=50000   general.save_and_sample_every=\${general.vis_every} \
  general.wandb=true   \
  +engine=move


-

python -m egorecon.training.trainer_hoi  -m    \
  expname=fix_bps/bps\${condition.bps}_contact\${output.contact}_\${hand_rep}_w\${loss.w_contact}_\${loss.w_smoothness}_\${loss.w_rel_contact}   \
  experiment=obj_only   \
  dyn_only=true   output.contact=true   \
  hand_rep=joint   \
  condition.bps=2   \
  loss.w_contact=1 loss.w_smoothness=0.1 loss.w_rel_contact=0,0.01 \
  general.wandb=true   \
  +engine=move




-
python -m egorecon.training.trainer_hoi  -m    \
  expname=fix_bps_overfit/beijing_bps\${condition.bps}_contact\${output.contact}_\${hand_rep}_w\${loss.w_contact}   \
  experiment=obj_only   \
  dyn_only=true   output.contact=true   \
  hand_rep=joint   \
  traindata=hotclip_mini   \
  condition.bps=1   \
  loss.w_contact=0 \
  general.eval_every=2000 general.vis_every=2000 general.train_num_steps=50000   general.save_and_sample_every=\${general.vis_every} \
  general.rerun=true general.wandb=true   \
  +engine=move


python -m egorecon.training.trainer_hoi  -m    \
  expname=fix_bps_overfit/beijing_bps\${condition.bps}_contact\${output.contact}_\${hand_rep}_w\${loss.w_contact}   \
  experiment=obj_only   \
  dyn_only=true   output.contact=true   \
  hand_rep=joint   \
  traindata=hotclip_mini   \
  condition.bps=2,1   \
  loss.w_contact=10 \
  general.eval_every=2000 general.vis_every=2000 general.train_num_steps=50000   general.save_and_sample_every=\${general.vis_every} \
  general.rerun=true general.wandb=true   \
  +engine=move

python -m egorecon.training.trainer_hoi  -m  \
  expname=hoi_overfit/bps\${condition.bps}_contact\${output.contact}_\${hand_rep}_t\${bps_per_t_start} \
  experiment=obj_only \
  dyn_only=true \
  output.contact=true,false \
  hand_rep=joint \
  traindata=hotclip_mini \
  condition.bps=2 \
  general.rerun=true general.wandb=true \
  bps_per_t_start=200,600 \
  general.eval_every=500 general.vis_every=500 general.train_num_steps=10000 \
  +engine=move




python -m egorecon.training.trainer_hoi  -m  \
  expname=hoi_overfit/bps\${condition.bps}_contact\${output.contact}_\${hand_rep} \
  experiment=obj_only \
  dyn_only=true \
  output.contact=true,false \
  hand_rep=joint \
  traindata=hotclip_mini \
  condition.bps=1,2 \
  general.rerun=true general.wandb=true \
  general.eval_every=500 general.vis_every=500 general.train_num_steps=10000 \
  +engine=move

  general.rerun=true general.wandb=true \
  +engine=move



-=
python -m egorecon.training.test_hoi  -m  \
  expname=hoi/contactTrue_theta \
  ckpt_index=model-3.pt \
  ddim=true \
  ckpt=\${exp_dir}/weights/model-19.pt \
  test_folder=eval_\${guide.hint}_ddim\${ddim}




-

python -m egorecon.training.trainer_hoi  -m  \
  expname=overfit_hoi/contact\${output.contact}_\${hand_rep} \
  experiment=hoi \
  dyn_only=true \
  output.contact=true \
  hand_rep=joint \
  general.rerun=true general.wandb=true \
  general.eval_every=1000 general.vis_every=1000 \
  traindata=hotclip_mini 



--------------------------------
python -m egorecon.training.trainer_hoi  -m  \
  expname=dev/contact\${output.contact}_\${hand_rep}_obj2 \
  experiment=obj_only \
  dyn_only=true \
  output.contact=true \
  hand_rep=joint \
  general.rerun=true general.wandb=true \
  +engine=move





python -m egorecon.training.trainer_hoi  -m  \
  expname=hoi/contact\${output.contact}_\${hand_rep}_obj2 \
  experiment=obj_only \
  dyn_only=true \
  output.contact=true \
  hand_rep=joint \
  general.rerun=true general.wandb=true \



python -m egorecon.training.trainer_hoi  -m  \
  expname=hoi/contact\${output.contact}_\${hand_rep}_obj \
  experiment=obj_only \
  dyn_only=true \
  output.contact=true \
  hand_rep=joint \
  general.rerun=true general.wandb=true \
  +engine=move


python -m egorecon.training.trainer_hoi  -m  \
  expname=hoi/contact\${output.contact}_\${hand_rep} \
  experiment=hoi \
  dyn_only=true \
  output.contact=false,true \
  hand_rep=theta,joint \
  general.rerun=true general.wandb=true \
  +engine=move


python -m egorecon.training.trainer_hoi  -m  \
  expname=hoi/contact\${output.contact}_\${hand_rep} \
  experiment=hoi \
  dyn_only=true \
  output.contact=false \
  hand_rep=joint \
  general.rerun=true general.wandb=true \



  traindata=hotclip_mini 




python -m egorecon.manip.data.hand2obj_w_geom_motion \
  experiment=hoi \
  output.contact=false hand_rep=theta hand=out traindata=hot3d_mini datasets.split=mini 

python -m egorecon.manip.model.guidance_optimizer_hoi_jax
7it / s  -> 5it? 

python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=clip_bps123/bps\${condition.bps}_motion\${dyn_only}_\${datasets.augument.motion_threshold} \
  dyn_only=true \
  condition.bps=2 condition.noisy_obj=false  \
  traindata=hotclip_train \
  test=true guide.hint=reproj_cd ddim=True \
  ckpt=\${exp_dir}/weights/model-19.pt \
  test_folder=eval_\${guide.hint}_ddim\${ddim}

  general.rerun=true general.wandb=true \

python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=geom/bps\${condition.bps}_noisy_obj\${condition.noisy_obj} \
  experiment=obj_only \
  dyn_only=true \
  condition.bps=true condition.noisy_obj=false  \
  test=true \
  ckpt=outputs/geom/bpsTrue_noisy_objFalse/weights/model-5.pt \
  general.rerun=True general.wandb=false \
  guide.hint=com \


python -m egorecon.manip.model.guidance_optimizer_jax --mode reproj_cd
-
python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=clip_bps123/bps\${condition.bps}_motion\${dyn_only}_\${datasets.augument.motion_threshold} \
  dyn_only=true \
  condition.bps=0,1,2 condition.noisy_obj=false  \
  traindata=hotclip_train \
  general.rerun=true general.wandb=true \
  +engine=move


python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=dev/tmp \
  experiment=obj_only \
  dyn_only=true \
  condition.bps=true condition.noisy_obj=false  \
  traindata=hot3d_mini datasets.split=mini \
  datasets.augument.motion_threshold=0.1 \

-
python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=geom_clip/bps\${condition.bps}_noisy_obj\${condition.noisy_obj}_motion\${dyn_only}_\${datasets.augument.motion_threshold} \
  dyn_only=true \
  condition.bps=true condition.noisy_obj=false  \
  traindata=hotclip_train \
  ckpt=outputs/\${expname}/weights/model-5.pt \
  test=true datasets.split=test \
  general.rerun=true general.wandb=true \
  +engine=move



python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=geom/bps\${condition.bps}_noisy_obj\${condition.noisy_obj} \
  experiment=obj_only \
  dyn_only=true \
  condition.bps=true condition.noisy_obj=false  \
  test=true \
  ckpt=outputs/geom/bpsTrue_noisy_objFalse/weights/model-5.pt \
  general.rerun=True general.wandb=false \
  guide.hint=com \
  +engine=move


-
python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=geom/bps\${condition.bps}_noisy_obj\${condition.noisy_obj}_motion\${datasets.augument.motion_threshold} \
  experiment=obj_only \
  dyn_only=true \
  condition.bps=true condition.noisy_obj=false  \
  general.rerun=True general.wandb=true \
  datasets.augument.motion_threshold=0.1,0.2,0.5 \
  +engine=move





python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=geom/bps\${condition.bps}_noisy_obj\${condition.noisy_obj} \
  experiment=obj_only \
  dyn_only=true \
  condition.bps=true,false condition.noisy_obj=false,true  \
  general.rerun=True general.wandb=true \
  +engine=move




python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=dev/tmp \
  experiment=obj_only \
  dyn_only=true condition.noisy_obj=false \
  condition.bps=true \
  general.rerun=True general.wandb=true \


# refinement later



python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=dynamic_obj2/better_noise_dyn_on\${dyn_only}_cond_noisy_obj\${condition.noisy_obj} \
  experiment=obj_only \
  dyn_only=true,false condition.noisy_obj=true \
  general.rerun=True general.wandb=true \
  +engine=move




python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=dynamic_obj2/dyn_on\${dyn_only}_cond_noisy_obj\${condition.noisy_obj} \
  experiment=obj_only \
  dyn_only=true,false condition.noisy_obj=false,true \
  general.rerun=True general.wandb=true \
  +engine=move



python -m egorecon.manip.data.hand_to_object_dataset




python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=dynamic_obj/dyn_on\${dyn_only}_cond_noisy_obj\${condition.noisy_obj} \
  experiment=obj_only \
  dyn_only=true condition.noisy_obj=false \
  ckpt=
  +engine=move




python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=dev/test_guide \
  experiment=obj_only \
  dyn_only=true condition.noisy_obj=false \
  +engine=move


python -m egorecon.training.trainer_proof_of_idea  -m  \
  expname=dynamic_obj/dyn_on\${dyn_only}_cond_noisy_obj\${condition.noisy_obj} \
  experiment=obj_only \
  dyn_only=true,false condition.noisy_obj=false \
  general.rerun=True general.wandb=true \
  +engine=move



-
python -m egorecon.training.trainer_proof_of_idea    \
  expname=dev/tmp_test_meta \
  experiment=init_large \
  datasets.split=trainone \


-
python -m egorecon.manip.data.hand_to_object_dataset \
  datasets.split=train 

python -m egorecon.training.trainer_proof_of_idea    \
  expname=seq/\${datasets.split} \
  experiment=init_large \
  datasets.split=trainone \



python -m egorecon.training.trainer_proof_of_idea    \
  expname=seq_go/\${datasets.split} \
  experiment=init_large \
  datasets.split=testone \
  general.rerun=True general.wandb=true \

python -m  scripts.view_foundation_pose --num 540



python -m egorecon.training.trainer_proof_of_idea    \
 expname=dev/tmp \
 general.rerun=True        \
 data.demo_id=P0001_624f2ba9   data.target_object_id=96945373046044+253405647833885+225397651484143  \
 datasets.one_window=True  \
 datasets.augument.use_constant_noise=True 


# overfit spoon
python src/training/trainer_hand_to_object_diffusion_overfit.py   \
  --demo_id P0001_624f2ba9     \
  --target_object_id 225397651484143  \
     --use_rerun    \
    --exp_name overfit_spoon_P0001_624f2ba9   \
    --use_wandb \