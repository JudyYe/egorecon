python preprocess/call_vlm_for_contact.py \
  --images_dir "data/HOT3D-CLIP/extract_images-rot90/clip-001911/" \
  --objects "milk bottle" \
  --prompt "For each listed object, determine whether the LEFT and RIGHT human hands are physically touching the object. 'Touching' means direct physical contact with the object, including grasping, holding, or pressing. If in contact, output 1; otherwise, output 0. Return only the contact labels according to the schema." \
  --model "gpt-4o" \
  --output contact_by_object.jsonl


[] relative contact: something is wrong for printing????? 
[] why guided last step != post first step? 


test hand 
GT mask 
[] change contact 

bowl: 194930206998778
spoon: 225397651484143



python -m egorecon.training.test_hoi  -m    expname=baselines/fp_hawor   dir=outputs/ready/ours/opt.yaml   \
  testdata=hotclip_train testdata.testsplit=test50obj     dyn_only=true   \
  fp_index
  test_folder=eval_test50obj_\${guide.hint} guide.hint=fp_simple   test_num=50    test_mode=fp \


python -m egorecon.training.test_hoi  -m    expname=baselines/fp_hawor_\${fp_index}   dir=outputs/ready/ours/opt.yaml   \
  testdata=hotclip_train testdata.testsplit=test50obj     dyn_only=true   \
  fp_index=foundation_pose_metric3d \
  test_folder=eval_\${fp_index}_test50obj_\${guide.hint} guide.hint=fp_full   test_num=50    test_mode=fp \





/move/u/yufeiy2/Package/blender-4.3.2-linux-x64/blender \
  --background \
  --python mayday/blender_vis.py -- \
  --mode render_assets \
  --asset-dir outputs/debug_blender/post_meshes/001874_000007 \
  --output-dir outputs/debug_blender/001874_000007 \
  --target-frame 0 \
  --allocentric-step 1 \
  --external-camera-location 1.5 -1.5 1.0 \
  --external-camera-target 0.0 0.0 0.3


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