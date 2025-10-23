[] bpspoints
[] bps=2

[] off-the-shelf obs pipeline
[] long seq
[] set up quant


[x] change model | TODO
[x] norm in data, data done

[x] get a mini hot3d clip? 
[x] vis: contact, vis other motion 

[x] faster vis


[x] change coord frame

[x] why vis in guidance is wrong? 
[x] only dynamic object 
[x] only hand conditioned 
[x] data cano??? data is somehow wrong? 
[x] guidance 
[x] noise eval
[x] metadata

bowl: 194930206998778
spoon: 225397651484143

python -m egorecon.training.trainer_hoi  -m    \
  expname=fix_bps_overfit/beijing_bps\${condition.bps}_contact\${output.contact}_\${hand_rep}_w\${loss.w_contact}   \
  experiment=obj_only   \
  dyn_only=true   output.contact=true   \
  hand_rep=joint   \
  traindata=hotclip_mini   \
  condition.bps=2,1   \
  loss.w_contact=0.1,1,0 \
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
  ddim=true

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

python -m egorecon.manip.model.guidance_optimizer_jax
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