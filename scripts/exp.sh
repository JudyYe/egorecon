[x] only dynamic object 
[x] only hand conditioned 
[] data cano??? data is somehow wrong? 
[] guidance 
[] noise eval
[] metadata

bowl: 194930206998778
spoon: 225397651484143


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