# TODO: 
# add hyra config
# estimate true noise distribution?
# add guided diffusion 

[x] try 3D points from monocular depth? 
# model architecture: multiple object traj?
# loss? 
# add geometry object feature
# more off-the-shelf 
# long-term inference
# enhanced hand feature
[x] move to Trainer? 




# data format: 
# "objects": ['uid1', 'uid2', ...], 
# "left_hand_theta": [T, 3+3+15], 3aa+3tsl+pcax15
# "left_hand_shape": [T, 10],
# "right_hand_theta": [T, 3+3+15],
# "right_hand_shape": [T, 10],
# "wTc": [T, 4, 4],
# intrinsic: [3, 3],
# "uid{i}_cTo_shelf": [T, 4, 4],
# "uid{i}_shelf_valid": [T, ],
# "uid{i}_wTo": [T, 4, 4],
# "uid{i}_gt_valid": [T, ],




t0: 300
t1: 540
seq: P0001_624f2ba9
bowl: 194930206998778
spoon: 225397651484143




194930206998778: bowl is not correctly registered in FP
96945373046044,253405647833885,225397651484143


python -m egorecon.training.trainer_proof_of_idea    \
 --demo_id P0001_624f2ba9           \
 --use_rerun        \
 --exp_name tmp \
 --target_object_id 96945373046044,253405647833885,225397651484143 --one_window  \
 --data_path /move/u/yufeiy2/data/HOT3D/pred_pose/mini_P0001_624f2ba9_3d.npz \
 --use_constant_noise 




python -m src.training.trainer_proof_of_idea    \
 --demo_id P0001_624f2ba9           \
 --use_rerun        \
 --exp_name ovft_one_multi_obj \
 --target_object_id 96945373046044,253405647833885,225397651484143 --one_window 

# train model 
python -m src.training.trainer_proof_of_idea   \
  --demo_id P0001_624f2ba9     \
  --target_object_id 225397651484143  \
    --use_rerun    \
    --exp_name ovft_one 





-
python src/training/trainer_hand_to_object_diffusion_overfit.py   \
  --demo_id P0001_624f2ba9     \
  --target_object_id 225397651484143  \
     --use_rerun    \
    --exp_name ovft_one   \
    --num_epochs 1_000_000 \







# overfit spoon
python src/training/trainer_hand_to_object_diffusion_overfit.py   \
  --demo_id P0001_624f2ba9     \
  --target_object_id 225397651484143  \
     --use_rerun    \
    --exp_name ovft_one   \
    --num_epochs 1_000_000 \

    --use_wandb \
