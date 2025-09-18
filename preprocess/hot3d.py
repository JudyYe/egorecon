import os.path as osp
import json
from glob import glob

root_dir = "/move/u/yufeiy2/data/HOT3D"
def make_split():
    # from hawor
    seqs = [
        "P0001_a68492d5", "P0001_9b6feab7", "P0014_8254f925", "P0011_76ea6d47", "P0014_84ea2dcc", "P0001_8d136980", "P0012_476bae57", "P0012_130a66e1", "P0014_24cb3bf0", "P0010_1c9fe708", "P0002_2ea9af5b", "P0011_11475e24", "P0010_0ecbf39f", "P0010_160e551c", "P0015_42b8b389", "P0012_915e71c6", "P0002_65085bfc", "P0011_47878e48", "P0011_cee8fe4f", "P0002_016222d1", "P0012_d85e10f6", "P0012_119de519", "P0010_41c4c626", "P0012_f7e3880b", "P0009_02511c2f", "P0011_72efb935", "P0010_924e574e",
    ]
    test_list = []

    # make sure they are all in the root_dir
    for seq in seqs:
        if not osp.exists(osp.join(root_dir, seq, 'metadata.json')):
            print(f"Seq {seq} not found")
            continue
        
        test_list.append(seq)
        with open(osp.join(root_dir, seq, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        if not metadata['have_hand_object_pose_gt']:
            print(f"Seq {seq} has no hand object pose gt")
    print('Total seqs:', len(seqs), f' --> {len(test_list)}')
    
    # list all seqs in the root_dir
    seq_list = glob(osp.join(root_dir, 'P*'))
    seq_list = [osp.basename(seq) for seq in seq_list]

    train_list = []
    for seq in seq_list:
        if seq in seqs:
            continue
        with open(osp.join(root_dir, seq, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        if not metadata['have_hand_object_pose_gt']:
            continue
    
        if metadata['headset'] == 'Aria':
            train_list.append(seq)
        
    print('Total train seqs:', len(train_list))


if __name__ == "__main__":
    make_split()